# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Remote compiled function.

Memref marshalling is done by:
  1. Allocating a remote data buffer for each memref's data section
  2. Copying the host's bytes into that remote buffer
  3. Constructing a memref descriptor whose pointers reference the remote buffer
  4. Invoking the kernel with those descriptor addresses
  5. Reading the data sections back to the host descriptors after the call returns
"""
import ctypes
import logging
import os
import pathlib
import platform
import struct
from typing import List, Optional

from catalyst.utils.runtime_environment import get_lib_path

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _find_driver_lib() -> Optional[str]:
    """Locate the libcatalyst_remote_driver.{so,dylib} library."""
    env = os.environ.get("CATALYST_REMOTE_DRIVER")
    if env and pathlib.Path(env).is_file():
        return env

    try:
        rt_lib_dir = get_lib_path("runtime", "RUNTIME_LIB_DIR")
        ext = ".so" if platform.system() == "Linux" else ".dylib"
        candidate = pathlib.Path(rt_lib_dir) / f"libcatalyst_remote_driver{ext}"
        if candidate.is_file():
            return str(candidate)
    except Exception:  # pylint: disable=broad-except
        pass

    return None


_RemoteSession = ctypes.c_void_p


def _bind(lib):
    """Apply ctypes signatures to the C abis"""

    lib.catalyst_remote_open.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.catalyst_remote_open.restype = _RemoteSession

    lib.catalyst_remote_close.argtypes = [_RemoteSession]
    lib.catalyst_remote_close.restype = None

    lib.catalyst_remote_lookup.argtypes = [
        _RemoteSession,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.catalyst_remote_lookup.restype = ctypes.c_int

    lib.catalyst_remote_alloc.argtypes = [
        _RemoteSession,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.catalyst_remote_alloc.restype = ctypes.c_int

    lib.catalyst_remote_free.argtypes = [
        _RemoteSession,
        ctypes.c_uint64,
        ctypes.c_size_t,
    ]
    lib.catalyst_remote_free.restype = ctypes.c_int

    lib.catalyst_remote_write.argtypes = [
        _RemoteSession,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.catalyst_remote_write.restype = ctypes.c_int

    lib.catalyst_remote_read.argtypes = [
        _RemoteSession,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.catalyst_remote_read.restype = ctypes.c_int

    lib.catalyst_remote_run_as_main.argtypes = [
        _RemoteSession,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.catalyst_remote_run_as_main.restype = ctypes.c_int

    lib.catalyst_remote_invoke_pyface.argtypes = [
        _RemoteSession,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int,
    ]
    lib.catalyst_remote_invoke_pyface.restype = ctypes.c_int

    return lib


class RemoteSharedObjectManager:
    """Remote shared object manager.

    Args:
        kernel_path (str): path to the kernel object file
        func_name (str): name of the function
        remote (str): remote address of the function

    Raises:
        RuntimeError: if the catalyst-runtime-driver library is not found
    """

    def __init__(self, kernel_path: str, func_name: str, remote: str):
        self.kernel_path = kernel_path
        self.func_name = func_name
        self.remote = remote

        lib_path = _find_driver_lib()
        if lib_path is None:
            raise RuntimeError(
                "catalyst-runtime-driver not found. Build catalyst's runtime "
                "with -DENABLE_REMOTE_DRIVER=ON"
            )
        self._lib = _bind(ctypes.CDLL(lib_path))
        self._sess = None

        self._addr_setup: Optional[int] = None
        self._addr_teardown: Optional[int] = None
        self._addr_pyface: Optional[int] = None

        self.function: Optional[int] = None
        self.setup = None
        self.teardown = None
        self.mem_transfer = None

        self._kept_buffers: list = []

        self.open()

    def open(self):
        """Open the shared object and load symbols."""
        err = ctypes.create_string_buffer(512)
        sess = self._lib.catalyst_remote_open(
            self.kernel_path.encode(), self.remote.encode(), err, len(err)
        )
        if not sess:
            raise RuntimeError(
                f"catalyst_remote_open({self.kernel_path}, {self.remote}) "
                f"failed: {err.value.decode(errors='replace')}"
            )
        self._sess = sess

        self._addr_setup = self._lookup("setup")
        self._addr_teardown = self._lookup("teardown")
        self._addr_pyface = self._lookup("_catalyst_pyface_" + self.func_name)
        self.function = self._addr_pyface

    def close(self):
        """Close the shared object and release all remote allocated resources."""
        if self._sess is not None:
            self._lib.catalyst_remote_close(self._sess)
            self._sess = None

    def _lookup(self, name: str) -> int:
        """Lookup a symbol in the shared object."""
        out = ctypes.c_uint64(0)
        rc = self._lib.catalyst_remote_lookup(self._sess, name.encode(), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(f"catalyst_remote_lookup({name!r}) failed (rc={rc})")
        return out.value

    def _alloc(self, size: int) -> int:
        """Allocate memory on the remote."""
        out = ctypes.c_uint64(0)
        rc = self._lib.catalyst_remote_alloc(self._sess, size, ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(f"catalyst_remote_alloc({size}) failed (rc={rc})")
        return out.value

    def _free(self, addr: int, size: int):
        """Free memory on the remote."""
        self._lib.catalyst_remote_free(self._sess, addr, size)

    def _write(self, addr: int, data: bytes):
        """Write data to the remote."""
        if not data:
            return
        buf = (ctypes.c_char * len(data)).from_buffer_copy(data)
        rc = self._lib.catalyst_remote_write(
            self._sess, addr, ctypes.cast(buf, ctypes.c_void_p), len(data)
        )
        if rc != 0:
            raise RuntimeError(
                f"catalyst_remote_write(addr=0x{addr:x}, size={len(data)}) " f"failed (rc={rc})"
            )

    def _read(self, addr: int, size: int) -> bytes:
        """Read data from the remote."""
        buf = (ctypes.c_char * size)()
        rc = self._lib.catalyst_remote_read(
            self._sess, addr, ctypes.cast(buf, ctypes.c_void_p), size
        )
        if rc != 0:
            raise RuntimeError(
                f"catalyst_remote_read(addr=0x{addr:x}, size={size}) " f"failed (rc={rc})"
            )
        return bytes(buf)

    def _run_as_main(self, addr: int, argv_strs):
        """Run a function as main."""
        argv_bytes = [s.encode() for s in argv_strs]
        argv_arr = (ctypes.c_char_p * len(argv_bytes))(*argv_bytes)
        out_rc = ctypes.c_int32(0)
        rc = self._lib.catalyst_remote_run_as_main(
            self._sess,
            ctypes.c_uint64(addr),
            ctypes.c_int(len(argv_bytes)),
            argv_arr,
            ctypes.byref(out_rc),
        )
        if rc != 0:
            raise RuntimeError(f"catalyst_remote_run_as_main(addr=0x{addr:x}) failed (rc={rc})")
        return out_rc.value

    def _invoke_pyface(self, entry_addr: int, arg_addrs: List[int]):
        """Invoke a function with a list of argument addresses."""
        n = len(arg_addrs)
        arr_t = ctypes.c_uint64 * max(n, 1)
        arr = arr_t(*arg_addrs)
        rc = self._lib.catalyst_remote_invoke_pyface(
            self._sess, ctypes.c_uint64(entry_addr), arr, ctypes.c_int(n)
        )
        if rc != 0:
            raise RuntimeError(
                f"catalyst_remote_invoke_pyface(entry=0x{entry_addr:x}, "
                f"n_args={n}) failed (rc={rc})"
            )

    def _shape_size_bytes(self, memref) -> int:
        """Compute the data-buffer size (in bytes) of an MLIR memref descriptor instance."""
        aligned_field_t = self._field_type(memref, "aligned")
        elem_t = getattr(aligned_field_t, "_type_", None)
        if elem_t is None:
            # Fall back??: assume 8-byte elements
            elem_size = 8
        else:
            elem_size = ctypes.sizeof(elem_t)
        n = 1
        if hasattr(memref, "shape"):
            for d in memref.shape:
                n *= int(d)
        return elem_size * n

    @staticmethod
    def _field_type(struct, name):
        """Get the ctypes type for the given field on a Structure instance."""
        for n, t in type(struct)._fields_:
            if n == name:
                return t
        raise KeyError(name)

    @staticmethod
    def _set_pointer_field(struct, name, addr):
        """Set the given field on a Structure instance to the given address."""
        field_t = next(t for n, t in type(struct)._fields_ if n == name)
        try:
            setattr(struct, name, ctypes.cast(addr, field_t))
        except TypeError:
            setattr(struct, name, addr)

    @staticmethod
    def _get_pointer_field_int(struct, name) -> int:
        """Get the given field on a Structure instance as a raw integer address."""
        val = getattr(struct, name)
        try:
            return int(ctypes.cast(val, ctypes.c_void_p).value or 0)
        except (TypeError, AttributeError):
            return int(val) if val else 0

    _PTR_SIZE = ctypes.sizeof(ctypes.c_void_p)

    def _patch_desc_pointers(self, desc_bytes: bytearray, addr: int) -> None:
        """Patch the given address into the given bytearray."""
        # two QWORDs: allocated and aligned
        struct.pack_into("<QQ", desc_bytes, 0, addr, addr)

    def call(self, args: tuple) -> None:
        """Invoke the kernel remotely."""
        cleanup: List[tuple] = []
        try:
            rv_remote, rv_struct, rv_size = self._marshal_return(args[0], cleanup)
            av_remote = self._marshal_args(args[1], cleanup)

            self._invoke_pyface(self._addr_pyface, [rv_remote, av_remote])

            if rv_struct is not None:
                self._unmarshal_return(rv_struct, rv_remote, rv_size)
        finally:
            for addr, size in cleanup:
                if addr:
                    self._free(addr, size)

    def _marshal_return(self, rv_pointer, cleanup):
        """Marshal the return value."""
        if not rv_pointer:
            return 0, None, 0
        rv_struct = rv_pointer.contents
        rv_size = ctypes.sizeof(type(rv_struct))
        rv_bytes = bytearray(bytes(rv_struct))

        # Zero allocated + aligned (offset 0..15) for each inline memref
        offset = 0
        for _field_name, field_class in type(rv_struct)._fields_:
            struct.pack_into("<QQ", rv_bytes, offset, 0, 0)
            offset += ctypes.sizeof(field_class)

        rv_remote = self._alloc(rv_size)
        cleanup.append((rv_remote, rv_size))
        self._write(rv_remote, bytes(rv_bytes))
        return rv_remote, rv_struct, rv_size

    def _unmarshal_return(self, rv_struct, rv_remote: int, rv_size: int):
        """Copy the updated rv_struct back into the host memref descriptors."""
        updated = self._read(rv_remote, rv_size)
        ctypes.memmove(ctypes.byref(rv_struct), updated, rv_size)

        for field_name, _field_class in type(rv_struct)._fields_:
            memref = getattr(rv_struct, field_name)
            data_bytes = self._shape_size_bytes(memref)
            remote_aligned = self._get_pointer_field_int(memref, "aligned")

            host_buf = (ctypes.c_uint8 * max(data_bytes, 1))()
            self._kept_buffers.append(host_buf)
            host_addr = ctypes.addressof(host_buf)

            if data_bytes and remote_aligned:
                blob = self._read(remote_aligned, data_bytes)
                ctypes.memmove(host_buf, blob, data_bytes)

            self._set_pointer_field(memref, "allocated", host_addr)
            self._set_pointer_field(memref, "aligned", host_addr)

    def _marshal_args(self, av_pointer, cleanup) -> int:
        """Build the remote ArgValueStruct."""
        if not av_pointer:
            return 0
        try:
            av_struct = av_pointer.contents
        except (AttributeError, ValueError):
            return 0

        av_bytes = bytearray(bytes(av_struct))
        av_size = ctypes.sizeof(type(av_struct))

        offset = 0
        for field_name, field_class in type(av_struct)._fields_:
            field_size = ctypes.sizeof(field_class)
            memref_ptr = getattr(av_struct, field_name)

            # Skip empty pointer fields
            if not memref_ptr:
                offset += field_size
                continue

            memref = memref_ptr.contents
            data_bytes = self._shape_size_bytes(memref)

            remote_data = 0
            if data_bytes:
                remote_data = self._alloc(data_bytes)
                cleanup.append((remote_data, data_bytes))
                # Copy host data buffer into the remote allocation.
                host_data = ctypes.string_at(memref.aligned, data_bytes)
                self._write(remote_data, host_data)

            # Build the remote memref descriptor
            desc_bytes = bytearray(bytes(memref))
            self._patch_desc_pointers(desc_bytes, remote_data)

            remote_desc = self._alloc(len(desc_bytes))
            cleanup.append((remote_desc, len(desc_bytes)))
            self._write(remote_desc, bytes(desc_bytes))

            # Patch the corresponding field in av_bytes to point to the
            # remote descriptor.
            struct.pack_into("<Q", av_bytes, offset, remote_desc)
            offset += field_size

        av_remote = self._alloc(av_size)
        cleanup.append((av_remote, av_size))
        self._write(av_remote, bytes(av_bytes))
        return av_remote

    def __enter__(self):
        self._run_as_main(self._addr_setup, ["jitted-function"])
        return self

    def __exit__(self, _type, _value, _traceback):
        try:
            self._run_as_main(self._addr_teardown, [])
        finally:
            self.close()
