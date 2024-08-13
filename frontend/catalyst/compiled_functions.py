# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains classes to manage compiled functions and their underlying resources."""

import ctypes
import logging
from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp
import numpy as np
from jax.interpreters import mlir
from jax.tree_util import PyTreeDef, tree_flatten, tree_unflatten
from mlir_quantum.runtime import (
    as_ctype,
    make_nd_memref_descriptor,
    make_zero_d_memref_descriptor,
)

from catalyst.jax_extras import get_implicit_and_explicit_flat_args
from catalyst.logging import debug_logger_init
from catalyst.tracing.type_signatures import (
    TypeCompatibility,
    filter_static_args,
    get_decomposed_signature,
    typecheck_signatures,
)
from catalyst.utils import wrapper
from catalyst.utils.c_template import get_template, mlir_type_to_numpy_type
from catalyst.utils.filesystem import Directory
from catalyst.utils.jnp_to_memref import get_ranked_memref_descriptor

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SharedObjectManager:
    """Shared object manager.

    Manages the life time of the shared object. When is it loaded, when to close it.

    Args:
        shared_object_file (str): path to shared object containing compiled function
        func_name (str): name of compiled function
    """

    @debug_logger_init
    def __init__(self, shared_object_file, func_name):
        self.shared_object_file = shared_object_file
        self.shared_object = None
        self.func_name = func_name
        self.function = None
        self.setup = None
        self.teardown = None
        self.mem_transfer = None
        self.open()

    def open(self):
        """Open the sharead object and load symbols."""
        self.shared_object = ctypes.CDLL(self.shared_object_file)
        self.function, self.setup, self.teardown, self.mem_transfer = self.load_symbols()

    def close(self):
        """Close the shared object"""
        self.function = None
        self.setup = None
        self.teardown = None
        self.mem_transfer = None
        dlclose = ctypes.CDLL(None).dlclose
        dlclose.argtypes = [ctypes.c_void_p]
        # pylint: disable=protected-access
        dlclose(self.shared_object._handle)

    def load_symbols(self):
        """Load symbols necessary for for execution of the compiled function.

        Returns:
            CFuncPtr: handle to the main function of the program
            CFuncPtr: handle to the setup function, which initializes the device
            CFuncPtr: handle to the teardown function, which tears down the device
            CFuncPtr: handle to the memory transfer function for program results
        """

        setup = self.shared_object.setup
        setup.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        setup.restypes = ctypes.c_int

        teardown = self.shared_object.teardown
        teardown.argtypes = None
        teardown.restypes = None

        # We are calling the c-interface
        function = self.shared_object["_catalyst_pyface_" + self.func_name]
        # Guaranteed from _mlir_ciface specification
        function.restypes = None
        # Not needed, computed from the arguments.
        # function.argyptes

        mem_transfer = self.shared_object["_mlir_memory_transfer"]

        return function, setup, teardown, mem_transfer

    def __enter__(self):
        params_to_setup = [b"jitted-function"]
        argc = len(params_to_setup)
        array_of_char_ptrs = (ctypes.c_char_p * len(params_to_setup))()
        array_of_char_ptrs[:] = params_to_setup
        self.setup(ctypes.c_int(argc), array_of_char_ptrs)
        return self

    def __exit__(self, _type, _value, _traceback):
        self.teardown()


class CompiledFunction:
    """Manages the compilation result of a user program. Holds a reference to the binary object and
    performs necessary processing to invoke the compiled program.

    Args:
        shared_object_file (str): path to shared object containing compiled function
        func_name (str): name of compiled function
        restype (Iterable): MLIR tensor types representing the result of the compiled function
        compile_options (CompileOptions): compilation options used
    """

    @debug_logger_init
    def __init__(
        self, shared_object_file, func_name, restype, out_type, compile_options
    ):  # pylint: disable=too-many-arguments
        self.shared_object = SharedObjectManager(shared_object_file, func_name)
        self.compile_options = compile_options
        self.return_type_c_abi = None
        self.func_name = func_name
        self.restype = restype
        self.out_type = out_type

    @staticmethod
    def _exec(shared_object, has_return, out_type, numpy_dict, *args):
        """Execute the compiled function with arguments ``*args``.

        Args:
            lib: Shared object
            has_return: whether the function returns a value or not
            out_type: Jaxpr output type holding information about implicit outputs
            numpy_dict: dictionary of numpy arrays of buffers from the runtime
            *args: arguments to the function

        Returns:
            the return values computed by the function or None if the function has no results
        """

        with shared_object as lib:
            result_desc = type(args[0].contents) if has_return else None
            retval = wrapper.wrap(lib.function, args, result_desc, lib.mem_transfer, numpy_dict)

        if out_type is not None:
            keep_outputs = [k for _, k in out_type]
            retval = [r for (k, r) in zip(keep_outputs, retval) if k]

        retval = [jnp.asarray(arr) for arr in retval]
        return retval

    @staticmethod
    def get_ranked_memref_descriptor_from_mlir_tensor_type(mlir_tensor_type):
        """Convert an MLIR tensor type to a memref descriptor.

        Args:
            mlir_tensor_type: an MLIR tensor type
        Returns:
            a memref descriptor with empty data
        """
        assert mlir_tensor_type
        assert mlir_tensor_type is not tuple
        shape = mlir.ir.RankedTensorType(mlir_tensor_type).shape
        mlir_element_type = mlir.ir.RankedTensorType(mlir_tensor_type).element_type
        numpy_element_type = mlir_type_to_numpy_type(mlir_element_type)
        ctp = as_ctype(numpy_element_type)
        if shape:
            memref_descriptor = make_nd_memref_descriptor(len(shape), ctp)()
        else:
            memref_descriptor = make_zero_d_memref_descriptor(ctp)()

        return memref_descriptor

    @staticmethod
    def get_etypes(mlir_tensor_type):
        """Get element type for an MLIR tensor type."""
        mlir_element_type = mlir.ir.RankedTensorType(mlir_tensor_type).element_type
        return mlir_type_to_numpy_type(mlir_element_type)

    @staticmethod
    def get_sizes(mlir_tensor_type):
        """Get element type size for an MLIR tensor type."""
        mlir_element_type = mlir.ir.RankedTensorType(mlir_tensor_type).element_type
        numpy_type = mlir_type_to_numpy_type(mlir_element_type)
        dtype = np.dtype(numpy_type)
        return dtype.itemsize

    @staticmethod
    def get_ranks(mlir_tensor_type):
        """Get rank for an MLIR tensor type."""
        shape = mlir.ir.RankedTensorType(mlir_tensor_type).shape
        return len(shape) if shape else 0

    def getCompiledReturnValueType(self, mlir_tensor_types):
        """Compute the type for the return value and memoize it

        This type does not need to be recomputed as it is generated once per compiled function.
        Args:
            mlir_tensor_types: a list of MLIR tensor types which match the expected return type
        Returns:
            a pointer to a CompiledFunctionReturnValue, which corresponds to a structure in which
            fields match the expected return types
        """

        if self.return_type_c_abi is not None:
            return self.return_type_c_abi

        error_msg = """This function must be called with a non-zero length list as an argument."""
        assert mlir_tensor_types, error_msg
        _get_rmd = CompiledFunction.get_ranked_memref_descriptor_from_mlir_tensor_type
        return_fields_types = [_get_rmd(mlir_tensor_type) for mlir_tensor_type in mlir_tensor_types]
        ranks = [
            CompiledFunction.get_ranks(mlir_tensor_type) for mlir_tensor_type in mlir_tensor_types
        ]

        etypes = [
            CompiledFunction.get_etypes(mlir_tensor_type) for mlir_tensor_type in mlir_tensor_types
        ]

        sizes = [
            CompiledFunction.get_sizes(mlir_tensor_type) for mlir_tensor_type in mlir_tensor_types
        ]

        class CompiledFunctionReturnValue(ctypes.Structure):
            """Programmatically create a structure which holds tensors of varying base types."""

            _fields_ = [("f" + str(i), type(t)) for i, t in enumerate(return_fields_types)]
            _ranks_ = ranks
            _etypes_ = etypes
            _sizes_ = sizes

        return_value = CompiledFunctionReturnValue()
        return_value_pointer = ctypes.pointer(return_value)
        self.return_type_c_abi = return_value_pointer
        return self.return_type_c_abi

    def restype_to_memref_descs(self, mlir_tensor_types):
        """Converts the return type to a compatible type for the expected ABI.

        Args:
            mlir_tensor_types: a list of MLIR tensor types which match the expected return type
        Returns:
            a pointer to a CompiledFunctionReturnValue, which corresponds to a structure in which
            fields match the expected return types
        """
        return self.getCompiledReturnValueType(mlir_tensor_types)

    def args_to_memref_descs(self, restype, args, **kwargs):
        """Convert ``args`` to memref descriptors.

        Besides converting the arguments to memrefs, it also prepares the return value. To respect
        the ABI, the return value is changed to a pointer and passed as the first parameter.

        Args:
            restype: the type of restype is a ``CompiledFunctionReturnValue``
            args: the JAX arrays to be used as arguments to the function

        Returns:
            List: a list of memref descriptor pointers to return values and parameters
            List: A list to the return values. It must be kept around until the function
                finishes execution as the memref descriptors will point to memory locations inside
                numpy arrays.

        """
        numpy_arg_buffer = []
        return_value_pointer = ctypes.POINTER(ctypes.c_int)()  # This is the null pointer

        if restype:
            return_value_pointer = self.restype_to_memref_descs(restype)

        c_abi_args = []

        args_data, args_shape = tree_flatten((args, kwargs))

        for arg in args_data:
            numpy_arg = np.asarray(arg)
            numpy_arg_buffer.append(numpy_arg)
            c_abi_ptr = ctypes.pointer(get_ranked_memref_descriptor(numpy_arg))
            c_abi_args.append(c_abi_ptr)

        args = tree_unflatten(args_shape, c_abi_args)

        class CompiledFunctionArgValue(ctypes.Structure):
            """Programmatically create a structure which holds tensors of varying base types."""

            _fields_ = [("f" + str(i), type(t)) for i, t in enumerate(c_abi_args)]

            def __init__(self, c_abi_args):
                for ft_tuple, c_abi_arg in zip(CompiledFunctionArgValue._fields_, c_abi_args):
                    f = ft_tuple[0]
                    setattr(self, f, c_abi_arg)

        arg_value_pointer = ctypes.POINTER(ctypes.c_int)()

        if len(args) > 0:
            arg_value = CompiledFunctionArgValue(c_abi_args)
            arg_value_pointer = ctypes.pointer(arg_value)

        c_abi_args = [return_value_pointer] + [arg_value_pointer]
        return c_abi_args, numpy_arg_buffer

    def get_cmain(self, *args):
        """Get a string representing a C program that can be linked against the shared object."""
        _, buffer = self.args_to_memref_descs(self.restype, args)

        return get_template(self.func_name, self.restype, *buffer)

    def __call__(self, *args, **kwargs):
        static_argnums = self.compile_options.static_argnums
        dynamic_args = filter_static_args(args, static_argnums)

        if self.compile_options.abstracted_axes is not None:
            abstracted_axes = self.compile_options.abstracted_axes
            dynamic_args = get_implicit_and_explicit_flat_args(
                abstracted_axes, *dynamic_args, **kwargs
            )

        abi_args, _buffer = self.args_to_memref_descs(self.restype, dynamic_args, **kwargs)

        numpy_dict = {nparr.ctypes.data: nparr for nparr in _buffer}

        result = CompiledFunction._exec(
            self.shared_object,
            self.restype,
            self.out_type,
            numpy_dict,
            *abi_args,
        )

        return result


@dataclass
class CacheKey:
    """A key by which to identify entries in the compiled function cache.

    The cache only rembers one compiled function for each possible combination of:
     - dynamic argument PyTree metadata
     - static argument values
    """

    treedef: PyTreeDef
    static_args: Tuple

    def __hash__(self) -> int:
        if not hasattr(self, "_hash"):
            # pylint: disable=attribute-defined-outside-init
            self._hash = hash((self.treedef, self.static_args))
        return self._hash


@dataclass
class CacheEntry:
    """An entry in the compiled function cache.

    For each compiled function, the cache stores the dynamic argument signature, the output PyTree
    definition, as well as the workspace in which compilation takes place.
    """

    compiled_fn: CompiledFunction
    signature: Tuple
    out_treedef: PyTreeDef
    workspace: Directory


class CompilationCache:
    """Class to manage CompiledFunction instances and retrieving previously compiled versions of
    a function.

    A full match requires the following properties to match:
     - dynamic argument signature (the shape and dtype of flattened arrays)
     - dynamic argument PyTree definitions
     - static argument values

    In order to allow some flexibility in the type of arguments provided by the user, a match is
    also produced if the dtype of dynamic arguments can be promoted to the dtype of an existing
    signature via JAX type promotion rules. Additional leniency is provided in the shape of
    dynamic arguments in accordance with the abstracted axis specification.

    To avoid excess promotion checks, only one function version is stored at a time for a given
    combination of PyTreeDefs and static arguments.
    """

    @debug_logger_init
    def __init__(self, static_argnums, abstracted_axes):
        self.static_argnums = static_argnums
        self.abstracted_axes = abstracted_axes
        self.cache = {}

    def get_function_status_and_key(self, args):
        """Check if the provided arguments match an existing function in the cache. The cache
        status of the function is returned as a compilation action:
         - no match: requires compilation
         - partial match: requires argument promotion
         - full match: skip promotion

        Args:
            args: arguments to match to existing functions

        Returns:
            TypeCompatibility
            CacheKey | None
        """
        if not self.cache:
            return TypeCompatibility.NEEDS_COMPILATION, None

        flat_runtime_sig, treedef, static_args = get_decomposed_signature(args, self.static_argnums)
        key = CacheKey(treedef, static_args)
        if key not in self.cache:
            return TypeCompatibility.NEEDS_COMPILATION, None

        entry = self.cache[key]
        runtime_signature = tree_unflatten(treedef, flat_runtime_sig)
        action = typecheck_signatures(entry.signature, runtime_signature, self.abstracted_axes)
        return action, key

    def lookup(self, args):
        """Get a function (if present) that matches the provided argument signature. Also computes
        whether promotion is necessary.

        Args:
            args (Iterable): the arguments to match to existing functions

        Returns:
            CacheEntry | None: the matched cache entry
            bool: whether the matched entry requires argument promotion
        """
        action, key = self.get_function_status_and_key(args)

        if action == TypeCompatibility.NEEDS_COMPILATION:
            return None, None
        elif action == TypeCompatibility.NEEDS_PROMOTION:
            return self.cache[key], True
        else:
            assert action == TypeCompatibility.CAN_SKIP_PROMOTION
            return self.cache[key], False

    def insert(self, fn, args, out_treedef, workspace):
        """Inserts the provided function into the cache.

        Args:
            fn (CompiledFunction): compilation result
            args (Iterable): arguments to determine cache key and additional metadata from
            out_treedef (PyTreeDef): the output shape of the function
            workspace (Directory): directory where compilation artifacts are stored
        """
        assert isinstance(fn, CompiledFunction)

        flat_signature, treedef, static_args = get_decomposed_signature(args, self.static_argnums)
        signature = tree_unflatten(treedef, flat_signature)

        key = CacheKey(treedef, static_args)
        entry = CacheEntry(fn, signature, out_treedef, workspace)
        self.cache[key] = entry
