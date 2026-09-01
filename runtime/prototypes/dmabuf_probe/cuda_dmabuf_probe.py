#!/usr/bin/env python3
"""Does this NVIDIA GPU export device memory as a dma-buf fd?

The GPU coprocessor registers device memory with the NIC as a dma-buf: hipMalloc, then
hipMemGetHandleForAddressRange(hipMemRangeHandleTypeDmaBufFd) in GpuRuntime.hip, which hipcc 7.1
maps onto cuMemGetHandleForAddressRange. That export is the GPU half of gpu_verbs and is
independent of the NIC, so it can be settled on any box with a GPU -- no RDMA device, no guest VM,
no HIP toolchain.

Attempts the export directly rather than reading CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED: the
attribute is new in CUDA 13.0, so on an older toolkit the call itself is the more reliable answer.
Both allocators the transport uses are covered -- device memory (hipMalloc) and page-locked host
memory (hipHostMalloc), which the docs gate behind separate attributes.

Run on the GPU host: python3 cuda_dmabuf_probe.py
Exit 0 = at least one export worked, 1 = none did, 2 = no usable CUDA driver.
"""

import ctypes
import os
import sys

CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD = 1
CU_MEMHOSTALLOC_DEVICEMAP = 0x02
SIZE = 1 << 20


def main() -> int:
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        print(f"no CUDA driver library: {exc}")
        return 2

    def check(name, code):
        if code == 0:
            return True
        s = ctypes.c_char_p()
        cuda.cuGetErrorName(code, ctypes.byref(s))
        text = s.value.decode() if s.value else "?"
        print(f"  {name} -> {text} ({code})")
        return False

    if not check("cuInit", cuda.cuInit(0)):
        return 2
    version = ctypes.c_int()
    cuda.cuDriverGetVersion(ctypes.byref(version))
    print(f"driver API version: {version.value // 1000}.{version.value % 1000 // 10}")

    dev = ctypes.c_int()
    if not check("cuDeviceGet", cuda.cuDeviceGet(ctypes.byref(dev), 0)):
        return 2
    name = ctypes.create_string_buffer(256)
    cuda.cuDeviceGetName(name, len(name), dev)
    print(f"device 0: {name.value.decode()}")

    ctx = ctypes.c_void_p()
    if not check("cuCtxCreate", cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)):
        return 2

    worked = []

    # Device memory, as GpuRuntime::alloc_hbm_ring does.
    dptr = ctypes.c_void_p()
    if check("cuMemAlloc", cuda.cuMemAlloc_v2(ctypes.byref(dptr), ctypes.c_size_t(SIZE))):
        fd = ctypes.c_int(-1)
        rc = cuda.cuMemGetHandleForAddressRange(
            ctypes.byref(fd),
            dptr,
            ctypes.c_size_t(SIZE),
            CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
            ctypes.c_ulonglong(0),
        )
        if check("export(device memory)", rc):
            print(f"  device memory exported as dma-buf fd={fd.value}")
            worked.append("device")
            os.close(fd.value)
        cuda.cuMemFree_v2(dptr)

    # Page-locked host memory, mapped -- the allocator the handoff path leans on.
    hptr = ctypes.c_void_p()
    if check(
        "cuMemHostAlloc",
        cuda.cuMemHostAlloc(ctypes.byref(hptr), ctypes.c_size_t(SIZE), CU_MEMHOSTALLOC_DEVICEMAP),
    ):
        # The exportable range for cuMemHostAlloc is the host pointer itself, not the mapped
        # device pointer from cuMemHostGetDevicePointer -- passing the latter earns
        # CUDA_ERROR_INVALID_VALUE regardless of whether the device supports the export.
        fd = ctypes.c_int(-1)
        rc = cuda.cuMemGetHandleForAddressRange(
            ctypes.byref(fd),
            hptr,
            ctypes.c_size_t(SIZE),
            CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
            ctypes.c_ulonglong(0),
        )
        if check("export(host alloc)", rc):
            print(f"  page-locked host memory exported as dma-buf fd={fd.value}")
            worked.append("host")
            os.close(fd.value)
        cuda.cuMemFreeHost(hptr)

    print(f"\nexportable: {', '.join(worked) if worked else 'nothing'}")
    return 0 if worked else 1


if __name__ == "__main__":
    sys.exit(main())
