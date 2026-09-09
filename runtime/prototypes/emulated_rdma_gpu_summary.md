# Emulated RDMA NIC + GPU: summary

Branch: `shuli/rdma-nic-amd-gpu-emulation`

## Goal

Check whether a one-controller-to-multiple-coprocessors topology can be tested via an emulated
GPU and NIC, to lift the real-hardware constraint entirely.

## Bottom line

Partially feasible, not fully.

- Multi-coprocessor over **`cpu_verbs`** (RDMA, CPU-only) — feasible today: soft-RoCE loopback
  already gives a real, working verbs device; scaling one controller to N coprocessors is a
  test-writing exercise, not a stack limitation.
- Multi-coprocessor over **`gpu_verbs`** (RDMA + GPU, the actual production topology) — **not
  feasible** with the current emulated stack: no emulated NIC ever reached a working verbs
  device, and the GPU stub can't emulate the dma-buf export a real coprocessor needs to
  register its memory with the NIC. This combination still requires real hardware.

## Actions taken

- Tried [`rocm-ernic`](https://github.com/ROCm/rocm-ernic) (built on
  [`libvfio-user`](https://github.com/nutanix/libvfio-user)) to emulate an RDMA NIC — builds and
  self-tests in a container, but stops at the server half; a real verbs device needs a guest VM
  (not built). See `runtime/prototypes/emulated_rnic/README.md` for the full spike writeup.
- Built a `TRANSPORT_GPU_STUB` (host memory + host threads, `GpuRuntimeHost.cpp` /
  `GpuLaunchersHost.cpp`) so `runner_tests_transport_gpu` builds and runs without a GPU, plus a
  `udmabuf`-backed MR registration test in `runtime/tests/Test_TransportCommon.cpp`.
- Added a compile-only `hip_amd_toolchain` CI job (ROCm `amdclang++` via Docker,
  `runtime/prototypes/amd_hip_compile/`) to catch AMD-only `.hip` compile errors.
- Added a diagnostic-only `gpu-dmabuf-probe` dispatch job
  (`runtime/prototypes/dmabuf_probe/cuda_dmabuf_probe.py`) to check dma-buf export on real GPU
  hardware.

## Packages / tools used

- `rocm-ernic` (libvfio-user NIC emulator)
- ROCm `amdclang++` / `hip-dev` / `rocm-device-libs` (installed in a Docker image)
- Linux `udmabuf` driver
- CUDA driver API (`libcuda.so.1`), for the dma-buf probe

## Tests added

- `Test_TransportCommon.cpp`: registers a `udmabuf`-backed host memory region as an MR.
- `runner_tests_transport_gpu`: now builds/runs via the host-memory/thread stub instead of
  needing HIP.
- `hip_amd_toolchain` CI job: compile-only check of the AMD `.hip` sources.
- `gpu-dmabuf-probe` CI job: diagnostic-only, `workflow_dispatch` gated.

## Limitations

- GPU stub only exercises the ABI/handoff protocol, not real GPU compute/decoder correctness.
- **The stub fundamentally can't emulate the GPU's dma-buf export**
  (`hipMemGetHandleForAddressRange(..., hipMemRangeHandleTypeDmaBufFd)`) — that call needs a
  real driver/PCIe device, and an emulation has no device memory to get a handle for, so it can
  only skip this path rather than test it.
- AMD check is compile-only, no execution on AMD hardware.
- dma-buf probe doesn't fail CI.
- Soft-RoCE (`rxe`) predates dma-buf MR support, so the udmabuf MR test itself can hit
  `EOPNOTSUPP` on current kernels.
- Emulated NIC never reached a real verbs device (needs an unbuilt guest VM).

## Verbs backend coverage

- `cpu_verbs`: fully tested end-to-end (real soft-RoCE device).
- `gpu_verbs`: only partially tested — ABI/memory-registration path via the stub, not real
  GPU-to-NIC dma-buf transfer or actual decode execution.
- `fpga_verbs` / `fpga_hwhs`: no coverage at all.
