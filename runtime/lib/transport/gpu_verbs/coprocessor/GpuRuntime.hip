// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdexcept>
#include <string>

#include "GpuRuntime.hpp"

#include <hip/hip_runtime.h>

namespace catalyst::transport::gpu_verbs {
namespace {
void hip_check(hipError_t e, const char *what)
{
    if (e != hipSuccess)
        throw std::runtime_error(std::string("GpuRuntime: ") + what + ": " + hipGetErrorString(e));
}
} // namespace

GpuRuntime::GpuRuntime()
{
    int dc = 0;
    hip_check(hipGetDeviceCount(&dc), "hipGetDeviceCount");
    if (dc <= 0)
        throw std::runtime_error("GpuRuntime: no HIP device");
    hip_check(hipSetDevice(0), "hipSetDevice(0)");
    hipStream_t s = nullptr;
    hip_check(hipStreamCreate(&s), "hipStreamCreate");
    stream_ = s;
}

GpuRuntime::~GpuRuntime()
{
    if (stop_host_)
        (void)hipHostFree(stop_host_);
    if (handoff_host_)
        (void)hipHostFree(handoff_host_);
    if (hbm_ptr_)
        (void)hipFree(hbm_ptr_);
    if (stream_)
        (void)hipStreamDestroy(static_cast<hipStream_t>(stream_));
}

GpuRuntime::HbmRing GpuRuntime::alloc_hbm_ring(std::size_t bytes)
{
    HbmRing r{.size =
                  bytes}; // Allocate exactly `bytes` (caller passes a page-aligned REGION_BYTES).
#if defined(__HIP_PLATFORM_AMD__)
    // NOTE: fine grained coherence is an AMD specific notion and does not have
    // a CUDA equivalent, therefore, use standard `hipMalloc` in that case
    hip_check(hipExtMallocWithFlags(&r.ptr, r.size, hipDeviceMallocFinegrained),
              "hipExtMallocWithFlags(finegrained)");
#elif defined(__HIP_PLATFORM_NVIDIA__)
    hip_check(hipMalloc(&r.ptr, r.size), "hipMalloc");
#else
#error "Unsupported HIP platform"
#endif
    hip_check(hipMemset(r.ptr, 0, r.size), "hipMemset(ring)");
    hip_check(hipMemGetHandleForAddressRange(&r.dmabuf_fd, reinterpret_cast<hipDeviceptr_t>(r.ptr),
                                             r.size, hipMemRangeHandleTypeDmaBufFd, 0),
              "hipMemGetHandleForAddressRange(DmaBufFd)");
    hbm_ptr_ = r.ptr;
    return r;
}

GpuRuntime::Handoff GpuRuntime::alloc_handoff(std::size_t slots)
{
    Handoff h;
    h.slots = slots;
    hip_check(hipHostMalloc((void **)&h.host, slots * sizeof(HandoffSlot),
                            hipHostMallocMapped | hipHostMallocCoherent),
              "hipHostMalloc(handoff)");
    for (std::size_t i = 0; i < slots; ++i) {
        h.host[i].correction = 0;
        h.host[i].seq = 0;
        h.host[i].pad = 0;
    }
    hip_check(hipHostGetDevicePointer((void **)&h.dev, h.host, 0),
              "hipHostGetDevicePointer(handoff)");
    hip_check(hipHostMalloc((void **)&h.stop_host, sizeof(std::uint32_t),
                            hipHostMallocMapped | hipHostMallocCoherent),
              "hipHostMalloc(stop)");
    *h.stop_host = 0;
    hip_check(hipHostGetDevicePointer((void **)&h.stop_dev, h.stop_host, 0),
              "hipHostGetDevicePointer(stop)");
    handoff_host_ = h.host;
    stop_host_ = h.stop_host;
    return h;
}

void GpuRuntime::sync()
{
    if (stream_)
        hip_check(hipStreamSynchronize(static_cast<hipStream_t>(stream_)), "hipStreamSynchronize");
}

} // namespace catalyst::transport::gpu_verbs
