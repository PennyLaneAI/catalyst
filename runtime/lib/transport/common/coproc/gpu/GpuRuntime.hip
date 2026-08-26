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

#include <algorithm>
#include <string>

#include "Error.hpp"
#include "GpuRuntime.hpp"
#include "HipCheck.hpp"

#include <hip/hip_runtime.h>

namespace catalyst::transport::coproc {

GpuRuntime::GpuRuntime(int device) {
    int dc = 0;
    HIP_CHECK(hipGetDeviceCount(&dc), "hipGetDeviceCount");
    TP_CHECK(dc > 0, "GpuRuntime: no HIP device");
    TP_CHECK(device >= 0 && device < dc, "GpuRuntime: HIP device %d out of range (%d present)",
             device, dc);
    HIP_CHECK(hipSetDevice(device), "hipSetDevice");
    hipStream_t s = nullptr;
    HIP_CHECK(hipStreamCreate(&s), "hipStreamCreate");
    stream_ = s;
}

GpuRuntime::~GpuRuntime() {
    if (stop_host_) {
        (void)hipHostFree(stop_host_);
    }
    if (handoff_host_) {
        (void)hipHostFree(handoff_host_);
    }
    if (hbm_ptr_) {
        (void)hipFree(hbm_ptr_);
    }
    if (stream_) {
        (void)hipStreamDestroy(static_cast<hipStream_t>(stream_));
    }
}

GpuRuntime::HbmRing GpuRuntime::alloc_hbm_ring(std::size_t bytes) {
    // Allocate exactly `bytes` (caller passes a page-aligned REGION_BYTES).
    HbmRing r{.size = bytes};
#if defined(__HIP_PLATFORM_AMD__)
    // NOTE: fine grained coherence is AMD specific and does not have
    // a CUDA equivalent, therefore, use standard `hipMalloc` in that case
    HIP_CHECK(hipExtMallocWithFlags(&r.ptr, r.size, hipDeviceMallocFinegrained),
              "hipExtMallocWithFlags(finegrained)");
#elif defined(__HIP_PLATFORM_NVIDIA__)
    HIP_CHECK(hipMalloc(&r.ptr, r.size), "hipMalloc");
#else
#error "Unsupported HIP platform"
#endif
    HIP_CHECK(hipMemset(r.ptr, 0, r.size), "hipMemset(ring)");
    HIP_CHECK(hipMemGetHandleForAddressRange(&r.dmabuf_fd, reinterpret_cast<hipDeviceptr_t>(r.ptr),
                                             r.size, hipMemRangeHandleTypeDmaBufFd, 0),
              "hipMemGetHandleForAddressRange(DmaBufFd)");
    hbm_ptr_ = r.ptr;
    return r;
}

GpuRuntime::Handoff GpuRuntime::alloc_handoff(std::size_t num_slots) {
    Handoff h;
    h.num_slots = num_slots;
    HIP_CHECK(hipHostMalloc((void **)&h.host, num_slots * sizeof(HandoffSlot),
                            hipHostMallocMapped | hipHostMallocCoherent),
              "hipHostMalloc(handoff)");
    // Zero every slot so a stale seq can never look like a published reply.
    std::fill_n(h.host, num_slots, HandoffSlot{});
    HIP_CHECK(hipHostGetDevicePointer((void **)&h.dev, h.host, 0),
              "hipHostGetDevicePointer(handoff)");
    HIP_CHECK(hipHostMalloc((void **)&h.stop_host, sizeof(std::uint32_t),
                            hipHostMallocMapped | hipHostMallocCoherent),
              "hipHostMalloc(stop)");
    *h.stop_host = 0;
    HIP_CHECK(hipHostGetDevicePointer((void **)&h.stop_dev, h.stop_host, 0),
              "hipHostGetDevicePointer(stop)");
    handoff_host_ = h.host;
    stop_host_ = h.stop_host;
    return h;
}

void GpuRuntime::sync() {
    if (stream_) {
        HIP_CHECK(hipStreamSynchronize(static_cast<hipStream_t>(stream_)), "hipStreamSynchronize");
    }
}

} // namespace catalyst::transport::coproc
