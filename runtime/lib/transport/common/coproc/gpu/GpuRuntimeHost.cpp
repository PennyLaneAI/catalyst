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

/**
 * Host-memory GpuRuntime, built in place of GpuRuntime.hip when TRANSPORT_GPU_STUB is on.
 *
 * The GPU-verbs session, its ring and handoff protocol, the RC bring-up and gpu_verbs_selftest are
 * all reachable without a GPU -- only the allocations and the persistent kernel are not. This
 * supplies the allocations from host RAM so the rest can be exercised on any machine; the kernel
 * side is in GpuLaunchersHost.cpp.
 *
 * Implements the same GpuRuntime.hpp declarations as the HIP version, so nothing above it changes.
 * alloc_hbm_ring reports dmabuf_fd = -1: host pages have no dma-buf export, and the session reads
 * that as "register this the ordinary way" (see GpuCoprocessorSession::alloc_memory).
 */
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <new>

#include "Error.hpp"
#include "GpuRuntime.hpp"

namespace catalyst::transport::coproc {

namespace {
// The HIP path hands the NIC device memory whose alignment it controls; match the 64 B slot
// alignment PayloadSlot/HandoffSlot assume so the layout assertions keep holding on host pages.
constexpr std::size_t K_ALIGN = 64;

void *aligned_zeroed(std::size_t bytes) {
    void *p = nullptr;
    const std::size_t rounded = ((bytes + K_ALIGN - 1) / K_ALIGN) * K_ALIGN;
    if (posix_memalign(&p, K_ALIGN, rounded) != 0 || p == nullptr) {
        throw std::bad_alloc();
    }
    std::memset(p, 0, rounded);
    return p;
}
} // namespace

GpuRuntime::GpuRuntime(int device) {
    // No device to select and no stream to create; the launcher runs a host thread instead.
    static_cast<void>(device);
    stream_ = nullptr;
}

GpuRuntime::~GpuRuntime() {
    std::free(hbm_ptr_);
    std::free(handoff_host_);
    std::free(stop_host_);
}

GpuRuntime::HbmRing GpuRuntime::alloc_hbm_ring(std::size_t bytes) {
    TP_CHECK(hbm_ptr_ == nullptr, "gpu stub: alloc_hbm_ring called twice");
    hbm_ptr_ = aligned_zeroed(bytes);
    return HbmRing{
        .ptr = hbm_ptr_,
        .size = bytes,
        .dmabuf_fd = -1, // host pages: no dma-buf export
    };
}

GpuRuntime::Handoff GpuRuntime::alloc_handoff(std::size_t num_slots) {
    TP_CHECK(handoff_host_ == nullptr, "gpu stub: alloc_handoff called twice");
    handoff_host_ = static_cast<HandoffSlot *>(aligned_zeroed(num_slots * sizeof(HandoffSlot)));
    stop_host_ = static_cast<std::uint32_t *>(aligned_zeroed(sizeof(std::uint32_t)));
    // One address space, so the "device view" of each mapping is the host pointer itself.
    return Handoff{
        .host = handoff_host_,
        .dev = handoff_host_,
        .stop_host = stop_host_,
        .stop_dev = stop_host_,
        .num_slots = num_slots,
    };
}

void GpuRuntime::sync() {
    // Stands in for hipStreamSynchronize: the host worker publishes through release stores, so a
    // full fence is all the caller needs to observe them.
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

} // namespace catalyst::transport::coproc
