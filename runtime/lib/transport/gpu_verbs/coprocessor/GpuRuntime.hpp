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

#pragma once
#include <cstddef>
#include <cstdint>

namespace catalyst::transport::gpu_verbs {

// Per-slot GPU->CPU handoff, written by the fused kernel and read by the CPU
// engine. `seq` is the trailer (== cursor + 1). When the slot is <= 16 B the
// kernel publishes it in one aligned vector store, so the CPU can never observe
// seq updated before correction (no intra-slot ordering fence needed);
// alignas(16) makes that single 16 B store well-defined. If the slot ever grows
// past 16 B the kernel falls back to the write-correction / fence / release-seq
// / fence trailer protocol instead.
struct alignas(16) HandoffSlot {
    std::uint64_t correction;
    std::uint32_t seq;
    std::uint32_t pad;
};

// PRECONDITION: if the persistent kernel was launched with total == 0, the
// caller must set *stop_dev to non-zero and call sync() before destroying this
// object. The
// destructor frees memory unconditionally without stopping the kernel;
// destroying while the fused kernel is still running is undefined behavior.
class GpuRuntime {
  public:
    GpuRuntime(); // hipSetDevice(0) + stream; throws std::runtime_error on
                  // failure
    ~GpuRuntime();
    GpuRuntime(const GpuRuntime &) = delete;
    GpuRuntime &operator=(const GpuRuntime &) = delete;

    struct HbmRing {
        void *ptr = nullptr;  // device pointer; also acts as MR addr and I/O
                              // virtual address for NIC
        std::size_t size = 0; // allocation size
        int dmabuf_fd = -1;   // export fd; caller closes after ibv_reg_dmabuf_mr
    };
    HbmRing alloc_hbm_ring(std::size_t bytes);

    struct Handoff {
        HandoffSlot *host = nullptr;
        HandoffSlot *dev = nullptr;
        std::uint32_t *stop_host = nullptr;
        std::uint32_t *stop_dev = nullptr;
        std::size_t slots = 0;
    };
    Handoff alloc_handoff(std::size_t slots);

    void sync(); // hipStreamSynchronize

    // Session owns the stream; expose it (as void*) so a launcher can enqueue the
    // persistent kernel on it via a CoprocLaunchDesc descriptor.
    void *stream() const { return stream_; }

  private:
    void *stream_ = nullptr;              // hipStream_t
    void *hbm_ptr_ = nullptr;             // owned; hipFree in dtor
    HandoffSlot *handoff_host_ = nullptr; // owned; hipHostFree in dtor
    std::uint32_t *stop_host_ = nullptr;  // owned; hipHostFree in dtor
};

} // namespace catalyst::transport::gpu_verbs
