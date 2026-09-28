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

namespace catalyst::transport::coproc {

// Per-slot GPU->CPU handoff, written by the fused kernel and read by the CPU
// engine. `seq` is the trailer (== cursor + 1). When the slot is <= 16 B the
// kernel publishes it in one aligned vector store, so the CPU can never observe
// seq updated before correction (no intra-slot ordering fence needed);
// alignas(16) makes that single 16 B store well-defined. If the slot ever grows
// past 16 B the kernel falls back to the write-correction / fence / release-seq
// / fence trailer protocol instead.
struct alignas(16) HandoffSlot {
    std::int64_t correction; // error qubit index the decoder produced; -1 for no error
    std::uint32_t seq;
    std::uint32_t pad;
};

class GpuRuntime {
  public:
    explicit GpuRuntime(int device);
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
        HandoffSlot *host = nullptr;        // CPU-side view of the ring
        HandoffSlot *dev = nullptr;         // device-side view of the same pages
        std::uint32_t *stop_host = nullptr; // CPU-side teardown flag
        std::uint32_t *stop_dev = nullptr;  // device-side view of the same flag
        std::size_t num_slots = 0;          // number of slots in the ring
    };
    Handoff alloc_handoff(std::size_t num_slots);

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

} // namespace catalyst::transport::coproc
