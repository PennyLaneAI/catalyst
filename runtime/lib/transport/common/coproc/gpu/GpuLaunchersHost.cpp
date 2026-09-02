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
 * Host-thread stand-ins for the persistent GPU kernels, built in place of GpuLaunchers.hip when
 * TRANSPORT_GPU_STUB is on. See GpuRuntimeHost.cpp for why.
 *
 * The worker mirrors `fused<DECODE>` one step at a time: spin on the request slot's seq_num until
 * it reaches the expected value, read the syndrome, decode, then publish correction and seq
 * together. Where the kernel uses __threadfence_system to reach host-coherent memory, this uses
 * acquire/release fences -- both sides are ordinary RAM here, so ordering is all that is needed.
 * The stop flag is sampled on the same interval as the kernel, keeping the polling shape honest.
 */
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <thread>

#include "GpuRuntime.hpp" // HandoffSlot
#include "SteaneLut.hpp"
#include "Transport.hpp"
#include "WireProtocol.hpp"

namespace catalyst::transport::coproc {

// HandoffSlot is this namespace's own (GpuRuntime.hpp); the ring types come from the wire protocol.
using common::K_RING_SLOTS;
using common::PayloadSlot;

namespace {

// Host mirrors of the __device__ decoders; the tables in SteaneLut.hpp are plain constexpr, so the
// arithmetic is shared with the device path rather than duplicated.
std::int64_t echo_decode_host(std::uint64_t syndrome) {
    return static_cast<std::int64_t>(syndrome);
}

std::int64_t steane_decode_host(std::uint64_t syndrome) {
    std::uint32_t packed = 0;
    for (int i = 0; i < STEANE_CHECKS; ++i) {
        packed = (packed << 1U) | static_cast<std::uint32_t>((syndrome >> (8 * i)) & 1U);
    }
    const std::uint32_t qubit = (STEANE_TABLE_PACKED >> (packed * 4U)) & 0xFU;
    return (qubit == 0xFU) ? -1 : static_cast<std::int64_t>(qubit);
}

// Matches the kernel's STOP_POLL_INTERVAL: the flag is checked once every N spins rather than on
// the request-detection hot path.
constexpr std::uint32_t K_STOP_POLL_INTERVAL = 1024;

void worker(const CoprocLaunchDesc desc, std::int64_t (*decode)(std::uint64_t)) {
    auto *ring = static_cast<volatile PayloadSlot *>(desc.ring);
    auto *handoff = static_cast<volatile HandoffSlot *>(desc.handoff);
    auto *stop = static_cast<volatile std::uint32_t *>(desc.stop);
    const std::uint64_t total = desc.total;

    for (std::uint64_t cursor = 0; total == 0 || cursor < total; ++cursor) {
        const std::uint32_t expect = static_cast<std::uint32_t>(cursor + 1);
        const std::size_t index = cursor & (K_RING_SLOTS - 1);
        volatile std::uint32_t *sptr = &ring[index].p.seq_num;
        for (std::uint32_t spins = 0; *sptr != expect; ++spins) {
            if ((spins & (K_STOP_POLL_INTERVAL - 1)) == 0u && stop != nullptr && *stop != 0) {
                return;
            }
            std::this_thread::yield();
        }
        // seq observed: order the value read after it, as __threadfence_system does on device.
        std::atomic_thread_fence(std::memory_order_acquire);
        const std::uint64_t syndrome = ring[index].p.value;

        volatile HandoffSlot *slot = &handoff[index];
        slot->correction = decode(syndrome);
        slot->pad = 0;
        // seq is the trailer: release it only once the correction is visible, so the session never
        // reads a seq without its matching correction.
        std::atomic_thread_fence(std::memory_order_release);
        slot->seq = expect;
        std::atomic_thread_fence(std::memory_order_release);
    }
}

int launch_noexcept(const CoprocLaunchDesc *desc, std::int64_t (*decode)(std::uint64_t)) noexcept {
    try {
        // Detached, like a kernel left running on its stream: the session tears it down through the
        // stop flag, which the worker samples while spinning.
        std::thread(worker, *desc, decode).detach();
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "[coproc] %s\n", e.what());
        return 1;
    } catch (...) {
        std::fprintf(stderr, "[coproc] host worker launch failed: unknown error\n");
        return 1;
    }
}

} // namespace

extern "C" int gpu_echo_launcher(const CoprocLaunchDesc *desc, void *) {
    return launch_noexcept(desc, echo_decode_host);
}

extern "C" int gpu_steane_launcher(const CoprocLaunchDesc *desc, void *) {
    return launch_noexcept(desc, steane_decode_host);
}

int default_echo_launcher(const CoprocLaunchDesc *desc, void *ctx) {
    return gpu_echo_launcher(desc, ctx);
}

} // namespace catalyst::transport::coproc
