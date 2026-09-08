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

#include <cstdio>
#include <exception>

#include "GpuLaunchers.hpp" // CoprocLaunchDesc (via Transport.hpp)
#include "GpuRuntime.hpp"   // HandoffSlot
#include "HipCheck.hpp"     // HIP_CHECK
#include "WireProtocol.hpp"
#include "echo_decoder.hpp"
#include "steane_decoder.hpp"

#include <hip/hip_runtime.h>

namespace catalyst::transport::coproc {
using namespace catalyst::transport::common;

// How often the spin loop samples the teardown flag. Must be a power of two: the
// loop tests `(spins & (STOP_POLL_INTERVAL - 1)) == 0`, i.e. spins that are a
// multiple of the interval.
constexpr std::uint32_t STOP_POLL_INTERVAL = 1024;

/**
 * @brief Persistent kernel templated on a device decoder: polls the HBM ring
 * buffer, executes the inline decoder, and writes corrections to the host
 * handoff buffer.
 *
 * Launched once with a single thread and runs until `total` shots
 * are done or the host raises `*stop`.
 *
 * @tparam DECODE Per-shot __device__ decode: syndrome word in, error qubit index out.
 * @param ring Request ring in local HBM, K_RING_SLOTS slots, written by the
 *             peer's RDMA_WRITEs. Slot `cursor % K_RING_SLOTS` carries shot
 *             `cursor`, published via its `seq_num` trailer.
 * @param handoff Host-mapped reply ring, same slot count; this kernel writes
 *                {correction, seq} and the CPU reads it.
 * @param stop Host-mapped teardown flag; nonzero means exit. Sampled only every
 *             STOP_POLL_INTERVAL spins (host-coherent read over PCIe).
 * @param total Number of shots to process, or 0 to run until stopped.
 */
template <std::int64_t (*DECODE)(std::uint64_t)>
__global__ void fused(volatile PayloadSlot *ring, volatile HandoffSlot *handoff,
                      volatile std::uint32_t *stop, std::uint64_t total) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    for (std::uint64_t cursor = 0; total == 0 || cursor < total; ++cursor) {
        const std::uint32_t expect = static_cast<std::uint32_t>(cursor + 1);
        const std::size_t index = cursor & (K_RING_SLOTS - 1);
        volatile std::uint32_t *sptr = &ring[index].p.seq_num;
        // Poll the request ring on local HBM. The stop flag is a host-coherent
        // read over PCIe, so sample it only every STOP_POLL_INTERVAL spins to
        // keep it off the request-detection hot path. Stop is a teardown flag.
        for (std::uint32_t spins = 0; *sptr != expect; ++spins) {
            if ((spins & (STOP_POLL_INTERVAL - 1)) == 0u && stop && *stop != 0) {
                return;
            }
        }
        __threadfence_system(); // seq observed -> ensure value read isn't hoisted/stale
        const std::uint64_t syndrome = ring[index].p.value;
        volatile HandoffSlot *slot = &handoff[index];
        HandoffSlot reply{
            .correction = DECODE(syndrome),
            .seq = expect,
            .pad = 0,
        };
        if constexpr (sizeof(HandoffSlot) <= sizeof(int4)) {
            // Reply fits one 16 B aligned store: publish correction+seq together
            // so the CPU cannot observe the seq trailer without its matching
            // correction. A single store has no intra-slot order to fence; one
            // trailing system fence pushes it into host-visible coherent memory.
            // NOTE: only lowers to a single global_store_dwordx4 at -O2+ (the build
            // defaults to RelWithDebInfo); at -O0 it splits into multiple stores.
            *reinterpret_cast<int4 *>(const_cast<HandoffSlot *>(slot)) =
                *reinterpret_cast<const int4 *>(&reply);
            __threadfence_system();
        } else {
            // Larger reply cannot be published atomically: fall back to the
            // seq-as-trailer protocol (write payload, fence, release seq, fence).
            slot->correction = reply.correction;
            __threadfence_system();
            slot->seq = reply.seq;
            __threadfence_system();
        }
    }
}

// Launch fused<DECODE> on the session-owned stream, wired to the datapath in
// `desc`.
template <std::int64_t (*DECODE)(std::uint64_t)> static void launch(const CoprocLaunchDesc *desc) {
    hipLaunchKernelGGL((fused<DECODE>), dim3(1), dim3(1), 0, static_cast<hipStream_t>(desc->stream),
                       reinterpret_cast<PayloadSlot *>(desc->ring),
                       reinterpret_cast<HandoffSlot *>(desc->handoff),
                       reinterpret_cast<std::uint32_t *>(desc->stop), desc->total);
    HIP_CHECK(hipGetLastError(), "kernel launch");
}

// Make sure we don't throw exception across C ABI.
template <std::int64_t (*DECODE)(std::uint64_t)>
static int launch_noexcept(const CoprocLaunchDesc *desc) noexcept {
    try {
        launch<DECODE>(desc);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "[coproc] %s\n", e.what());
        return 1;
    } catch (...) {
        std::fprintf(stderr, "[coproc] kernel launch failed: unknown error\n");
        return 1;
    }
}

extern "C" int gpu_echo_launcher(const CoprocLaunchDesc *desc, void *) {
    return launch_noexcept<echo_decode>(desc);
}

extern "C" int gpu_steane_launcher(const CoprocLaunchDesc *desc, void *) {
    return launch_noexcept<steane_decode>(desc);
}

int default_echo_launcher(const CoprocLaunchDesc *desc, void *ctx) {
    return gpu_echo_launcher(desc, ctx);
}

} // namespace catalyst::transport::coproc
