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

#include "GpuLaunchers.hpp" // CoprocLaunchDesc (via Transport.hpp)
#include "GpuRuntime.hpp"   // HandoffSlot
#include "WireProtocol.hpp"

#include "coprocessor_functions/echo_decoder.hpp"
#include "coprocessor_functions/steane_decoder.hpp"
#include <hip/hip_runtime.h>

namespace catalyst::transport::gpu_verbs {
using namespace catalyst::transport::common;

// Persistent detect -> decode -> write-handoff kernel, parameterized on the
// per-shot decode (a __device__ function). Body transcribed verbatim from the
// former GpuRuntime.hip fused_decode; the only change is gpu_decode(syndrome)
// -> DECODE(syndrome).
template <std::uint64_t (*DECODE)(std::uint64_t)>
__global__ void fused(volatile PayloadSlot *ring, volatile HandoffSlot *handoff,
                      volatile std::uint32_t *stop, std::uint64_t total)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    for (std::uint64_t cursor = 0; total == 0 || cursor < total; ++cursor) {
        const std::uint32_t expect = static_cast<std::uint32_t>(cursor + 1);
        const std::size_t index = cursor & (K_RING_SLOTS - 1);
        volatile std::uint32_t *sptr = &ring[index].p.seq_num;
        // Poll the request ring on local HBM; only check the stop
        // flag (host-coherent read over PCIe) ~every 1024 spins so it stays
        // off the request-detection hot path. Stop is a teardown flag.
        for (std::uint32_t spins = 0; *sptr != expect; ++spins) {
            if ((spins & 0x3ffu) == 0u && stop && *stop != 0)
                return;
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
        }
        else {
            // Larger reply cannot be published atomically: fall back to the
            // seq-as-trailer protocol (write payload, fence, release seq, fence).
            slot->correction = reply.correction;
            __threadfence_system();
            slot->seq = reply.seq;
            __threadfence_system();
        }
    }
}

// Enqueue fused<DECODE> on the session-owned stream, wired to the datapath in
// `desc`. The session keeps ownership of the stream, stop flag, sync() and the
// host handoff->reply thread; this only launches.
template <std::uint64_t (*DECODE)(std::uint64_t)> static int launch(const CoprocLaunchDesc *desc)
{
    hipLaunchKernelGGL((fused<DECODE>), dim3(1), dim3(1), 0, static_cast<hipStream_t>(desc->stream),
                       reinterpret_cast<PayloadSlot *>(desc->ring),
                       reinterpret_cast<HandoffSlot *>(desc->handoff),
                       reinterpret_cast<std::uint32_t *>(desc->stop), desc->total);
    // Launch is fire-and-forget (kernel runs async on the stream), but a bad
    // launch (e.g. invalid stream/config) surfaces synchronously here via
    // hipGetLastError(). Report it instead of returning "success" and leaving
    // the handoff ring unwritten (which would spin run_coprocessor forever).
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[gpu_verbs] kernel launch failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    return 0;
}

extern "C" int catalyst_gpu_echo_launcher(const CoprocLaunchDesc *desc, void *)
{
    return launch<echo_decode>(desc);
}

extern "C" int catalyst_gpu_steane_launcher(const CoprocLaunchDesc *desc, void *)
{
    return launch<steane_decode>(desc);
}

int default_echo_launcher(const CoprocLaunchDesc *desc, void *ctx)
{
    return catalyst_gpu_echo_launcher(desc, ctx);
}

} // namespace catalyst::transport::gpu_verbs
