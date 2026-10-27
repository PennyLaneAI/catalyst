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

#include "DeviceCopy.hpp"
#include "Frame.hpp"
#include "GpuLaunchers.hpp" // CoprocLaunchDesc (via Transport.hpp)
#include "HipCheck.hpp"     // HIP_CHECK
#include "TransportABI.h"   // CatalystCoprocessorFnInfo
#include "echo_decoder.hpp"
#include "steane_decoder.hpp"

#include <hip/hip_runtime.h>

namespace catalyst::transport::coproc {
using namespace catalyst::transport::common;

// Threads in the kernel's one block: a single warp on NVIDIA, half a wavefront on AMD, so the
// block runs in lockstep and its barrier is cheap. The lead polls; for a BLV frame all of them
// share the payload copy (see device_copy). A 16 B frame needs one load and one store, so the lead
// handles it alone: about 4.1 us round trip on the setup device_copy describes, the same as
// with all 32 threads running it.
constexpr unsigned KERNEL_THREADS = 32;

// How often the spin loop samples the teardown flag. Must be a power of two: the
// loop tests `(spins & (STOP_POLL_INTERVAL - 1)) == 0`, i.e. spins that are a
// multiple of the interval.
constexpr std::uint32_t STOP_POLL_INTERVAL = 1024;

/// A device decoder: reads `len` payload bytes from `in`, writes at most `cap` reply bytes to
/// `out`, and returns how many it wrote, which become the reply frame's length. Every thread of a
/// group of `lanes` makes the same call, this one being `lane`, and all must return the same
/// length; device_copy shares a write among them. It cannot fail a message. The host zero-fills
/// the reply past its length, which is cheaper there than over PCIe.
using DeviceDecodeFn = std::uint32_t (*)(const std::uint8_t *in, std::uint32_t len,
                                         std::uint32_t fn_id, std::uint8_t *out, std::uint32_t cap,
                                         std::uint32_t lane, std::uint32_t lanes);

/**
 * @brief Persistent kernel templated on a device decoder: polls the request ring, runs the
 * decoder on each message, and answers it with a reply frame in the handoff ring.
 *
 * Launched once as one block of KERNEL_THREADS threads, and runs until `total` messages are done
 * or the host raises `*stop`. Both rings hold frames in the layout `ring` describes (see
 * CoprocLaunchDesc).
 *
 * @tparam DECODE The device decoder.
 * @tparam MODE The frame layout of both rings. A compile-time mode keeps the other layouts'
 *              branches out of the 16 B hot loop, which costs about 0.1 us per message otherwise
 *              (MI210).
 * @param requests Request ring, written by the peer: RDMA writes into HBM, or the host.
 * @param replies Host-mapped reply ring the CPU reads, same layout.
 * @param stop Host-mapped teardown flag; nonzero means exit. Sampled only every
 *             STOP_POLL_INTERVAL spins (host-coherent read over PCIe).
 * @param total Number of messages to process, or 0 to run until stopped.
 * @param ring The frame layout and slot geometry of both rings.
 * @param rx_flags, tx_flags CUQR only: the per-slot arrival and reply flags (CoprocLaunchDesc).
 */
template <DeviceDecodeFn DECODE, frame::Mode MODE>
__global__ void fused(const std::uint8_t *requests, std::uint8_t *replies,
                      volatile std::uint32_t *stop, std::uint64_t total, frame::RingGeometry ring,
                      volatile std::uint32_t *rx_flags, volatile std::uint32_t *tx_flags) {
    if (blockIdx.x != 0) {
        return;
    }
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t lanes = blockDim.x;
    const bool lead = lane == 0;
    constexpr bool bl16 = MODE == frame::Mode::BL16;
    constexpr bool cuqr = MODE == frame::Mode::CUQR;
    // A 16 B frame is decoded and published by one thread, so the others leave, and the lead
    // runs with no barrier on its hot path.
    if (bl16 && !lead) {
        return;
    }
    // Where the payload starts, fixed by the layout.
    constexpr std::size_t header = cuqr ? sizeof(frame::CuqrHeader) : sizeof(frame::BlvHeader);
    const auto cap =
        static_cast<std::uint32_t>(ring.stride - header - (cuqr ? 0 : frame::BLV_COMPLETION_BYTES));
    __shared__ bool halted; // the lead's stop decision, one for the whole block
    for (std::uint64_t cursor = 0; total == 0 || cursor < total; ++cursor) {
        const std::uint32_t expect = static_cast<std::uint32_t>(cursor + 1);
        const std::uint8_t *req = requests + ring.slot_offset(cursor);
        std::uint8_t *rep = replies + ring.slot_offset(cursor);
        // The lead polls the request's completion word and, for BLV, the others wait at the
        // barrier, so the block makes one stop decision and never splits at a barrier. The stop
        // flag is a host-coherent read over PCIe, so it is sampled only every STOP_POLL_INTERVAL
        // spins to keep it off the request-detection hot path. Stop is a teardown flag.
        bool stopping = false;
        if (lead) {
            // CUQR polls rx_flags in host memory, across PCIe (see doc/dev/backline.rst).
            const volatile std::uint32_t *done =
                cuqr ? rx_flags + (cursor & (ring.slots - 1))
                     : reinterpret_cast<const volatile std::uint32_t *>(req + ring.poll_offset());
            for (std::uint32_t spins = 0; *done != expect; ++spins) {
                if ((spins & (STOP_POLL_INTERVAL - 1)) == 0u && stop && *stop != 0) {
                    stopping = true;
                    break;
                }
            }
        }
        if (!bl16) { // share the lead's decision; a 16 B frame's lead is alone
            if (lead) {
                halted = stopping;
            }
            __syncthreads();
            stopping = halted;
        }
        if (stopping) {
            return;
        }
        __threadfence_system(); // completion observed -> the payload read below is not stale
        if (bl16) {
            // One 16 B load brings the payload and fn_id together; the fence above keeps it after
            // the poll.
            const int4 raw = *reinterpret_cast<const int4 *>(req);
            Payload request;
            __builtin_memcpy(&request, &raw, sizeof(request));
            // The whole 16 B reply frame is built here and published in one aligned store, so
            // the CPU cannot see its seq without its payload and fn_id.
            // NOTE: only lowers to a single global_store_dwordx4 at -O2+, which the build sets.
            alignas(16)
                Payload reply{.value = 0, .decoder_id = request.decoder_id, .seq_num = expect};
            // The lead decodes alone (a group of one), so the 8 B payload and reply stay in its
            // registers.
            DECODE(reinterpret_cast<const std::uint8_t *>(&request.value), PAYLOAD_DATA_BYTES,
                   request.decoder_id, reinterpret_cast<std::uint8_t *>(&reply.value),
                   PAYLOAD_DATA_BYTES, 0, 1);
            *reinterpret_cast<int4 *>(rep) = *reinterpret_cast<const int4 *>(&reply);
            __threadfence_system();
            continue;
        }
        if (cuqr) {
            // CUQR: the payload, then cudaq-realtime's RPCResponse, then the reply flag last.
            const volatile frame::CuqrHeader *rh =
                reinterpret_cast<const volatile frame::CuqrHeader *>(req);
            // Not a frame (wrong magic, or longer than its slot): no reply, the host drops it.
            if (rh->magic != frame::CUQR_MAGIC || rh->len > cap) {
                if (lead) {
                    *reinterpret_cast<volatile std::uint32_t *>(rep) = 0;
                    __threadfence_system();
                    tx_flags[cursor & (ring.slots - 1)] = expect;
                    __threadfence_system();
                }
                continue;
            }
            const std::uint32_t fn_id = rh->word;
            const std::uint32_t len = rh->len;
            const std::uint32_t request_id = rh->request_id;
            const std::uint64_t t_ptp_ns = rh->t_ptp_ns;
            const std::uint32_t written =
                DECODE(req + header, len, fn_id, rep + header, cap, lane, lanes);
            if (lead) { // magic, status 0 (a device decoder cannot fail), len, the request's id
                *reinterpret_cast<int4 *>(rep) =
                    int4{static_cast<int>(frame::CUQS_MAGIC), 0, static_cast<int>(written),
                         static_cast<int>(request_id)};
                *reinterpret_cast<std::uint64_t *>(rep + 16) = t_ptp_ns;
            }
            __threadfence_system();
            __syncthreads();
            if (lead) {
                tx_flags[cursor & (ring.slots - 1)] = expect;
                __threadfence_system();
            }
            continue;
        }
        // BLV: the payload, then the header, then the completion word last.
        const volatile frame::BlvHeader *rh =
            reinterpret_cast<const volatile frame::BlvHeader *>(req);
        const std::uint32_t fn_id = rh->fn_id;
        const std::uint32_t len = rh->len < cap ? rh->len : cap;
        std::uint8_t *out = rep + header;
        const std::uint32_t written = DECODE(req + header, len, fn_id, out, cap, lane, lanes);
        // The header, as two 16 B stores built in registers: a local BlvHeader would live in
        // scratch memory, which slows every message, 16 B frames included.
        if (lead) {
            int4 *h = reinterpret_cast<int4 *>(rep);
            h[0] = int4{static_cast<int>(frame::BLV_MAGIC), static_cast<int>(frame::BLV_VERSION),
                        static_cast<int>(fn_id), static_cast<int>(expect)}; // flags 0
            h[1] = int4{0, static_cast<int>(written), 0, 0};                // round, len, t_ptp_ns
        }
        // Every thread's share of the frame is visible before the lead writes its completion.
        __threadfence_system();
        __syncthreads();
        if (lead) {
            *reinterpret_cast<volatile std::uint32_t *>(rep + ring.poll_offset()) = expect;
            __threadfence_system();
        }
    }
}

static_assert(offsetof(frame::BlvHeader, version) == 4 && offsetof(frame::BlvHeader, flags) == 6 &&
                  offsetof(frame::BlvHeader, fn_id) == 8 && offsetof(frame::BlvHeader, seq) == 12 &&
                  offsetof(frame::BlvHeader, round) == 16 &&
                  offsetof(frame::BlvHeader, len) == 20 &&
                  offsetof(frame::BlvHeader, t_ptp_ns) == 24,
              "fused writes the BLV header as two int4 in this layout");
static_assert(offsetof(frame::CuqrHeader, len) == 8 &&
                  offsetof(frame::CuqrHeader, request_id) == 12 &&
                  offsetof(frame::CuqrHeader, t_ptp_ns) == 16,
              "fused writes the CUQR header as an int4 and a uint64 in this layout");

// Launch fused<DECODE, mode> on the session-owned stream, wired to the datapath in `desc`.
template <DeviceDecodeFn DECODE> static void launch(const CoprocLaunchDesc *desc) {
    TP_CHECK(desc->frame_mode <= static_cast<std::uint32_t>(frame::Mode::CUQR),
             "GPU launcher: frame mode %u is not implemented", desc->frame_mode);
    const frame::RingGeometry ring{
        .mode = static_cast<frame::Mode>(desc->frame_mode),
        .slots = desc->ring_slots,
        .stride = desc->slot_stride,
    };
    TP_CHECK(ring.mode != frame::Mode::CUQR || (desc->rx_flags && desc->tx_flags),
             "GPU launcher: CUQR frames need rx_flags and tx_flags");
    auto *kernel = ring.mode == frame::Mode::BL16  ? fused<DECODE, frame::Mode::BL16>
                   : ring.mode == frame::Mode::BLV ? fused<DECODE, frame::Mode::BLV>
                                                   : fused<DECODE, frame::Mode::CUQR>;
    hipLaunchKernelGGL(
        kernel, dim3(1), dim3(KERNEL_THREADS), 0, static_cast<hipStream_t>(desc->stream),
        static_cast<const std::uint8_t *>(desc->ring), static_cast<std::uint8_t *>(desc->handoff),
        static_cast<volatile std::uint32_t *>(desc->stop), desc->total, ring,
        static_cast<volatile std::uint32_t *>(desc->rx_flags),
        static_cast<volatile std::uint32_t *>(desc->tx_flags));
    HIP_CHECK(hipGetLastError(), "kernel launch");
}

// Make sure we don't throw exception across C ABI.
template <DeviceDecodeFn DECODE> static int launch_noexcept(const CoprocLaunchDesc *desc) noexcept {
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

// Both launchers are built for this runtime's frame layout (see CoprocLaunchDesc).
static const CatalystCoprocessorFnInfo LAUNCHER_INFO{
    .abi_version = CATALYST_COPROCESSOR_FN_ABI_VERSION,
    .reserved = 0,
    .init = nullptr,
    .fini = nullptr,
};

extern "C" int gpu_echo_launcher(const CoprocLaunchDesc *desc, void *) {
    return launch_noexcept<echo_decode>(desc);
}
extern "C" const CatalystCoprocessorFnInfo *gpu_echo_launcher_info() { return &LAUNCHER_INFO; }

extern "C" int gpu_steane_launcher(const CoprocLaunchDesc *desc, void *) {
    return launch_noexcept<steane_decode>(desc);
}
extern "C" const CatalystCoprocessorFnInfo *gpu_steane_launcher_info() { return &LAUNCHER_INFO; }

int default_echo_launcher(const CoprocLaunchDesc *desc, void *ctx) {
    return gpu_echo_launcher(desc, ctx);
}

} // namespace catalyst::transport::coproc
