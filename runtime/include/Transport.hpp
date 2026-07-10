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
#include <string>

namespace catalyst::transport {

/**
 * @brief Data-plane strategy: which engine issues the transfer.
 */
enum class DataPath {
    CpuVerbs,    // Plain ibverbs on CPU.
    NicEngine,   // Hardware-embedded RNIC (e.g. ERNIC hardware handshake).
    KernelFused, // Fused with a kernel (e.g. on GPU). The decode runs
                 // inside this fused kernel (poll+decode as one GPU kernel), so
                 // set_decoder / DecodeFn are not used - the user/compiler
                 // supplies the fused kernel itself.
};

/**
 * @brief Memory kind: selects the allocation and registration path.
 */
enum class MemKind {
    CpuRam,   // Plain host RAM.
    GpuHbm,   // GPU HBM, registered via dma-buf.
    FpgaDdr,  // FPGA DDR, allocated via the Xilinx UMM allocator.
    FpgaBram, // FPGA on-chip block RAM.
};

/**
 * @brief Endpoint role in a session: the controller drives requests; the
 * coprocessor handles (e.g. runs the decoder) and returns them.
 */
enum class Role { Controller, Coprocessor };

struct ConnectInfo {
    std::string peer;
    std::uint16_t oob_port;
    Role role;
};
struct MemRegion {
    void *addr = nullptr;
    std::size_t size = 0;
    std::uint32_t lkey = 0;
    std::uint32_t rkey = 0;
    MemKind kind = MemKind::CpuRam;
};
struct PeerRef {
    std::uint32_t rkey = 0;
    std::uint64_t remote_addr = 0;
    std::size_t size = 0;
};
struct ChannelDesc {
    DataPath data_path = DataPath::CpuVerbs;
    bool persistent = true;
    std::size_t syndrome_bytes = 0;   // controller sends / coprocessor receives (decode in_len)
    std::size_t correction_bytes = 0; // coprocessor sends / controller receives (decode out_len)
};

// Per-shot compute run on the coprocessor. Reads the syndrome from `in`
// and writes the correction into `out`, in place on the transport's buffers.
// Should not throw error.
//
// This is the per-shot host-callback model, used by the per-message data paths
// (e.g. DataPath::CpuVerbs). A DataPath::KernelFused device does not use this:
// its poll+decode is a single fused GPU kernel supplied by the user/compiler,
// so no per-shot callback is registered or invoked.
using DecodeFn = void (*)(void *ctx, const void *in, std::size_t in_len, void *out,
                          std::size_t out_len);

class TransportSession {
  public:
    virtual ~TransportSession() = default;

    // Bring up the connection (out-of-band handshake and QP transition to RTS).
    virtual int connect(const ConnectInfo &info) = 0;

    // Allocate and register a memory region on the device.
    virtual MemRegion alloc_memory(std::size_t size, MemKind kind, std::uint32_t access) = 0;

    // Advertise a local region and receive the peer's region over the out-of-band channel.
    virtual PeerRef exchange_keys(const MemRegion &local) = 0;

    // Program the data movement this session will run (single channel per session).
    virtual void establish_channel(const ChannelDesc &desc, const MemRegion &local,
                                   const PeerRef &peer) = 0;

    // Launch the engine (non-blocking; runs until stop()).
    virtual void start() = 0;

    // Wait for a result and write it out.
    virtual int collect(void *const *outputs, std::size_t n) = 0;

    // Stop the engine and join. Idempotent.
    virtual void stop() = 0;

    // Register the per-shot decoder (coprocessor role). No-op default so
    // controller-only / non-decoding devices need not implement it. A
    // DataPath::KernelFused device also leaves this unused: its fused poll+decode
    // kernel is provided by the user/compiler, not registered here.
    virtual void set_decoder(DecodeFn /*fn*/, void * /*ctx*/) {}
};

} // namespace catalyst::transport
