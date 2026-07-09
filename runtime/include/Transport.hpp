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
    KernelFused, // Fused with a kernel on GPU (e.g. BlueFlame).
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
    std::size_t message_size = 0;
}; // advisory; a backend may use its own framing

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
};

} // namespace catalyst::transport
