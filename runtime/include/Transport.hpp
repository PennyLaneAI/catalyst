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
enum class DataPath : std::uint8_t {
    CpuVerbs,    // Plain ibverbs on CPU.
    NicEngine,   // Hardware-embedded RNIC (e.g. ERNIC hardware handshake).
    GpuEngine,   // Gpu-initiated comms.
};

/**
 * @brief Memory kind: selects the allocation and registration path.
 */
enum class MemKind : std::uint8_t {
    CpuRam,   // Plain host RAM.
    GpuHbm,   // GPU HBM, registered via dma-buf.
    FpgaDdr,  // FPGA DDR, allocated via the Xilinx UMM allocator.
    FpgaBram, // FPGA on-chip block RAM.
};

struct ConnectInfo {
    std::string peer;
    std::uint16_t oob_port;
};
struct MemRegion {
    void *addr = nullptr;
    std::uint64_t size = 0;
    std::uint32_t lkey = 0;
    std::uint32_t rkey = 0;
    MemKind kind = MemKind::CpuRam;
};
struct PeerRef {
    std::uint32_t rkey = 0;
    std::uint64_t remote_addr = 0;
    std::uint64_t size = 0;
};
struct ChannelDesc {
    DataPath data_path = DataPath::CpuVerbs;
    bool persistent = true;
};

// Stateful session: methods must be called in this order:
//   1. connect            - bring up QPs + the out-of-band channel
//   2. alloc_memory       - register the region (needs the connected context)
//   3. exchange_keys      - swap region handles over the out-of-band channel
//   4. establish_channel  - program the channel from the local + peer regions
//   5. (coprocessor) set_decoder_schema / (controller) set_max_in_flight - before start()
//   6. start / collect / stop
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

// The decode I/O contract: a guarantee pinned by the frontend/compiler and
// carried down to the coprocessor.
struct DecoderSchema {
    std::uint64_t in_bytes = 0;  // syndrome length (decode input)
    std::uint64_t out_bytes = 0; // correction length (decode output)
};

// Controller role: writes syndrome out and receive correction.
class ControllerSession : public TransportSession {
  public:
    // Sliding-window depth: how many syndromes to keep in flight. Call before
    // start(). 1 = strict one-in-flight.
    virtual void set_max_in_flight(std::uint32_t n) = 0;
};

// Coprocessor role: receives syndromes, decodes, returns corrections. The decode
// definition is implemented on by the device; only the decode
// schema (I/O shape guarantee) is part of this contract.
class CoprocessorSession : public TransportSession {
  public:
    // Pin the decode I/O shape this coprocessor must honor. Call before start().
    virtual void set_decoder_schema(const DecoderSchema &schema) = 0;
};

} // namespace catalyst::transport
