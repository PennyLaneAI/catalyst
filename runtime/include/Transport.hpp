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
#include <memory>
#include <string>
#include <vector>

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
    CpuRam,  // Plain host RAM.
    GpuHbm,  // GPU HBM, registered via dma-buf.
    FpgaDdr, // FPGA DDR, allocated via the Xilinx UMM allocator.
    Bram,    // FPGA on-chip block RAM.
};

/**
 * @brief Host-provided connection configuration for the out-of-band handshake.
 */
struct ConnectInfo {
    std::string peer;       // Peer host[:port] for the out-of-band handshake.
    std::uint16_t oob_port; // Out-of-band TCP port.
    bool is_server;         // Which side listens on the OOB channel.
};

/**
 * @brief A memory region, allocated and registered by the device at runtime.
 *
 * The address and keys are produced by the device; the host
 * holds the region as an opaque handle.
 */
struct MemRegion {
    void *addr = nullptr;           // Device-allocated address.
    std::size_t size = 0;           // Region size in bytes.
    std::uint32_t lkey = 0;         // Local key.
    std::uint32_t rkey = 0;         // Remote key.
    MemKind kind = MemKind::CpuRam; // Memory kind / registration path.
    int device_id =
        -1; // For cases when we manage multiple devices (e.g. multi-GPU on a single host).
};

/**
 * @brief A peer's memory region, received over the out-of-band key exchange.
 *
 * The target of this endpoint's one-sided RDMA operations.
 */
struct PeerRef {
    std::uint32_t rkey = 0;        // Peer's remote key.
    std::uint64_t remote_addr = 0; // Peer's remote address.
    std::size_t size = 0;          // Peer region size in bytes.
};

/**
 * @brief A single RDMA operation in a channel's per-round schedule.
 */
struct RdmaOp {
    std::uint32_t opcode = 0;   // WRITE / READ / SEND.
    std::uint64_t xfer_len = 0; // Transfer length in bytes.
    bool inl = false;           // Inline payload (e.g. bf-inline small writes).
};

/**
 * @brief Describes a channel's data movement: the data-path engine that drives it, how many rounds
 *        to run, and the ordered RDMA operations it performs each round.
 */
struct ChannelDesc {
    DataPath data_path = DataPath::CpuVerbs; // Engine that drives the transfers.
    std::uint32_t n_rounds =
        0; // Rounds to run; 0 = persistent (run until close(), e.g. a GPU persistent kernel).
    std::vector<RdmaOp> ops; // The directed RDMA operations, in order.
};

/**
 * @brief Data-plane channel. Lifecycle only; the steady-state loop runs in the engine after
 * start().
 */
class Channel {
  public:
    virtual ~Channel() = default;

    /**
     * @brief Start the engine driving this channel's transfers.
     *
     * Non-blocking: after this call the engine runs autonomously and the CPU is out of the
     * per-round loop. A bounded channel runs `ChannelDesc::n_rounds` rounds; a persistent channel
     * (`n_rounds == 0`) runs until `close`.
     */
    virtual void start() = 0;

    /**
     * @brief Wait for the channel to reach its terminal state and write the result(s) out.
     *
     * @param outputs Array of memref data pointers to receive the terminal result(s).
     * @param n Number of output buffers.
     *
     * @return `int` Status (0 on success).
     */
    virtual int collect(void *const *outputs, std::size_t n) = 0;

    /**
     * @brief Tear down the channel, stopping a persistent engine if one is running.
     */
    virtual void close() = 0;
};

/**
 * @brief Control-plane transport session. Verbs-backed for every backend.
 *
 * A backend implements this interface via ibverbs. It runs on the host where its
 * hardware lives.
 */
class TransportSession {
  public:
    virtual ~TransportSession() = default;

    /**
     * @brief Bring up the connection (out-of-band handshake and QP transition to RTS).
     *
     * @param info Host-provided connection configuration.
     *
     * @return `int` Status (0 on success).
     */
    virtual int connect(const ConnectInfo &info) = 0;

    /**
     * @brief Allocate and register a memory region on the device.
     *
     * The device owns the allocation; the returned region's address and keys are outputs.
     *
     * @param size Region size in bytes.
     * @param kind Memory kind / registration path.
     * @param device_id Target device (e.g. GPU index).
     * @param access Access flags (map to `IBV_ACCESS_*`).
     *
     * @return `MemRegion` The allocated, registered region.
     */
    virtual MemRegion alloc_memory(std::size_t size, MemKind kind, int device_id,
                                   std::uint32_t access) = 0;

    /**
     * @brief Advertise a local region and receive the peer's region over the OOB channel.
     *
     * @param local The local region to advertise.
     * @param peer Index of the peer to exchange with.
     *
     * @return `PeerRef` The peer's region.
     */
    virtual PeerRef exchange_keys(const MemRegion &local, int peer) = 0;

    /**
     * @brief Program the transport descriptor and return the channel.
     *
     * @param desc The channel description (data path, rounds, RDMA ops).
     * @param local The local memory region.
     * @param peer The peer's region.
     *
     * @return `std::unique_ptr<Channel>` The established channel.
     */
    virtual std::unique_ptr<Channel>
    establish_channel(const ChannelDesc &desc, const MemRegion &local, const PeerRef &peer) = 0;
};

} // namespace catalyst::transport
