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
    CpuVerbs,  // Plain ibverbs on CPU.
    GpuEngine, // Gpu-initiated comms.
    Other,
};

/**
 * @brief Memory kind: selects the allocation and registration path.
 */
enum class MemKind : std::uint8_t {
    CpuRam,
    GpuHbm,
    Ddr,
    Other,
};

/**
 * @brief Out-of-band connection parameters for bringing up a session.
 */
struct ConnectInfo {
    std::string peer;
    std::uint16_t oob_port;
};

/**
 * @brief Locally allocated and registered memory region.
 */
struct MemRegion {
    void *addr = nullptr;
    std::uint64_t size = 0;
    std::uint32_t lkey = 0;
    std::uint32_t rkey = 0;
    MemKind kind = MemKind::CpuRam;
};

/**
 * @brief Handle to a peer's memory region, exchanged over the out-of-band channel.
 */
struct PeerRef {
    std::uint32_t rkey = 0;
    std::uint64_t remote_addr = 0;
    std::uint64_t size = 0;
};

/**
 * @brief Configuration for the data-movement channel a session uses.
 */
struct ChannelDesc {
    DataPath data_path = DataPath::CpuVerbs;
};

/**
 * @brief Stateful transport session shared by the controller and coprocessor roles.
 *
 * Methods must be called in this order:
 *   1. connect           - bring up QPs + the out-of-band channel
 *   2. alloc_memory      - register the region (needs the connected context)
 *   3. exchange_keys     - swap region handles over the out-of-band channel
 *   4. establish_channel - program the channel from the local + peer regions
 *   5. (coprocessor) set_coprocessor_fn before start()
 *   6. start / collect / stop
 */
class TransportSession {
  public:
    virtual ~TransportSession() = default;

    /**
     * @brief Bring up the connection (out-of-band handshake and QP transition to RTS).
     *
     * @param info Peer address and out-of-band port.
     *
     * @return `int`
     */
    virtual int connect(const ConnectInfo &info) = 0;

    /**
     * @brief Allocate and register a memory region on the device.
     *
     * @param size Size of the region in bytes.
     * @param kind Memory kind selecting the allocation and registration path.
     * @param access Access flags for the registration.
     *
     * @return `MemRegion` The allocated and registered region.
     */
    virtual MemRegion alloc_memory(std::size_t size, MemKind kind, std::uint32_t access) = 0;

    /**
     * @brief Advertise a local region and receive the peer's region over the out-of-band channel.
     *
     * @param local The local region to advertise.
     *
     * @return `PeerRef` The peer's advertised region.
     */
    virtual PeerRef exchange_keys(const MemRegion &local) = 0;

    /**
     * @brief Program the data movement this session will run (single channel per session).
     *
     * @param desc Channel configuration.
     * @param local The local memory region.
     * @param peer The peer's memory region.
     */
    virtual void establish_channel(const ChannelDesc &desc, const MemRegion &local,
                                   const PeerRef &peer) = 0;

    /**
     * @brief Launch the engine (non-blocking; runs until stop()).
     */
    virtual void start() = 0;

    /**
     * @brief Wait for a result and write it out.
     *
     * @param outputs Array of output buffers to write into.
     * @param output_bytes Array of output buffer sizes.
     * @param n Number of output buffers.
     *
     * @return `int`
     */
    virtual int collect(void *const *outputs, const std::uint64_t *output_bytes, std::size_t n) = 0;

    /**
     * @brief Stop the engine and join. Idempotent.
     */
    virtual void stop() = 0;

    /**
     * @brief Last round-trip time, in nanoseconds (for testing purposes).
     *
     * @return `std::uint64_t`
     */
    virtual std::uint64_t last_rtt_ns() const { return 0; }
};

/**
 * @brief Controller role: writes messages out and receives corrections.
 */
class ControllerSession : public TransportSession {
  public:
    // Build the work item in slot `work_item_idx` from in_bytes and out_bytes.
    virtual void commit_work_item(std::uint32_t work_item_idx, std::uint64_t in_bytes,
                                  std::uint64_t out_bytes) = 0;

    // Fire one round using work item `work_item_idx` and whatever payload is currently in
    // data_slot(). Pairs with a subsequent collect(). Returns 0 on success.
    virtual int kick(std::uint32_t work_item_idx) = 0;

    // Current round's outbound slot in the transport-owned ring.
    virtual void *data_slot() = 0;
};

/**
 * @brief Opaque function to run on the coprocessor. May include a persistent kernel on the GPU.
 */
using CoprocessorFn = std::size_t (*)(const void *in, std::size_t in_len, void *out,
                                      std::size_t out_cap, void *ctx);

/**
 * @brief Coprocessor role: receives messages, process, and returns corrections.
 */
class CoprocessorSession : public TransportSession {
  public:
    /**
     * @brief Bind the coprocessor function this session runs.
     *
     * Call before start(). `fn` is a local function pointer; `ctx` is passed
     * back to `fn` on every invocation and may be null.
     *
     * @param fn The coprocessor function to run per received message.
     * @param ctx Opaque context passed to `fn` on each invocation; may be null.
     */
    virtual void set_coprocessor_fn(CoprocessorFn fn, void *ctx) = 0;
};

} // namespace catalyst::transport
