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
#include <stdexcept>
#include <string>

namespace catalyst::transport {

/**
 * @brief Memory kind: selects the allocation and registration path.
 */
enum class MemKind : int {
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
 * @brief Transport kind a session uses for request/reply movement.
 */
struct ChannelDesc {
    std::string transport = "rdma";
};

/**
 * @brief Stateful transport session shared by the controller and coprocessor roles.
 *
 * Methods must be called in this order:
 *   1. connect           - bring up QPs + the out-of-band channel
 *   2. alloc_memory      - register the region (needs the connected context)
 *   3. exchange_keys     - swap region handles over the out-of-band channel
 *   4. establish_channel - program the channel from the local + peer regions
 *   5. (coprocessor) bind a coprocessor function before start()
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
     *
     * @return `MemRegion` The allocated and registered region.
     */
    virtual MemRegion alloc_memory(std::size_t size, MemKind kind) = 0;

    /**
     * @brief Where this backend wants the regions the core provisions for it.
     *
     * The core does not know which memory a backend can register; a device-side one may accept
     * only its own. Reported here so the core asks for what the backend supports.
     *
     * @return `MemKind`
     */
    virtual MemKind preferred_mem_kind() const { return MemKind::CpuRam; }

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
     * @brief Wait for a result and scatter it into the reply buffers.
     *
     * @param replies Array of `n` buffers to write the results into.
     * @param replies_bytes Array of `n` capacities (bytes), one per reply buffer.
     * @param n Number of reply buffers.
     *
     * @return `int`
     */
    virtual int collect(void *const *replies, const std::uint64_t *replies_bytes,
                        std::size_t n) = 0;

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
 * @brief Controller role: writes messages out and receives replies.
 */
class ControllerSession : public TransportSession {
  public:
    // Build the work item in slot `work_item_idx` from in_bytes and out_bytes.
    virtual void commit_work_item(std::uint32_t work_item_idx, std::uint64_t in_bytes,
                                  std::uint64_t out_bytes) = 0;

    // Fire one round using work item `work_item_idx` and whatever payload is currently in
    // data_slot(). Pairs with a subsequent collect(). Returns 0 on success.
    virtual int kick(std::uint32_t work_item_idx) = 0;

    // Current round's outbound slot in the transport-owned ring. The slot's capacity is
    // backend-defined and not exposed here; prefer write_data_slot(), which enforces it.
    virtual void *data_slot() = 0;

    // Copy `bytes` of payload into the current round's outbound slot, ready for kick().
    // Throws if `bytes` exceeds what the round was committed to carry, so an oversized
    // payload will fail. `decoder_id` picks the coprocessor-side decoder for this round.
    virtual void write_data_slot(const void *src, std::uint64_t bytes,
                                 std::uint32_t decoder_id) = 0;

    // Current round's reply slot in the transport-owned reply ring.
    virtual void *reply_slot() { return nullptr; }
};

/**
 * @brief Per-message coprocessor function (CPU-style). Invoked once per received
 * message: decode `in` (`in_len` bytes) into `out` (capacity `out_cap`) and
 * return the number of bytes written.
 */
using CoprocessorFn = std::size_t (*)(const void *in, std::size_t in_len, void *out,
                                      std::size_t out_cap, void *ctx);

/**
 * @brief Data description for a persistent engine to receive and consume
 * request messages, then autonomously return processed message to the handoff buffer,
 * which is then replied by a coprocessor host thread.
 *
 * The session owns the buffers and keeps it valid until the engine has stopped
 * (set `*stop` nonzero, then synchronize) or has processed `total` messages.
 */
struct CoprocLaunchDesc {
    void *ring;               // recv request-slot ring base (device HBM or host RAM)
    std::uint32_t ring_slots; // slot count of both rings; must be a power of two
    void *handoff;            // reply-slot ring base (engine -> session, e.g. gpu to cpu)
    void *stop;               // uint32_t teardown flag the engine polls
    std::uint64_t total;      // messages to process, or 0 = run until stopped
    void *stream;             // device queue to launch on (e.g. hipStream_t); null for host
};

/**
 * @brief Launch-once coprocessor function (GPU-style). Invoked once at start():
 * launches a persistent worker on the given datapath. Returns 0 on a successful
 * launch, nonzero on failure.
 */
using CoprocessorLauncherFn = int (*)(const CoprocLaunchDesc *desc, void *ctx);

/**
 * @brief Which of the two bind methods below a coprocessor function is passed to.
 */
enum class CoprocConvention : std::int32_t {
    PerMessage = 0, // CoprocessorFn: host, invoked once per received message
    LaunchOnce = 1, // CoprocessorLauncherFn: launches a persistent engine
};

/**
 * @brief Coprocessor role: receives messages, process, and returns replies.
 *
 * A coprocessor function comes in two conventions:
 * - a host/cpu per-message function vs.
 * - a device kernel launched once.
 * Each backend overrides only the setter it supports; the other keeps
 * the throwing default, so a mis-bind fails loudly at bind time.
 */
class CoprocessorSession : public TransportSession {
  public:
    /**
     * @brief Bind a per-message coprocessor function (CPU-style).
     *
     * Call before start(). `fn` is invoked once per received message, receiving the
     * message's decoder_id so it can dispatch internally; `ctx` is passed back on
     * every invocation and may be null.
     */
    virtual void set_coprocessor_fn(CoprocessorFn /*fn*/, void * /*ctx*/) {
        throw std::logic_error(
            "transport: per-message coprocessor function not supported by this backend");
    }

    /**
     * @brief Bind a launch-once coprocessor function (GPU-style).
     *
     * Call before start(). `fn` is invoked once (in start()) to launch a
     * persistent engine on the session's datapath; `ctx` may be null.
     */
    virtual void set_coprocessor_launcher(CoprocessorLauncherFn /*fn*/, void * /*ctx*/) {
        throw std::logic_error(
            "transport: launch-once coprocessor function not supported by this backend");
    }

    /**
     * @brief Which of the two coprocessor function convention this backend takes.
     *
     * Defaults to per-message.
     */
    virtual CoprocConvention coprocessor_fn_convention() const {
        return CoprocConvention::PerMessage;
    }
};

} // namespace catalyst::transport
