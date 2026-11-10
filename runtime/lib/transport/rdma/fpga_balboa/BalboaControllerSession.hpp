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
 * @file
 * BalboaControllerSession - the VPK120 controller when the FPGA carries ETH Zurich's BALBOA RoCE
 * v2 stack and backline-roce's hwhs-balboa engine in front of it, in place of ERNIC and hwhs.
 *
 * The engine plays the same part as hwhs: it stamps the round's seq into the request, sends it,
 * recognizes the reply by that seq and times the round in its own clock. BALBOA turns the request
 * into an RC WRITE to the coprocessor and hands the reply WRITE back, so this session holds no
 * queue pair; it reaches everything through UIO and talks to the coprocessor over the
 * out-of-band socket only to set the session up (BalboaWire.hpp).
 *
 *   sw-loop: each kick() copies the request into slot RAM and starts one round; collect() waits
 *            for the engine to finish it and reads the reply and the RTT.
 *   pacer:   (`pacer=hw`) transport_demo, the PL pacer, issues every round from a table;
 *            run_paced() starts it and waits for the run to end, and stop() writes the RTT
 *            trace, the only record of the rounds, to `pacer_trace_out`.
 */

#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "BalboaDevice.hpp"
#include "BalboaWire.hpp"
#include "OobSocket.hpp"
#include "Transport.hpp"

namespace catalyst::transport::balboa {

/// The pacer's settings; the keys keep the spelling of the hwhs backend's `pacer_*`.
struct PacerCfg {
    bool enable = false;
    std::uint64_t freq = 0;    // net clock cycles between sends; 0 = back to back
    std::uint32_t cnt = 1;     // rounds per run
    std::uint32_t depth = 256; // syndrome table, bytes, a multiple of 64
    std::uint32_t span = 0;    // jitter mask, 2^N-1; 0 = fixed interval
    std::uint32_t seed = 0;    // LFSR seed; 0 = the engine's
    std::string trace_out;     // where to write the trace; empty = do not
    std::string table_path;    // a byte image of the table; empty = slot i holds i + 1
};

class BalboaControllerSession final : public ControllerSession {
  public:
    BalboaControllerSession(std::uint64_t reply_to_us, PacerCfg pacer);
    ~BalboaControllerSession() override;

    BalboaControllerSession(const BalboaControllerSession &) = delete;
    BalboaControllerSession &operator=(const BalboaControllerSession &) = delete;

    // TransportSession
    int connect(const ConnectInfo &info) override;
    MemRegion alloc_memory(std::size_t size, MemKind kind) override;
    PeerRef exchange_keys(const MemRegion &local) override;
    void establish_channel(const ChannelDesc &desc, const MemRegion &local,
                           const PeerRef &peer) override;
    void start() override {}
    int collect(void *const *replies, const std::uint64_t *replies_bytes, std::size_t n) override;
    void stop() override;
    std::uint64_t last_rtt_ns() const override { return last_rtt_ns_; }

    // ControllerSession
    void commit_work_item(std::uint32_t work_item_idx, std::uint64_t in_bytes,
                          std::uint64_t out_bytes) override;
    void *data_slot() override { return request_; }
    void write_data_slot(const void *src, std::uint64_t bytes, std::uint32_t decoder_id) override;
    void *reply_slot() override;
    int kick(std::uint32_t work_item_idx = 0) override;
    bool self_paced() const override { return pacer_.enable; }
    int run_paced() override;

  private:
    void pacer_program();
    void pacer_arm();
    void pacer_report() const;
    void pacer_dump_trace() const;
    void report_rtt() const;
    std::uint64_t cycles_to_ns(std::uint64_t cycles) const;

    std::uint64_t reply_to_us_;
    PacerCfg pacer_;

    std::optional<BalboaDevice> dev_;
    common::FdGuard oob_;
    common::balboa::HostInfo host_{};
    std::vector<std::uint8_t> local_; // what alloc_memory hands the core; BALBOA never reads it
    // Rounds before the engine's RING_IDX wraps to slot 0: the coprocessor's ring slots.
    std::uint32_t ring_len_ = 0;

    // The request is built here and copied into slot RAM at kick(); replies are copied out of
    // reply RAM into these slots, which is what reply_slot() hands out. Both are host memory, so
    // the program never touches a device window directly.
    alignas(64) std::uint8_t request_[HWHS_SLOT_BYTES] = {};
    std::vector<std::uint8_t> replies_; // one HWHS_SLOT_BYTES slot per engine ring slot

    bool armed_ = false;
    bool stopped_ = false;
    std::uint64_t submitted_ = 0;
    std::uint64_t collected_ = 0;
    // The round in flight: its slot and reply RAM index, and the counters before it started.
    std::uint32_t round_idx_ = 0;
    std::uint32_t done_before_ = 0;
    std::uint32_t timeouts_before_ = 0;
    std::uint64_t last_rtt_ns_ = 0;
    std::vector<std::uint64_t> rtt_samples_;

    // pacer
    bool pacer_armed_ = false;
    bool pacer_finished_ = false;
    std::uint32_t rounds_base_ = 0;
    std::uint32_t pacer_timeouts_base_ = 0;
};

} // namespace catalyst::transport::balboa
