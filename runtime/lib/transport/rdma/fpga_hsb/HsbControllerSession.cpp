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

#include "HsbControllerSession.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib> // getenv (HWHS_RTT_WARMUP)
#include <cstring>
#include <ctime>
#include <vector>

#include "Error.hpp"
#include "TransportCAPI.h" // CATALYST_TRANSPORT_* status codes
#include "WireProtocol.hpp"
#include "hsb_regs.h"

#include <unistd.h>

namespace catalyst::transport::hsb {

namespace common = catalyst::transport::common;
namespace wire = catalyst::transport::common::hsb;

namespace {

constexpr int kOk = CATALYST_TRANSPORT_OK;
constexpr int kErrTimeout = CATALYST_TRANSPORT_ERR_TIMEOUT;
constexpr int kErrStuck = CATALYST_TRANSPORT_ERR_STUCK;

// Rounds before the engine's RING_IDX wraps to slot 0; slot and reply RAM hold HWHS_SLOTS.
constexpr std::uint32_t kRingLen = 256;
static_assert(kRingLen <= HWHS_SLOTS, "the engine ring must fit in slot RAM");

constexpr std::uint64_t kCollectTimeoutNs = 5'000'000'000ULL;
constexpr std::size_t kRttSamplesReserve = 3'000'000; // 24 MB
constexpr int kHsbResetTimeoutMs = 1000;

std::uint64_t now_ns() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ULL +
           static_cast<std::uint64_t>(ts.tv_nsec);
}

} // namespace

HsbControllerSession::HsbControllerSession(std::uint64_t reply_to_us, PacerCfg pacer)
    : reply_to_us_(reply_to_us), pacer_(std::move(pacer)),
      replies_(std::size_t{kRingLen} * HWHS_SLOT_BYTES, 0) {
    if (pacer_.enable) {
        TP_CHECK(pacer_.depth != 0 && pacer_.depth % HWHS_SLOT_BYTES == 0 &&
                     2ull * pacer_.depth <= HWHS_SIZE_DEMO,
                 "hsb: pacer_depth must be a non-zero multiple of 64 and at most %u",
                 HWHS_SIZE_DEMO / 2);
        TP_CHECK(pacer_.cnt != 0, "hsb: pacer_cnt must be non-zero");
        TP_CHECK((pacer_.span & (pacer_.span + 1u)) == 0,
                 "hsb: pacer_span must be 2^N-1 (all ones), e.g. 0xFFFFF");
    }
    rtt_samples_.reserve(kRttSamplesReserve);
}

HsbControllerSession::~HsbControllerSession() { stop(); }

std::uint64_t HsbControllerSession::cycles_to_ns(std::uint64_t cycles) const {
    return static_cast<std::uint64_t>(static_cast<double>(cycles) * 1e9 / dev_->clk_hz());
}

int HsbControllerSession::connect(const ConnectInfo &info) {
    dev_.emplace();
    std::fprintf(stderr, "[hsb] hwhs-hsb BUILD_ID 0x%08x, hif clock %u Hz\n", dev_->build_id(),
                 dev_->clk_hz());
    oob_ = common::tcp_connect(info.peer.c_str(), info.oob_port);
    return kOk;
}

MemRegion HsbControllerSession::alloc_memory(std::size_t size, MemKind kind) {
    // Nothing on this side is an RDMA target: HSB writes the coprocessor's ring and the replies
    // arrive through the engine. The core still wants a region, so it gets host memory.
    local_.assign(size, 0);
    return MemRegion{.addr = local_.data(), .size = size, .lkey = 0, .rkey = 0, .kind = kind};
}

PeerRef HsbControllerSession::exchange_keys(const MemRegion & /*local*/) {
    TP_CHECK(oob_.valid(), "hsb: exchange_keys before connect()");
    common::recv_exact(oob_.get(), &host_, sizeof(host_));
    TP_CHECK(host_.magic == wire::HS_MAGIC && host_.version == wire::HS_VERSION,
             "hsb: the coprocessor does not speak the HSB wire (magic 0x%08x version %u): give "
             "it peer=hsb",
             host_.magic, host_.version);

    wire::FpgaInfo fo{};
    fo.magic = wire::HS_MAGIC;
    fo.version = wire::HS_VERSION;
    fo.ip = HSB_DEFAULT_IP; // host port 0 after reset
    std::memcpy(fo.mac, HsbDevice::HSB_MAC0, sizeof(fo.mac));
    fo.qpn = wire::FPGA_RX_QPN;
    fo.frame_bytes = wire::FRAME_BYTES;
    fo.reply_bytes = wire::REPLY_BYTES;
    fo.seq_off = static_cast<std::uint32_t>(offsetof(common::Payload, seq_num));
    common::send_exact(oob_.get(), &fo, sizeof(fo));

    return PeerRef{.rkey = host_.rkey,
                   .remote_addr = host_.ring_iova,
                   .size = std::uint64_t{host_.ring_slots} * host_.ring_pitch};
}

void HsbControllerSession::establish_channel(const ChannelDesc &desc, const MemRegion & /*local*/,
                                             const PeerRef & /*peer*/) {
    TP_CHECK(desc.transport == "rdma", "hsb: only the \"rdma\" transport");
    wire::Ready rd{};
    common::recv_exact(oob_.get(), &rd, sizeof(rd));
    TP_CHECK(rd.magic == wire::HS_MAGIC, "hsb: the coprocessor did not get ready");

    // HSB from reset, so its slot pointer and PSN start at 0, then its data plane.
    dev_->hsb_reset(kHsbResetTimeoutMs);
    DataPlane dp{};
    dp.qpn = host_.qpn;
    dp.rkey = host_.rkey;
    dp.ring_iova = host_.ring_iova;
    dp.ring_pitch = host_.ring_pitch;
    dp.ring_slots = host_.ring_slots;
    dp.host_ip = host_.ip;
    std::memcpy(dp.host_mac, host_.mac, sizeof(dp.host_mac));
    dp.frame_bytes = wire::FRAME_BYTES;
    dp.imm = host_.imm != 0;
    dev_->hsb_configure(dp);

    // The engine: fresh counters, seq from 1, stamped into every request.
    const std::uint64_t reply_to = reply_to_us_ * dev_->clk_hz() / 1'000'000ULL;
    TP_CHECK(reply_to <= 0xFFFFFFFFULL, "hsb: reply_to_us %llu is too long for the engine",
             static_cast<unsigned long long>(reply_to_us_));
    dev_->wr(HWHS_CTRL, HWHS_CTRL_SOFT_RESET);
    dev_->wr(HWHS_CTRL, 0);
    dev_->wr(HWHS_REPLY_TO, static_cast<std::uint32_t>(reply_to));
    dev_->wr(HWHS_RING_LEN, kRingLen);
    dev_->wr(HWHS_FLAGS, 1); // seq_stamp
}

void HsbControllerSession::commit_work_item(std::uint32_t work_item_idx, std::uint64_t in_bytes,
                                            std::uint64_t out_bytes) {
    TP_CHECK(dev_.has_value(), "hsb: commit_work_item before connect()");
    TP_CHECK(work_item_idx == 0, "hsb: only a single work item (idx 0) is supported");
    TP_CHECK(in_bytes != 0 && in_bytes <= common::PAYLOAD_DATA_BYTES,
             "hsb: in_bytes %llu, want 1..%zu", static_cast<unsigned long long>(in_bytes),
             common::PAYLOAD_DATA_BYTES);
    TP_CHECK(out_bytes <= common::PAYLOAD_DATA_BYTES, "hsb: out_bytes %llu, want at most %zu",
             static_cast<unsigned long long>(out_bytes), common::PAYLOAD_DATA_BYTES);

    std::memset(request_, 0, sizeof(request_));
    std::fill(replies_.begin(), replies_.end(), 0);
    submitted_ = 0;
    collected_ = 0;
    armed_ = true;
    pacer_program();
}

void HsbControllerSession::write_data_slot(const void *src, std::uint64_t bytes,
                                           std::uint32_t decoder_id) {
    TP_CHECK(bytes <= common::PAYLOAD_DATA_BYTES, "hsb: write_data_slot: %llu bytes exceed %zu",
             static_cast<unsigned long long>(bytes), common::PAYLOAD_DATA_BYTES);
    if (pacer_.enable) {
        return; // the pacer sends from its own table
    }
    std::memset(request_, 0, common::PAYLOAD_DATA_BYTES);
    std::memcpy(request_, src, bytes);
    // seq_num is the engine's to stamp; decoder_id is the transport's to write.
    std::memcpy(request_ + offsetof(common::Payload, decoder_id), &decoder_id, sizeof(decoder_id));
}

void *HsbControllerSession::reply_slot() {
    return replies_.data() + (collected_ % kRingLen) * HWHS_SLOT_BYTES;
}

int HsbControllerSession::kick(std::uint32_t work_item_idx) {
    TP_CHECK(!pacer_.enable, "hsb: kick: pacer=hw issues its own rounds, use run_paced()");
    TP_CHECK(work_item_idx == 0, "hsb: only a single work item (idx 0) is supported");
    TP_CHECK(armed_, "hsb: kick before commit_work_item()");
    TP_CHECK(submitted_ == collected_,
             "hsb: kick: previous round still in flight, collect() first");

    // As hwhs_round: the engine sends slot RAM[RING_IDX] and replies into reply RAM[RING_IDX].
    round_idx_ = dev_->rd(HWHS_RING_IDX);
    done_before_ = dev_->rd(HWHS_DONE_CNT);
    timeouts_before_ = dev_->rd(HWHS_TIMEOUT_CNT);
    volatile std::uint8_t *slot = dev_->slot(round_idx_);
    HsbDevice::copy_to_dev64(slot, request_);
    // Slot RAM (LPD) and CTRL (FPD) are on different AXI paths: read the slot back so the payload
    // has landed before START can reach the engine.
    HsbDevice::io_barrier();
    (void)reinterpret_cast<volatile std::uint64_t *>(slot)[HWHS_SLOT_BYTES / 8 - 1];
    dev_->wr(HWHS_CTRL, HWHS_CTRL_START);
    ++submitted_;
    return kOk;
}

// The pacer issues every round from its table; the trace is the only record of them.
int HsbControllerSession::run_paced() {
    TP_CHECK(pacer_.enable, "hsb: run_paced needs pacer=hw");
    TP_CHECK(armed_, "hsb: run_paced before commit_work_item()");
    TP_CHECK(!pacer_armed_, "hsb: run_paced: one run per commit_work_item()");
    pacer_arm();
    pacer_armed_ = true;

    std::uint32_t last = dev_->rd(HWHS_ROUND_CNT);
    std::uint64_t stall_deadline = now_ns() + kCollectTimeoutNs;
    for (;;) {
        // The pacer reached pacer_cnt and stopped; the trace header and the "[hsb pacer]" lines
        // say how the run went.
        if (dev_->rd(HWHS_DEMO_STS) & 1u) {
            pacer_finished_ = true;
            return kOk;
        }
        const std::uint32_t rounds = dev_->rd(HWHS_ROUND_CNT);
        if (rounds != last) {
            last = rounds;
            stall_deadline = now_ns() + kCollectTimeoutNs;
        } else if (now_ns() >= stall_deadline) {
            std::fprintf(stderr, "[hsb pacer] run stalled: STATUS 0x%08x, %u round(s)\n",
                         dev_->rd(HWHS_STATUS), rounds - rounds_base_);
            return kErrStuck;
        }
        ::usleep(200); // nothing here is latency-critical
    }
}

int HsbControllerSession::collect(void *const *replies, const std::uint64_t *replies_bytes,
                                  std::size_t n) {
    TP_CHECK(armed_, "hsb: collect before commit_work_item()");
    TP_CHECK(submitted_ != collected_, "hsb: collect: nothing in flight");

    // DONE_CNT goes up when the engine finishes the round, with a reply or by timing out.
    const std::uint64_t deadline = now_ns() + reply_to_us_ * 1000ULL + kCollectTimeoutNs;
    while (dev_->rd(HWHS_DONE_CNT) == done_before_) {
        if (now_ns() >= deadline) {
            std::fprintf(stderr, "[hsb] round did not finish: STATUS 0x%08x\n",
                         dev_->rd(HWHS_STATUS));
            ++collected_;
            return kErrStuck;
        }
    }
    std::uint8_t *shadow = static_cast<std::uint8_t *>(reply_slot());
    ++collected_;
    if (dev_->rd(HWHS_TIMEOUT_CNT) != timeouts_before_) {
        return kErrTimeout;
    }

    const std::uint64_t rtt =
        dev_->rd(HWHS_RTT_LO) | (static_cast<std::uint64_t>(dev_->rd(HWHS_RTT_HI)) << 32);
    last_rtt_ns_ = cycles_to_ns(rtt);
    rtt_samples_.push_back(last_rtt_ns_);

    HsbDevice::copy_from_dev64(shadow, dev_->reply(round_idx_), sizeof(common::Payload));
    if (n > 0 && replies != nullptr && replies[0] != nullptr && replies[0] != shadow) {
        const std::uint64_t cap = replies_bytes ? replies_bytes[0] : common::PAYLOAD_DATA_BYTES;
        std::memcpy(replies[0], shadow, std::min<std::uint64_t>(cap, common::PAYLOAD_DATA_BYTES));
    }
    return kOk;
}

// Load the pacer's table and settings; run_paced() starts it.
void HsbControllerSession::pacer_program() {
    if (!pacer_.enable) {
        return;
    }
    const std::uint32_t entries = pacer_.depth / HWHS_SLOT_BYTES;
    std::vector<std::uint8_t> img(2 * std::size_t{pacer_.depth}, 0);
    if (!pacer_.table_path.empty()) {
        // a byte image: the syndromes, then the expected replies, pacer_depth bytes each
        std::FILE *f = std::fopen(pacer_.table_path.c_str(), "rb");
        TP_CHECK(f != nullptr, "hsb: pacer_table: cannot open %s", pacer_.table_path.c_str());
        const std::size_t got = std::fread(img.data(), 1, img.size(), f);
        const bool too_long = std::fgetc(f) != EOF;
        std::fclose(f);
        TP_CHECK(got == img.size() && !too_long,
                 "hsb: pacer_table %s is %zu bytes%s; pacer_depth=%u needs exactly %zu",
                 pacer_.table_path.c_str(), got, too_long ? "+" : "", pacer_.depth, img.size());
    } else {
        // Entry i: Payload{value = i + 1, decoder_id = 0, seq = i + 1}, which an echo returns.
        for (std::uint32_t half = 0; half < 2; ++half) {
            for (std::uint32_t i = 0; i < entries; ++i) {
                common::Payload p{};
                p.value = std::uint64_t{i} + 1;
                p.seq_num = i + 1;
                std::memcpy(img.data() + half * std::size_t{pacer_.depth} +
                                std::size_t{i} * HWHS_SLOT_BYTES,
                            &p, sizeof(p));
            }
        }
    }

    dev_->wr(HWHS_DEMO_CTRL, HWHS_DEMO_RESET);
    for (std::uint32_t i = 0; i < 2 * entries; ++i) {
        HsbDevice::copy_to_dev64(dev_->pacer() + std::size_t{i} * HWHS_SLOT_BYTES,
                                 img.data() + std::size_t{i} * HWHS_SLOT_BYTES);
    }
    HsbDevice::io_barrier();
    (void)reinterpret_cast<volatile std::uint64_t *>(dev_->pacer())[0];

    dev_->wr(HWHS_SYN_DEPTH, pacer_.depth);
    dev_->wr(HWHS_CMD_CNT, pacer_.cnt);
    dev_->wr(HWHS_FREQ_LO, static_cast<std::uint32_t>(pacer_.freq));
    dev_->wr(HWHS_FREQ_HI, static_cast<std::uint32_t>(pacer_.freq >> 32));
    dev_->wr(HWHS_FREQ_SPAN, pacer_.span);
    if (pacer_.seed != 0) {
        dev_->wr(HWHS_LFSR_SEED, pacer_.seed);
    }
    pacer_armed_ = false;
    pacer_finished_ = false;
}

void HsbControllerSession::pacer_arm() {
    rounds_base_ = dev_->rd(HWHS_ROUND_CNT);
    pacer_timeouts_base_ = dev_->rd(HWHS_TIMEOUT_CNT);
    dev_->wr(HWHS_TRACE_CTRL, 1); // clear, then record
    dev_->wr(HWHS_TRACE_CTRL, 0);
    dev_->wr(HWHS_DEMO_CTRL, 0); // reset_demo off
    HsbDevice::io_barrier();
    dev_->wr(HWHS_DEMO_CTRL, HWHS_DEMO_START);
}

void HsbControllerSession::pacer_report() const {
    const std::uint32_t rounds = dev_->rd(HWHS_ROUND_CNT) - rounds_base_;
    const std::uint32_t timeouts = dev_->rd(HWHS_TIMEOUT_CNT) - pacer_timeouts_base_;
    const std::uint32_t errors = dev_->rd(HWHS_STS1);
    std::fprintf(stderr, "[hsb pacer] engine ran %u round(s)%s\n", rounds,
                 pacer_finished_ ? "  [run complete]" : "");
    std::fprintf(stderr, "[hsb pacer] pacer_cnt=%u timeouts=%u compare_errors=%u%s\n", pacer_.cnt,
                 timeouts, errors, errors ? "   <-- DATA MISMATCH" : "");
}

// The trace RAM, as `cycles,ns`, in the hwhs backend's format with the hif clock.
void HsbControllerSession::pacer_dump_trace() const {
    if (pacer_.trace_out.empty()) {
        return;
    }
    const std::uint32_t cnt = std::min(dev_->rd(HWHS_TRACE_CNT), HWHS_TRACE_ENTRIES);
    const std::uint32_t sts = dev_->rd(HWHS_TRACE_STS);
    if (cnt == 0) {
        std::fprintf(stderr, "[hsb trace] nothing recorded\n");
        return;
    }
    std::FILE *f = std::fopen(pacer_.trace_out.c_str(), "w");
    if (f == nullptr) {
        std::fprintf(stderr, "[hsb trace] cannot write %s\n", pacer_.trace_out.c_str());
        return;
    }
    std::fprintf(f, "# rtt trace, one row per completed round, in order\n");
    std::fprintf(f, "# clk_hz=%u entries=%u full=%u saturated=%u\n", dev_->clk_hz(), cnt, sts & 1u,
                 (sts >> 1) & 1u);
    std::fprintf(f, "cycles,ns\n");
    for (std::uint32_t i = 0; i < cnt; ++i) {
        const std::uint16_t cyc = dev_->trace(i);
        std::fprintf(f, "%u,%llu\n", cyc, static_cast<unsigned long long>(cycles_to_ns(cyc)));
    }
    std::fclose(f);
    std::fprintf(stderr, "[hsb trace] wrote %u round(s) to %s%s%s\n", cnt, pacer_.trace_out.c_str(),
                 (sts & 1u) ? "   <-- buffer filled: rounds after this were not recorded" : "",
                 (sts & 2u) ? "   <-- some entry clamped to 0xFFFF: a ceiling, not a value" : "");
}

void HsbControllerSession::report_rtt() const {
    if (rtt_samples_.empty()) {
        return;
    }
    std::size_t warmup = 0;
    if (const char *w = std::getenv("HWHS_RTT_WARMUP")) {
        warmup = static_cast<std::size_t>(std::strtoull(w, nullptr, 10));
    }
    if (warmup >= rtt_samples_.size()) {
        warmup = 0;
    }
    std::vector<std::uint64_t> v(rtt_samples_.begin() + static_cast<std::ptrdiff_t>(warmup),
                                 rtt_samples_.end());
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    long double sum = 0.0L;
    for (std::uint64_t s : v) {
        sum += static_cast<long double>(s);
    }
    const auto pct = [&](double p) { return v[static_cast<std::size_t>(p * (n - 1))]; };
    std::fprintf(stderr,
                 "\n=== engine RTT (n=%zu, %zu warmup dropped, hwhs-hsb) ===\n"
                 "  min      %8llu ns\n  p50      %8llu ns\n  p95      %8llu ns\n"
                 "  p99      %8llu ns\n  p99.9    %8llu ns\n  max      %8llu ns\n"
                 "  mean     %8llu ns\n",
                 n, warmup, (unsigned long long)v.front(), (unsigned long long)pct(0.50),
                 (unsigned long long)pct(0.95), (unsigned long long)pct(0.99),
                 (unsigned long long)pct(0.999), (unsigned long long)v.back(),
                 (unsigned long long)(sum / n));
}

void HsbControllerSession::stop() {
    if (stopped_) {
        return;
    }
    stopped_ = true;
    report_rtt();
    if (dev_) {
        if (pacer_.enable && armed_) {
            pacer_report();
            pacer_dump_trace();
            dev_->wr(HWHS_DEMO_CTRL, HWHS_DEMO_RESET); // the engine leaves pacer mode
        }
        dev_->wr(HWHS_CTRL, HWHS_CTRL_ABORT);
        dev_->wr(HWHS_CTRL, 0);
        dev_->hsb_stop();
    }
    armed_ = false;
    dev_.reset();
    oob_.reset();
}

} // namespace catalyst::transport::hsb
