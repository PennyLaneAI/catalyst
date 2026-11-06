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

#include "BalboaControllerSession.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib> // getenv (HWHS_RTT_WARMUP)
#include <cstring>
#include <ctime>
#include <vector>

#include "Error.hpp"
#include "TransportCAPI.h" // CATALYST_TRANSPORT_* status codes
#include "WireProtocol.hpp"

#include <unistd.h>

namespace catalyst::transport::balboa {

namespace common = catalyst::transport::common;
namespace wire = catalyst::transport::common::balboa;

namespace {

constexpr int kOk = CATALYST_TRANSPORT_OK;
constexpr int kErrTimeout = CATALYST_TRANSPORT_ERR_TIMEOUT;
constexpr int kErrStuck = CATALYST_TRANSPORT_ERR_STUCK;
// As the hwhs backend: the pacer reached demo_cnt and stopped, which ends a bounded run. The
// trace header and the "[balboa demo]" lines say how the run went.
constexpr int kDemoComplete = CATALYST_TRANSPORT_RUN_COMPLETE;

constexpr std::uint64_t kCollectTimeoutNs = 5'000'000'000ULL;
constexpr std::size_t kRttSamplesReserve = 3'000'000; // 24 MB
constexpr int kStackResetTimeoutMs = 1000;
constexpr int kArpTimeoutMs = 1000;

std::uint64_t now_ns() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ULL +
           static_cast<std::uint64_t>(ts.tv_nsec);
}

} // namespace

BalboaControllerSession::BalboaControllerSession(std::uint64_t reply_to_us, DemoCfg demo)
    : reply_to_us_(reply_to_us), demo_(std::move(demo)),
      replies_(std::size_t{HWHS_SLOTS} * HWHS_SLOT_BYTES, 0) {
    if (demo_.enable) {
        TP_CHECK(demo_.depth != 0 && demo_.depth % HWHS_SLOT_BYTES == 0 &&
                     2ull * demo_.depth <= HWHS_SIZE_DEMO,
                 "balboa: demo_depth must be a non-zero multiple of 64 and at most %u",
                 HWHS_SIZE_DEMO / 2);
        TP_CHECK(demo_.cnt != 0, "balboa: demo_cnt must be non-zero");
        TP_CHECK((demo_.span & (demo_.span + 1u)) == 0,
                 "balboa: demo_span must be 2^N-1 (all ones), e.g. 0xFFFFF");
    }
    rtt_samples_.reserve(kRttSamplesReserve);
}

BalboaControllerSession::~BalboaControllerSession() { stop(); }

std::uint64_t BalboaControllerSession::cycles_to_ns(std::uint64_t cycles) const {
    return static_cast<std::uint64_t>(static_cast<double>(cycles) * 1e9 / dev_->clk_hz());
}

int BalboaControllerSession::connect(const ConnectInfo &info) {
    dev_.emplace();
    std::fprintf(stderr, "[balboa] hwhs-balboa BUILD_ID 0x%08x, net clock %u Hz\n",
                 dev_->build_id(), dev_->clk_hz());
    oob_ = common::tcp_connect(info.peer.c_str(), info.oob_port);
    return kOk;
}

MemRegion BalboaControllerSession::alloc_memory(std::size_t size, MemKind kind) {
    // Nothing on this side is an RDMA target: BALBOA writes the coprocessor's ring and the replies
    // arrive through the engine. The core still wants a region, so it gets host memory.
    local_.assign(size, 0);
    return MemRegion{.addr = local_.data(), .size = size, .lkey = 0, .rkey = 0, .kind = kind};
}

PeerRef BalboaControllerSession::exchange_keys(const MemRegion & /*local*/) {
    TP_CHECK(oob_.valid(), "balboa: exchange_keys before connect()");
    common::recv_exact(oob_.get(), &host_, sizeof(host_));
    TP_CHECK(host_.magic == wire::HS_MAGIC && host_.version == wire::HS_VERSION,
             "balboa: the coprocessor does not speak the BALBOA wire (magic 0x%08x version %u): "
             "give it wire=balboa",
             host_.magic, host_.version);
    unsigned stride_log2 = 0;
    while ((1u << stride_log2) < host_.ring_stride) {
        ++stride_log2;
    }
    TP_CHECK(host_.ring_stride >= HWHS_SLOT_BYTES && (1u << stride_log2) == host_.ring_stride &&
                 stride_log2 <= 15 && host_.ring_slots != 0 && host_.ring_slots <= HWHS_SLOTS,
             "balboa: the coprocessor's ring is %u slots of %u B; the engine takes a power of two "
             ">= %u B per slot and at most %u slots",
             host_.ring_slots, host_.ring_stride, HWHS_SLOT_BYTES, HWHS_SLOTS);
    ring_len_ = host_.ring_slots;

    // BALBOA from reset (QP state, PSNs, ARP table), its address, the coprocessor's MAC, the QP.
    dev_->stack_reset(kStackResetTimeoutMs);
    dev_->set_addr(BalboaDevice::FPGA_IP, BalboaDevice::FPGA_MAC);
    if (!dev_->arp(host_.ip, kArpTimeoutMs)) {
        std::fprintf(stderr,
                     "[balboa] no ARP reply from %u.%u.%u.%u; is its RoCE netdev up, in "
                     "192.168.0.0/24?\n",
                     host_.ip >> 24, (host_.ip >> 16) & 255, (host_.ip >> 8) & 255, host_.ip & 255);
    }
    dev_->connect(Connection{.remote_qpn = host_.qpn,
                             .psn = host_.psn & 0xFFFFFFu,
                             .rkey = host_.rkey,
                             .ring_iova = host_.ring_iova,
                             .remote_ip = host_.ip});

    // The engine: fresh counters, seq from 1, stamped into every request, which goes into
    // ring slot RING_IDX on the coprocessor.
    const std::uint64_t reply_to = reply_to_us_ * dev_->clk_hz() / 1'000'000ULL;
    TP_CHECK(reply_to <= 0xFFFFFFFFULL, "balboa: reply_to_us %llu is too long for the engine",
             static_cast<unsigned long long>(reply_to_us_));
    dev_->wr(HWHS_CTRL, HWHS_CTRL_SOFT_RESET);
    dev_->wr(HWHS_CTRL, 0);
    dev_->wr(HWHS_REPLY_TO, static_cast<std::uint32_t>(reply_to));
    dev_->wr(HWHS_RING_LEN, ring_len_);
    dev_->wr(HWHS_FLAGS, 1); // seq_stamp
    dev_->wr(HWHS_LOCAL_QPN, BalboaDevice::FPGA_QPN);
    dev_->wr(HWHS_REQ_LEN, wire::FRAME_BYTES);
    dev_->wr(HWHS_RMT_BASE_LO, static_cast<std::uint32_t>(host_.ring_iova));
    dev_->wr(HWHS_RMT_BASE_HI, static_cast<std::uint32_t>(host_.ring_iova >> 32));
    dev_->wr(HWHS_RMT_STRIDE, stride_log2);
    const std::uint32_t seq_off = dev_->rd(HWHS_SEQ_OFF);
    TP_CHECK(seq_off == offsetof(common::Payload, seq_num),
             "balboa: the engine stamps the seq at byte %u; a Payload carries it at %zu", seq_off,
             offsetof(common::Payload, seq_num));

    wire::FpgaInfo fo{};
    fo.magic = wire::HS_MAGIC;
    fo.version = wire::HS_VERSION;
    fo.qpn = BalboaDevice::FPGA_QPN;
    fo.psn = host_.psn & 0xFFFFFFu;
    fo.ip = BalboaDevice::FPGA_IP;
    std::memcpy(fo.mac, BalboaDevice::FPGA_MAC, sizeof(fo.mac));
    fo.reply_rkey = 0; // the engine takes every WRITE, wherever it points
    fo.reply_addr = 0;
    fo.req_bytes = wire::FRAME_BYTES;
    fo.reply_bytes = wire::REPLY_BYTES;
    fo.seq_off = seq_off;
    common::send_exact(oob_.get(), &fo, sizeof(fo));

    return PeerRef{.rkey = host_.rkey,
                   .remote_addr = host_.ring_iova,
                   .size = std::uint64_t{host_.ring_slots} * host_.ring_stride};
}

void BalboaControllerSession::establish_channel(const ChannelDesc &desc,
                                                const MemRegion & /*local*/,
                                                const PeerRef & /*peer*/) {
    TP_CHECK(desc.transport == "rdma", "balboa: only the \"rdma\" transport");
    wire::Ready rd{};
    common::recv_exact(oob_.get(), &rd, sizeof(rd));
    TP_CHECK(rd.magic == wire::HS_MAGIC, "balboa: the coprocessor did not get ready");
}

void BalboaControllerSession::commit_work_item(std::uint32_t work_item_idx, std::uint64_t in_bytes,
                                               std::uint64_t out_bytes) {
    TP_CHECK(dev_.has_value(), "balboa: commit_work_item before connect()");
    TP_CHECK(work_item_idx == 0, "balboa: only a single work item (idx 0) is supported");
    TP_CHECK(in_bytes != 0 && in_bytes <= common::PAYLOAD_DATA_BYTES,
             "balboa: in_bytes %llu, want 1..%zu", static_cast<unsigned long long>(in_bytes),
             common::PAYLOAD_DATA_BYTES);
    TP_CHECK(out_bytes <= common::PAYLOAD_DATA_BYTES, "balboa: out_bytes %llu, want at most %zu",
             static_cast<unsigned long long>(out_bytes), common::PAYLOAD_DATA_BYTES);

    std::memset(request_, 0, sizeof(request_));
    std::fill(replies_.begin(), replies_.end(), 0);
    submitted_ = 0;
    collected_ = 0;
    armed_ = true;
    demo_program();
}

void BalboaControllerSession::write_data_slot(const void *src, std::uint64_t bytes,
                                              std::uint32_t decoder_id) {
    TP_CHECK(bytes <= common::PAYLOAD_DATA_BYTES, "balboa: write_data_slot: %llu bytes exceed %zu",
             static_cast<unsigned long long>(bytes), common::PAYLOAD_DATA_BYTES);
    if (demo_.enable) {
        return; // the pacer sends from its own table
    }
    std::memset(request_, 0, common::PAYLOAD_DATA_BYTES);
    std::memcpy(request_, src, bytes);
    // seq_num is the engine's to stamp; decoder_id is the transport's to write.
    std::memcpy(request_ + offsetof(common::Payload, decoder_id), &decoder_id, sizeof(decoder_id));
}

void *BalboaControllerSession::reply_slot() {
    return replies_.data() + (collected_ % ring_len_) * HWHS_SLOT_BYTES;
}

int BalboaControllerSession::kick(std::uint32_t work_item_idx) {
    TP_CHECK(work_item_idx == 0, "balboa: only a single work item (idx 0) is supported");
    TP_CHECK(armed_, "balboa: kick before commit_work_item()");
    TP_CHECK(submitted_ == collected_,
             "balboa: kick: previous round still in flight, collect() first");

    if (demo_.enable) {
        if (!demo_armed_) {
            demo_arm();
            demo_armed_ = true;
        }
        ++submitted_;
        return kOk;
    }

    // As hwhs_round: the engine sends slot RAM[RING_IDX] and replies into reply RAM[RING_IDX].
    round_idx_ = dev_->rd(HWHS_RING_IDX);
    done_before_ = dev_->rd(HWHS_DONE_CNT);
    timeouts_before_ = dev_->rd(HWHS_TIMEOUT_CNT);
    volatile std::uint8_t *slot = dev_->slot(round_idx_);
    BalboaDevice::copy_to_dev64(slot, request_);
    // Slot RAM (LPD) and CTRL (FPD) are on different AXI paths: read the slot back so the payload
    // has landed before START can reach the engine.
    BalboaDevice::io_barrier();
    (void)reinterpret_cast<volatile std::uint64_t *>(slot)[HWHS_SLOT_BYTES / 8 - 1];
    dev_->wr(HWHS_CTRL, HWHS_CTRL_START);
    ++submitted_;
    return kOk;
}

int BalboaControllerSession::collect(void *const *replies, const std::uint64_t *replies_bytes,
                                     std::size_t n) {
    TP_CHECK(armed_, "balboa: collect before commit_work_item()");
    TP_CHECK(submitted_ != collected_, "balboa: collect: nothing in flight");

    if (demo_.enable) { // the trace is the only record of a paced run
        std::uint32_t last = dev_->rd(HWHS_ROUND_CNT);
        std::uint64_t stall_deadline = now_ns() + kCollectTimeoutNs;
        for (;;) {
            if (dev_->rd(HWHS_DEMO_STS) & 1u) {
                demo_finished_ = true;
                ++collected_;
                return kDemoComplete;
            }
            const std::uint32_t rounds = dev_->rd(HWHS_ROUND_CNT);
            if (rounds != last) {
                last = rounds;
                stall_deadline = now_ns() + kCollectTimeoutNs;
            } else if (now_ns() >= stall_deadline) {
                std::fprintf(stderr, "[balboa demo] run stalled: STATUS 0x%08x, %u round(s)\n",
                             dev_->rd(HWHS_STATUS), rounds - rounds_base_);
                ++collected_;
                return kErrStuck;
            }
            ::usleep(200); // nothing here is latency-critical
        }
    }

    // DONE_CNT goes up when the engine finishes the round, with a reply or by timing out.
    const std::uint64_t deadline = now_ns() + reply_to_us_ * 1000ULL + kCollectTimeoutNs;
    while (dev_->rd(HWHS_DONE_CNT) == done_before_) {
        if (now_ns() >= deadline) {
            std::fprintf(stderr, "[balboa] round did not finish: STATUS 0x%08x\n",
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

    BalboaDevice::copy_from_dev64(shadow, dev_->reply(round_idx_), sizeof(common::Payload));
    if (n > 0 && replies != nullptr && replies[0] != nullptr && replies[0] != shadow) {
        const std::uint64_t cap = replies_bytes ? replies_bytes[0] : common::PAYLOAD_DATA_BYTES;
        std::memcpy(replies[0], shadow, std::min<std::uint64_t>(cap, common::PAYLOAD_DATA_BYTES));
    }
    return kOk;
}

// Load the pacer's table and settings; it starts at the first kick().
void BalboaControllerSession::demo_program() {
    if (!demo_.enable) {
        return;
    }
    const std::uint32_t entries = demo_.depth / HWHS_SLOT_BYTES;
    std::vector<std::uint8_t> img(2 * std::size_t{demo_.depth}, 0);
    if (!demo_.table_path.empty()) {
        // a byte image: the syndromes, then the expected replies, demo_depth bytes each
        std::FILE *f = std::fopen(demo_.table_path.c_str(), "rb");
        TP_CHECK(f != nullptr, "balboa: demo_table: cannot open %s", demo_.table_path.c_str());
        const std::size_t got = std::fread(img.data(), 1, img.size(), f);
        const bool too_long = std::fgetc(f) != EOF;
        std::fclose(f);
        TP_CHECK(got == img.size() && !too_long,
                 "balboa: demo_table %s is %zu bytes%s; demo_depth=%u needs exactly %zu",
                 demo_.table_path.c_str(), got, too_long ? "+" : "", demo_.depth, img.size());
    } else {
        // Entry i: Payload{value = i + 1, decoder_id = 0, seq = i + 1}, which an echo returns.
        for (std::uint32_t half = 0; half < 2; ++half) {
            for (std::uint32_t i = 0; i < entries; ++i) {
                common::Payload p{};
                p.value = std::uint64_t{i} + 1;
                p.seq_num = i + 1;
                std::memcpy(img.data() + half * std::size_t{demo_.depth} +
                                std::size_t{i} * HWHS_SLOT_BYTES,
                            &p, sizeof(p));
            }
        }
    }

    dev_->wr(HWHS_DEMO_CTRL, HWHS_DEMO_RESET);
    for (std::uint32_t i = 0; i < 2 * entries; ++i) {
        BalboaDevice::copy_to_dev64(dev_->demo() + std::size_t{i} * HWHS_SLOT_BYTES,
                                    img.data() + std::size_t{i} * HWHS_SLOT_BYTES);
    }
    BalboaDevice::io_barrier();
    (void)reinterpret_cast<volatile std::uint64_t *>(dev_->demo())[0];

    dev_->wr(HWHS_SYN_DEPTH, demo_.depth);
    dev_->wr(HWHS_CMD_CNT, demo_.cnt);
    dev_->wr(HWHS_FREQ_LO, static_cast<std::uint32_t>(demo_.freq));
    dev_->wr(HWHS_FREQ_HI, static_cast<std::uint32_t>(demo_.freq >> 32));
    dev_->wr(HWHS_FREQ_SPAN, demo_.span);
    if (demo_.seed != 0) {
        dev_->wr(HWHS_LFSR_SEED, demo_.seed);
    }
    demo_armed_ = false;
    demo_finished_ = false;
}

void BalboaControllerSession::demo_arm() {
    rounds_base_ = dev_->rd(HWHS_ROUND_CNT);
    demo_timeouts_base_ = dev_->rd(HWHS_TIMEOUT_CNT);
    dev_->wr(HWHS_TRACE_CTRL, 1); // clear, then record
    dev_->wr(HWHS_TRACE_CTRL, 0);
    dev_->wr(HWHS_DEMO_CTRL, 0); // reset_demo off
    BalboaDevice::io_barrier();
    dev_->wr(HWHS_DEMO_CTRL, HWHS_DEMO_START);
}

void BalboaControllerSession::demo_report() const {
    const std::uint32_t rounds = dev_->rd(HWHS_ROUND_CNT) - rounds_base_;
    const std::uint32_t timeouts = dev_->rd(HWHS_TIMEOUT_CNT) - demo_timeouts_base_;
    const std::uint32_t errors = dev_->rd(HWHS_STS1);
    std::fprintf(stderr, "[balboa demo] engine ran %u round(s)%s\n", rounds,
                 demo_finished_ ? "  [run complete]" : "");
    std::fprintf(stderr, "[balboa demo] demo_cnt=%u timeouts=%u compare_errors=%u%s\n", demo_.cnt,
                 timeouts, errors, errors ? "   <-- DATA MISMATCH" : "");
}

// The trace RAM, as `cycles,ns`, in the hwhs backend's format with the net clock.
void BalboaControllerSession::demo_dump_trace() const {
    if (demo_.trace_out.empty()) {
        return;
    }
    const std::uint32_t cnt = std::min(dev_->rd(HWHS_TRACE_CNT), HWHS_TRACE_ENTRIES);
    const std::uint32_t sts = dev_->rd(HWHS_TRACE_STS);
    if (cnt == 0) {
        std::fprintf(stderr, "[balboa trace] nothing recorded\n");
        return;
    }
    std::FILE *f = std::fopen(demo_.trace_out.c_str(), "w");
    if (f == nullptr) {
        std::fprintf(stderr, "[balboa trace] cannot write %s\n", demo_.trace_out.c_str());
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
    std::fprintf(stderr, "[balboa trace] wrote %u round(s) to %s%s%s\n", cnt,
                 demo_.trace_out.c_str(),
                 (sts & 1u) ? "   <-- buffer filled: rounds after this were not recorded" : "",
                 (sts & 2u) ? "   <-- some entry clamped to 0xFFFF: a ceiling, not a value" : "");
}

void BalboaControllerSession::report_rtt() const {
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
                 "\n=== engine RTT (n=%zu, %zu warmup dropped, hwhs-balboa) ===\n"
                 "  min      %8llu ns\n  p50      %8llu ns\n  p95      %8llu ns\n"
                 "  p99      %8llu ns\n  p99.9    %8llu ns\n  max      %8llu ns\n"
                 "  mean     %8llu ns\n",
                 n, warmup, (unsigned long long)v.front(), (unsigned long long)pct(0.50),
                 (unsigned long long)pct(0.95), (unsigned long long)pct(0.99),
                 (unsigned long long)pct(0.999), (unsigned long long)v.back(),
                 (unsigned long long)(sum / n));
}

void BalboaControllerSession::stop() {
    if (stopped_) {
        return;
    }
    stopped_ = true;
    report_rtt();
    if (dev_) {
        if (demo_.enable && armed_) {
            demo_report();
            demo_dump_trace();
            dev_->wr(HWHS_DEMO_CTRL, HWHS_DEMO_RESET); // the engine leaves demo mode
        }
        dev_->wr(HWHS_CTRL, HWHS_CTRL_ABORT);
        dev_->wr(HWHS_CTRL, 0);
    }
    armed_ = false;
    dev_.reset();
    oob_.reset();
}

} // namespace catalyst::transport::balboa
