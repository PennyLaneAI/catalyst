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

#include "HsbLink.hpp"

#include <cstdio>
#include <cstring>
#include <string>

#include "Error.hpp"

#include <arpa/inet.h>

namespace catalyst::transport::common {

namespace {

// One line of a sysfs file, without its newline; empty if it cannot be read.
std::string read_sysfs(const std::string &path) {
    std::FILE *f = std::fopen(path.c_str(), "r");
    if (f == nullptr) {
        return {};
    }
    char buf[128] = {};
    const bool ok = std::fgets(buf, sizeof(buf), f) != nullptr;
    std::fclose(f);
    if (!ok) {
        return {};
    }
    buf[std::strcspn(buf, "\n")] = 0;
    return buf;
}

constexpr std::uint8_t V4_PREFIX[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};

} // namespace

HsbLink::HsbLink(std::string dev, int gid_idx) : dev_name_(std::move(dev)), gid_idx_(gid_idx) {}

void HsbLink::connect(std::uint16_t oob_port) {
    ctx_ = std::make_shared<Context>(dev_name_);
    pd_ = std::make_shared<ProtectionDomain>(ctx_);
    rcq_ = std::make_shared<CompletionQueue>(ctx_, 2 * MAX_RECV);
    scq_ = std::make_shared<CompletionQueue>(ctx_, 2 * MAX_SEND);
    qp_ = std::make_shared<QueuePair>(pd_, scq_, rcq_, MAX_SEND, /*max_inline=*/64, QpType::UC,
                                      MAX_RECV);
    qp_->to_init(port_);
    post_recvs(MAX_RECV);

    active_mtu_ = static_cast<std::uint32_t>(ctx_->port_attr(port_).active_mtu);

    // HSB speaks RoCE v2 over IPv4: the GID has to be an IPv4-mapped one, whose address and
    // netdev MAC are what HSB is told to send to.
    const ibv_gid gid = ctx_->gid(port_, gid_idx_);
    TP_CHECK(std::memcmp(gid.raw, V4_PREFIX, sizeof(V4_PREFIX)) == 0,
             "hsb: GID %d of %s is not an IPv4 GID; pick the RoCE v2 IPv4 one (gid=)", gid_idx_,
             dev_name_.c_str());
    ip_ = (static_cast<std::uint32_t>(gid.raw[12]) << 24) |
          (static_cast<std::uint32_t>(gid.raw[13]) << 16) |
          (static_cast<std::uint32_t>(gid.raw[14]) << 8) | gid.raw[15];
    const std::string attrs =
        "/sys/class/infiniband/" + dev_name_ + "/ports/" + std::to_string(port_) + "/gid_attrs/";
    const std::string type = read_sysfs(attrs + "types/" + std::to_string(gid_idx_));
    if (!type.empty() && type.find("v2") == std::string::npos) {
        std::fprintf(stderr, "[hsb] warning: GID %d is \"%s\"; HSB speaks RoCE v2\n", gid_idx_,
                     type.c_str());
    }
    const std::string netdev = read_sysfs(attrs + "ndevs/" + std::to_string(gid_idx_));
    TP_CHECK(!netdev.empty(), "hsb: no netdev behind GID %d of %s", gid_idx_, dev_name_.c_str());
    unsigned m[6] = {};
    const std::string mac = read_sysfs("/sys/class/net/" + netdev + "/address");
    TP_CHECK(std::sscanf(mac.c_str(), "%x:%x:%x:%x:%x:%x", &m[0], &m[1], &m[2], &m[3], &m[4],
                         &m[5]) == 6,
             "hsb: cannot read the MAC of %s", netdev.c_str());
    for (int i = 0; i < 6; ++i) {
        mac_[i] = static_cast<std::uint8_t>(m[i]);
    }

    oob_ = tcp_listen_accept(oob_port);
}

void HsbLink::exchange(std::uint32_t rkey, const frame::RingGeometry &ring) {
    TP_CHECK(oob_.valid(), "hsb: exchange before connect()");
    hsb::HostInfo hi{};
    hi.magic = hsb::HS_MAGIC;
    hi.version = hsb::HS_VERSION;
    hi.qpn = qp_->qpn();
    hi.rkey = rkey;
    hi.ring_iova = 0;
    hi.ring_pitch = ring.stride;
    hi.ring_slots = ring.slots;
    hi.ip = ip_;
    std::memcpy(hi.mac, mac_, sizeof(mac_));
    hi.imm = 1;
    send_exact(oob_.get(), &hi, sizeof(hi));
    recv_exact(oob_.get(), &fpga_, sizeof(fpga_));

    TP_CHECK(fpga_.magic == hsb::HS_MAGIC && fpga_.version == hsb::HS_VERSION,
             "hsb: bad handshake from the controller (magic 0x%08x version %u)", fpga_.magic,
             fpga_.version);
    // The sessions read and write a Payload: the request's seq is its seq_num, and the reply is
    // one Payload long.
    TP_CHECK(fpga_.seq_off == offsetof(Payload, seq_num) && fpga_.reply_bytes == sizeof(Payload) &&
                 fpga_.frame_bytes >= sizeof(Payload) && fpga_.frame_bytes <= ring.stride,
             "hsb: the controller asks for %u-byte frames, %u-byte replies and the seq at %u; "
             "this coprocessor speaks %zu-byte Payloads with the seq at %zu",
             fpga_.frame_bytes, fpga_.reply_bytes, fpga_.seq_off, sizeof(Payload),
             offsetof(Payload, seq_num));
}

void HsbLink::establish() {
    std::uint8_t dgid[16] = {};
    std::memcpy(dgid, V4_PREFIX, sizeof(V4_PREFIX));
    dgid[12] = static_cast<std::uint8_t>(fpga_.ip >> 24);
    dgid[13] = static_cast<std::uint8_t>(fpga_.ip >> 16);
    dgid[14] = static_cast<std::uint8_t>(fpga_.ip >> 8);
    dgid[15] = static_cast<std::uint8_t>(fpga_.ip);
    qp_->to_rtr(fpga_.qpn, /*dest_psn=*/0, dgid, gid_idx_, port_, active_mtu_);
    qp_->to_rts(/*sq_psn=*/0);

    const hsb::Ready rd{hsb::HS_MAGIC, IBV_QPS_RTS};
    send_exact(oob_.get(), &rd, sizeof(rd));
}

void HsbLink::send(const void *reply) {
    // The data is copied into the WQE (inline), so the lkey is never used.
    ibv_sge sge{
        .addr = reinterpret_cast<std::uint64_t>(reply),
        .length = fpga_.reply_bytes,
        .lkey = 0,
    };
    const bool signaled = (++sent_ % SIGNAL_EVERY) == 0;
    ibv_send_wr wr{};
    wr.wr_id = sent_;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_INLINE | (signaled ? IBV_SEND_SIGNALED : 0u);
    while (inflight_ >= MAX_SEND) { // only if idle reaping fell behind
        reap_send();
    }
    ibv_send_wr *bad = nullptr;
    const int rc = ibv_post_send(qp_->get(), &wr, &bad);
    TP_CHECK(rc == 0, "hsb: ibv_post_send rc=%d (%s)", rc, std::strerror(rc));
    ++inflight_;
}

void HsbLink::reap_send() {
    ibv_wc wc{};
    const int n = ibv_poll_cq(scq_->get(), 1, &wc);
    TP_CHECK(n >= 0, "hsb: ibv_poll_cq (send) rc=%d", n);
    if (n == 0) {
        return;
    }
    TP_CHECK(wc.status == IBV_WC_SUCCESS, "hsb: send completion %s", ibv_wc_status_str(wc.status));
    inflight_ -= SIGNAL_EVERY;
}

int HsbLink::reap_recv(std::uint32_t slots[RECV_BATCH]) {
    ibv_wc wc[RECV_BATCH];
    const int n = ibv_poll_cq(rcq_->get(), RECV_BATCH, wc);
    TP_CHECK(n >= 0, "hsb: ibv_poll_cq (recv) rc=%d", n);
    for (int i = 0; i < n; ++i) {
        TP_CHECK(wc[i].status == IBV_WC_SUCCESS && wc[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM,
                 "hsb: receive completion %s, opcode %d", ibv_wc_status_str(wc[i].status),
                 static_cast<int>(wc[i].opcode));
        // immediate = {psn[19:0], slot[11:0]}
        slots[i] = ntohl(wc[i].imm_data) & 0xFFFu;
    }
    if (n > 0) {
        post_recvs(static_cast<unsigned>(n));
    }
    return n;
}

void HsbLink::post_recvs(unsigned n) {
    // WRITE_WITH_IMM consumes a receive but scatters nothing into it: zero-SGE receives.
    for (unsigned i = 0; i < n; ++i) {
        ibv_recv_wr wr{};
        ibv_recv_wr *bad = nullptr;
        const int rc = ibv_post_recv(qp_->get(), &wr, &bad);
        TP_CHECK(rc == 0, "hsb: ibv_post_recv rc=%d (%s)", rc, std::strerror(rc));
    }
}

} // namespace catalyst::transport::common
