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

#include "BalboaLink.hpp"

#include <cstdio>
#include <cstring>
#include <random>
#include <string>

#include "Error.hpp"

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

BalboaLink::BalboaLink(std::string dev, int gid_idx)
    : dev_name_(std::move(dev)), gid_idx_(gid_idx) {}

void BalboaLink::connect(std::uint16_t oob_port) {
    ctx_ = std::make_shared<Context>(dev_name_);
    pd_ = std::make_shared<ProtectionDomain>(ctx_);
    cq_ = std::make_shared<CompletionQueue>(ctx_, 2 * MAX_SEND);
    // Nothing is ever received: BALBOA's requests and the replies are both RDMA WRITEs.
    qp_ = std::make_shared<QueuePair>(pd_, cq_, cq_, MAX_SEND, /*max_inline=*/64, QpType::RC,
                                      /*max_recv_wr=*/1);
    qp_->to_init(port_);

    active_mtu_ = static_cast<std::uint32_t>(ctx_->port_attr(port_).active_mtu);

    // BALBOA speaks RoCE v2 over IPv4: the GID has to be an IPv4-mapped one, whose address
    // BALBOA sends to and resolves with ARP.
    const ibv_gid gid = ctx_->gid(port_, gid_idx_);
    TP_CHECK(std::memcmp(gid.raw, V4_PREFIX, sizeof(V4_PREFIX)) == 0,
             "balboa: GID %d of %s is not an IPv4 GID; pick the RoCE v2 IPv4 one (gid=)", gid_idx_,
             dev_name_.c_str());
    ip_ = (static_cast<std::uint32_t>(gid.raw[12]) << 24) |
          (static_cast<std::uint32_t>(gid.raw[13]) << 16) |
          (static_cast<std::uint32_t>(gid.raw[14]) << 8) | gid.raw[15];
    const std::string attrs =
        "/sys/class/infiniband/" + dev_name_ + "/ports/" + std::to_string(port_) + "/gid_attrs/";
    const std::string type = read_sysfs(attrs + "types/" + std::to_string(gid_idx_));
    if (!type.empty() && type.find("v2") == std::string::npos) {
        std::fprintf(stderr, "[balboa] warning: GID %d is \"%s\"; BALBOA speaks RoCE v2\n",
                     gid_idx_, type.c_str());
    }
    const std::string netdev = read_sysfs(attrs + "ndevs/" + std::to_string(gid_idx_));
    TP_CHECK(!netdev.empty(), "balboa: no netdev behind GID %d of %s", gid_idx_, dev_name_.c_str());
    unsigned m[6] = {};
    const std::string mac = read_sysfs("/sys/class/net/" + netdev + "/address");
    TP_CHECK(std::sscanf(mac.c_str(), "%x:%x:%x:%x:%x:%x", &m[0], &m[1], &m[2], &m[3], &m[4],
                         &m[5]) == 6,
             "balboa: cannot read the MAC of %s", netdev.c_str());
    for (int i = 0; i < 6; ++i) {
        mac_[i] = static_cast<std::uint8_t>(m[i]);
    }
    psn_ = std::random_device{}() & 0xFFFFFFu;

    oob_ = tcp_listen_accept(oob_port);
}

void BalboaLink::exchange(std::uint32_t rkey, const frame::RingGeometry &ring) {
    TP_CHECK(oob_.valid(), "balboa: exchange before connect()");
    balboa::HostInfo hi{};
    hi.magic = balboa::HS_MAGIC;
    hi.version = balboa::HS_VERSION;
    hi.qpn = qp_->qpn();
    hi.psn = psn_;
    hi.rkey = rkey;
    hi.ring_slots = ring.slots;
    hi.ring_iova = 0;
    hi.ring_stride = ring.stride;
    hi.ip = ip_;
    std::memcpy(hi.mac, mac_, sizeof(mac_));
    send_exact(oob_.get(), &hi, sizeof(hi));
    recv_exact(oob_.get(), &fpga_, sizeof(fpga_));

    TP_CHECK(fpga_.magic == balboa::HS_MAGIC && fpga_.version == balboa::HS_VERSION,
             "balboa: bad handshake from the controller (magic 0x%08x version %u)", fpga_.magic,
             fpga_.version);
    // The sessions read and write a Payload: the request's seq is its seq_num, and the reply is
    // one Payload long.
    TP_CHECK(fpga_.seq_off == offsetof(Payload, seq_num) && fpga_.reply_bytes == sizeof(Payload) &&
                 fpga_.req_bytes >= sizeof(Payload) && fpga_.req_bytes <= ring.stride,
             "balboa: the controller asks for %u-byte requests, %u-byte replies and the seq at %u; "
             "this coprocessor speaks %zu-byte Payloads with the seq at %zu",
             fpga_.req_bytes, fpga_.reply_bytes, fpga_.seq_off, sizeof(Payload),
             offsetof(Payload, seq_num));
}

void BalboaLink::establish() {
    std::uint8_t dgid[16] = {};
    std::memcpy(dgid, V4_PREFIX, sizeof(V4_PREFIX));
    dgid[12] = static_cast<std::uint8_t>(fpga_.ip >> 24);
    dgid[13] = static_cast<std::uint8_t>(fpga_.ip >> 16);
    dgid[14] = static_cast<std::uint8_t>(fpga_.ip >> 8);
    dgid[15] = static_cast<std::uint8_t>(fpga_.ip);
    qp_->to_rtr(fpga_.qpn, fpga_.psn, dgid, gid_idx_, port_, active_mtu_);
    qp_->to_rts(fpga_.psn);

    const balboa::Ready rd{balboa::HS_MAGIC, IBV_QPS_RTS};
    send_exact(oob_.get(), &rd, sizeof(rd));
}

void BalboaLink::send(const void *reply) {
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
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_INLINE | (signaled ? IBV_SEND_SIGNALED : 0u);
    wr.wr.rdma.remote_addr = fpga_.reply_addr;
    wr.wr.rdma.rkey = fpga_.reply_rkey;
    while (inflight_ >= MAX_SEND) { // only if idle reaping fell behind
        reap_send();
    }
    ibv_send_wr *bad = nullptr;
    const int rc = ibv_post_send(qp_->get(), &wr, &bad);
    TP_CHECK(rc == 0, "balboa: ibv_post_send rc=%d (%s)", rc, std::strerror(rc));
    ++inflight_;
}

void BalboaLink::reap_send() {
    ibv_wc wc{};
    const int n = ibv_poll_cq(cq_->get(), 1, &wc);
    TP_CHECK(n >= 0, "balboa: ibv_poll_cq rc=%d", n);
    if (n == 0) {
        return;
    }
    TP_CHECK(wc.status == IBV_WC_SUCCESS, "balboa: WRITE completion %s",
             ibv_wc_status_str(wc.status));
    inflight_ -= SIGNAL_EVERY;
}

} // namespace catalyst::transport::common
