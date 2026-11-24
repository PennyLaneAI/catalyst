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

#include <cstring>
#include <random>
#include <utility>

#include "Error.hpp"

namespace catalyst::transport::common {

BalboaLink::BalboaLink(std::string dev, int gid_idx, frame::RingGeometry ring)
    : PeerLink(Peer::BALBOA, std::move(dev), gid_idx, ring) {}

void BalboaLink::connect(const ConnectInfo &info) {
    ctx_ = std::make_shared<Context>(dev_name_);
    pd_ = std::make_shared<ProtectionDomain>(ctx_);
    cq_ = std::make_shared<CompletionQueue>(ctx_, 2 * MAX_SEND);
    // Nothing is ever received: BALBOA's requests and the replies are both RDMA WRITEs.
    qp_ = std::make_shared<QueuePair>(pd_, cq_, cq_, MAX_SEND, /*max_inline=*/64);
    qp_->to_init(port_);
    active_mtu_ = static_cast<std::uint32_t>(ctx_->port_attr(port_).active_mtu);
    // BALBOA speaks RoCE v2 over IPv4, sends to the GID's address and resolves it with ARP.
    local_ = ctx_->ipv4_gid(port_, gid_idx_);
    psn_ = std::random_device{}() & 0xFFFFFFu;
    oob_ = tcp_listen_accept(info.oob_port);
}

PeerRef BalboaLink::exchange(const MemRegion &local) {
    TP_CHECK(oob_.valid(), "balboa: exchange before connect()");
    balboa::HostInfo hi{};
    hi.magic = balboa::HS_MAGIC;
    hi.version = balboa::HS_VERSION;
    hi.qpn = qp_->qpn();
    hi.psn = psn_;
    hi.rkey = local.rkey;
    hi.ring_slots = ring_.slots;
    hi.ring_iova = 0;
    hi.ring_stride = ring_.stride;
    hi.ip = local_.ip;
    std::memcpy(hi.mac, local_.mac, sizeof(local_.mac));
    send_exact(oob_.get(), &hi, sizeof(hi));
    recv_exact(oob_.get(), &fpga_, sizeof(fpga_));

    TP_CHECK(fpga_.magic == balboa::HS_MAGIC && fpga_.version == balboa::HS_VERSION,
             "balboa: bad handshake from the controller (magic 0x%08x version %u)", fpga_.magic,
             fpga_.version);
    // The session reads and writes a Payload: the request's seq is its seq_num, and the reply is
    // one Payload long.
    TP_CHECK(fpga_.seq_off == offsetof(Payload, seq_num) && fpga_.reply_bytes == sizeof(Payload) &&
                 fpga_.req_bytes >= sizeof(Payload) && fpga_.req_bytes <= ring_.stride,
             "balboa: the controller asks for %u-byte requests, %u-byte replies and the seq at %u; "
             "this coprocessor speaks %zu-byte Payloads with the seq at %zu",
             fpga_.req_bytes, fpga_.reply_bytes, fpga_.seq_off, sizeof(Payload),
             offsetof(Payload, seq_num));
    return PeerRef{};
}

void BalboaLink::establish(const MemRegion & /*local*/, const PeerRef & /*peer*/) {
    const ibv_gid dgid = ipv4_mapped_gid(fpga_.ip);
    qp_->to_rtr(fpga_.qpn, fpga_.psn, dgid.raw, gid_idx_, port_, active_mtu_);
    qp_->to_rts(fpga_.psn);

    const balboa::Ready rd{balboa::HS_MAGIC, IBV_QPS_RTS};
    send_exact(oob_.get(), &rd, sizeof(rd));
}

void BalboaLink::send(std::uint32_t seq) {
    frame::seal<frame::Mode::BL16>(ring_, &send_frame_, seq);
    // The data is copied into the WQE (inline), so the lkey is never used.
    ibv_sge sge{
        .addr = reinterpret_cast<std::uint64_t>(&send_frame_),
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
