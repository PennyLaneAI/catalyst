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

#include "QueuePair.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "Error.hpp"

namespace catalyst::transport::common {

namespace {
// RC QP tuning attributes for ibv_modify_qp (RTR/RTS). Encodings per the IB
// spec; see ibv_modify_qp(3).
constexpr std::uint8_t MIN_RNR_TIMER = 12; // RNR NAK timer code
constexpr std::uint8_t HOP_LIMIT = 64;     // GRH hop limit
constexpr std::uint8_t QP_TIMEOUT = 14;    // ACK timeout code
constexpr std::uint8_t RETRY_CNT = 7;      // transport retry count
constexpr std::uint8_t RNR_RETRY = 7;      // RNR retry count (7 = infinite)
constexpr std::uint8_t MAX_RD_ATOMIC = 1;  // outstanding RDMA read/atomic ops
constexpr std::uint32_t MAX_RECV_WR = 4;
constexpr std::uint32_t MAX_SGE = 1; // single-SGE descriptors on both queues
} // namespace

QueuePair::QueuePair(std::shared_ptr<ProtectionDomain> pd, std::shared_ptr<CompletionQueue> send_cq,
                     std::shared_ptr<CompletionQueue> recv_cq, int max_send_wr, int max_inline)
    : pd_(std::move(pd)), send_cq_(std::move(send_cq)), recv_cq_(std::move(recv_cq)) {
    ibv_qp_init_attr a{
        .send_cq = send_cq_->get(),
        .recv_cq = recv_cq_->get(),
        .cap =
            {
                .max_send_wr = static_cast<std::uint32_t>(max_send_wr),
                .max_recv_wr = MAX_RECV_WR,
                .max_send_sge = MAX_SGE,
                .max_recv_sge = MAX_SGE,
                .max_inline_data = static_cast<std::uint32_t>(max_inline),
            },
        .qp_type = IBV_QPT_RC,
        .sq_sig_all = 0,
    };
    qp_ = ibv_create_qp(pd_->get(), &a);
    TP_CHECK_ERRNO(qp_, "ibv_create_qp");
}

QueuePair::~QueuePair() {
    if (qp_) {
        ibv_destroy_qp(qp_);
    }
}

ibv_qp *QueuePair::get() const { return qp_; }
std::uint32_t QueuePair::qpn() const { return qp_->qp_num; }
QpState QueuePair::state() const { return state_; }

void QueuePair::check_transition(QpState to) const {
    if (!is_valid_transition(state_, to)) {
        char m[128];
        std::snprintf(m, sizeof(m), "invalid QP transition %s -> %s", to_string(state_),
                      to_string(to));
        throw BadTransition(m);
    }
}

void QueuePair::modify(QpState to, ibv_qp_attr &attr, int mask, const char *what) {
    check_transition(to);
    const int rc = ibv_modify_qp(qp_, &attr, mask);
    TP_CHECK(rc == 0, "%s rc=%d (%s)", what, rc, std::strerror(rc));
    state_ = to; // advance only after a successful modify
}

void QueuePair::to_init(std::uint8_t port) {
    ibv_qp_attr a{
        .qp_state = IBV_QPS_INIT,
        .qp_access_flags =
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ,
        .pkey_index = 0,
        .port_num = port,
    };
    modify(QpState::INIT, a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS,
           "modify_to_init");
}

void QueuePair::to_rtr(std::uint32_t dest_qpn, std::uint32_t dest_psn,
                       const std::uint8_t dest_gid[16], int sgid_idx, std::uint8_t port,
                       std::uint32_t mtu_enum) {
    ibv_qp_attr a{
        .qp_state = IBV_QPS_RTR,
        .path_mtu = static_cast<ibv_mtu>(mtu_enum),
        .rq_psn = dest_psn,
        .dest_qp_num = dest_qpn,
        .ah_attr =
            {
                .grh = {.sgid_index = static_cast<std::uint8_t>(sgid_idx), .hop_limit = HOP_LIMIT},
                .is_global = 1,
                .port_num = port,
            },
        .max_dest_rd_atomic = MAX_RD_ATOMIC,
        .min_rnr_timer = MIN_RNR_TIMER,
    };
    std::memcpy(&a.ah_attr.grh.dgid, dest_gid, 16);
    modify(QpState::RTR, a,
           IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
               IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER,
           "modify_to_rtr");
}

void QueuePair::to_rts(std::uint32_t sq_psn) {
    ibv_qp_attr a{
        .qp_state = IBV_QPS_RTS,
        .sq_psn = sq_psn,
        .max_rd_atomic = MAX_RD_ATOMIC,
        .timeout = QP_TIMEOUT,
        .retry_cnt = RETRY_CNT,
        .rnr_retry = RNR_RETRY,
    };
    modify(QpState::RTS, a,
           IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
               IBV_QP_MAX_QP_RD_ATOMIC,
           "modify_to_rts");
}

} // namespace catalyst::transport::common
