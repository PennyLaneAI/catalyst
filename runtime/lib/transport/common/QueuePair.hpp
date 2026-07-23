#pragma once
#include <cstdint>
#include <memory>

#include <infiniband/verbs.h>

#include "CompletionQueue.hpp"
#include "ProtectionDomain.hpp"
#include "QpState.hpp"

namespace rdma::devices::common {
class QueuePair {
  public:
    QueuePair(std::shared_ptr<ProtectionDomain> pd, std::shared_ptr<CompletionQueue> send_cq,
              std::shared_ptr<CompletionQueue> recv_cq, int max_send_wr, int max_inline = 0);
    ~QueuePair();
    QueuePair(const QueuePair &) = delete;
    QueuePair &operator=(const QueuePair &) = delete;

    ibv_qp *get() const;
    std::uint32_t qpn() const;
    QpState state() const;

    void to_init(std::uint8_t port);
    void to_rtr(std::uint32_t dest_qpn, std::uint32_t dest_psn, const std::uint8_t dest_gid[16],
                int sgid_idx, std::uint8_t port, std::uint32_t mtu_enum);
    void to_rts(std::uint32_t sq_psn);

  private:
    void check_transition(QpState to) const;
    void modify(QpState to, ibv_qp_attr &attr, int mask, const char *what);
    std::shared_ptr<ProtectionDomain> pd_;     // keeps PD + (transitively) Context alive
    std::shared_ptr<CompletionQueue> send_cq_; // keeps the CQ alive
    std::shared_ptr<CompletionQueue> recv_cq_;
    ibv_qp *qp_ = nullptr;
    QpState state_ = QpState::RESET;
};
} // namespace rdma::devices::common
