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
#include <cstdint>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <vector>

#include "CompletionQueue.hpp"
#include "Context.hpp"
#include "MemoryRegion.hpp"
#include "OobSocket.hpp"
#include "ProtectionDomain.hpp"
#include "QueuePair.hpp"
#include "Transport.hpp"
#include "WireProtocol.hpp"

namespace catalyst::transport::cpu_verbs {
using namespace catalyst::transport;

/**
 * @brief Connection bring-up and data-path primitives shared by both roles.
 *
 * Parameterised on the role interface (`ControllerSession` or
 * `CoprocessorSession`) so each role derives from these directly.
 */
template <class Role> class CpuSessionBase : public Role {
  public:
    explicit CpuSessionBase(std::string dev, int gid_idx);
    ~CpuSessionBase() override = default;

    CpuSessionBase(const CpuSessionBase &) = delete;
    CpuSessionBase &operator=(const CpuSessionBase &) = delete;
    CpuSessionBase(CpuSessionBase &&) = delete;
    CpuSessionBase &operator=(CpuSessionBase &&) = delete;

    int connect(const ConnectInfo &info) override;
    MemRegion alloc_memory(std::size_t size, MemKind kind) override;
    PeerRef exchange_keys(const MemRegion &local) override;
    void establish_channel(const ChannelDesc &desc, const MemRegion &local,
                           const PeerRef &peer) override;

  protected:
    // True for the coprocessor (listens/sends first on the OOB socket).
    virtual bool oob_listens() const = 0;

    void post_write(ibv_qp *qp, std::uint64_t cursor, bool inline_data, bool signaled);
    void reap(ibv_cq *cq, int &outstanding, bool drain);
    common::Payload *poll_message_arrival(std::uint64_t cursor, std::stop_token st);
    common::Payload *send_payload() {
        return reinterpret_cast<common::Payload *>(send_buf_->addr());
    }

    std::string dev_name_;
    int gid_idx_;
    ibv_gid mygid_{};              // local GID, cached in connect() for exchange_keys()
    std::uint32_t active_mtu_ = 0; // local active_mtu enum, cached in connect()
    std::shared_ptr<common::Context> ctx_;
    std::shared_ptr<common::ProtectionDomain> pd_;
    std::shared_ptr<common::CompletionQueue> fwd_cq_, bwd_cq_;
    std::shared_ptr<common::QueuePair> fwd_qp_, bwd_qp_;
    common::FdGuard oob_fd_;
    std::vector<common::MemoryRegion> caller_memory_regions_;
    std::optional<common::MemoryRegion> send_buf_; // local send source (one Payload)
    MemRegion local_{};
    PeerRef peer_{};
    ChannelDesc desc_{};
};

extern template class CpuSessionBase<ControllerSession>;
extern template class CpuSessionBase<CoprocessorSession>;

} // namespace catalyst::transport::cpu_verbs
