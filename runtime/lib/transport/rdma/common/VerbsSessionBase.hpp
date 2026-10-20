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
#include "Frame.hpp"
#include "MemoryRegion.hpp"
#include "OobSocket.hpp"
#include "ProtectionDomain.hpp"
#include "QueuePair.hpp"
#include "Transport.hpp"
#include "WireProtocol.hpp"

namespace catalyst::transport::common {
using namespace catalyst::transport;

/// Sizes of one queue pair: send-queue depth, completion-queue depth and inline capacity.
struct QpShape {
    int sq_depth;
    int cq_depth;
    int inline_max;
};

/// The shape both queue pairs get unless a backend asks otherwise: deep enough for a full ring of
/// signaled sends, with inline capacity for a Payload.
inline constexpr QpShape DEFAULT_QP_SHAPE{.sq_depth = 4096, .cq_depth = 4096, .inline_max = 256};

/**
 * @brief Connection bring-up and data-path primitives shared by the libibverbs backends.
 *
 * Parameterised on the role interface (`ControllerSession` or
 * `CoprocessorSession`) so each role derives from these directly. cpu_verbs uses it as is;
 * gpu_verbs replaces the memory allocation with GPU memory and shrinks the fwd queue pair, which
 * its coprocessor only receives on.
 */
template <class Role> class VerbsSessionBase : public Role {
  public:
    /// @param backend Name used in error messages, e.g. "cpu_verbs".
    /// @param fwd_shape Shape of the controller-to-coprocessor queue pair.
    VerbsSessionBase(std::string dev, int gid_idx, const char *backend,
                     QpShape fwd_shape = DEFAULT_QP_SHAPE);
    ~VerbsSessionBase() override = default;

    VerbsSessionBase(const VerbsSessionBase &) = delete;
    VerbsSessionBase &operator=(const VerbsSessionBase &) = delete;
    VerbsSessionBase(VerbsSessionBase &&) = delete;
    VerbsSessionBase &operator=(VerbsSessionBase &&) = delete;

    int connect(const ConnectInfo &info) override;
    MemRegion alloc_memory(std::size_t size, MemKind kind) override;
    PeerRef exchange_keys(const MemRegion &local) override;
    void establish_channel(const ChannelDesc &desc, const MemRegion &local,
                           const PeerRef &peer) override;

  protected:
    // True for the coprocessor (listens/sends first on the OOB socket).
    virtual bool oob_listens() const = 0;

    // Seal the staged send frame with round `cursor`'s seq and write it into the peer's ring.
    void post_write(ibv_qp *qp, std::uint64_t cursor, bool inline_data, bool signaled);
    void reap(ibv_cq *cq, int &outstanding, bool drain);
    // Wait for round `cursor`'s frame in the local ring and decode it into `out`. Returns false if
    // `st` is stopped first.
    bool poll_message_arrival(std::uint64_t cursor, std::stop_token st, frame::FrameView &out);
    // The frame the next post_write sends; stage it with frame::stage first.
    void *send_frame() { return send_buf_->addr(); }
    void *ring_slot(std::uint64_t cursor) {
        return static_cast<std::uint8_t *>(local_.addr) + ring_.slot_offset(cursor);
    }

    std::string dev_name_;
    int gid_idx_;
    const char *backend_;
    QpShape fwd_shape_;
    ibv_gid mygid_{};              // local GID, cached in connect() for exchange_keys()
    std::uint32_t active_mtu_ = 0; // local active_mtu enum, cached in connect()
    std::shared_ptr<common::Context> ctx_;
    std::shared_ptr<common::ProtectionDomain> pd_;
    std::shared_ptr<common::CompletionQueue> fwd_cq_, bwd_cq_;
    std::shared_ptr<common::QueuePair> fwd_qp_, bwd_qp_;
    common::FdGuard oob_fd_;
    std::vector<common::MemoryRegion> caller_memory_regions_;
    std::optional<common::MemoryRegion> send_buf_; // local send source (one frame)
    frame::RingGeometry ring_ = frame::BL16_RING;  // both rings, local and the peer's
    MemRegion local_{};
    PeerRef peer_{};
    ChannelDesc desc_{};
};

extern template class VerbsSessionBase<ControllerSession>;
extern template class VerbsSessionBase<CoprocessorSession>;

} // namespace catalyst::transport::common
