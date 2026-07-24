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

#include "CpuSessionBase.hpp"

namespace catalyst::transport::cpu_verbs {

// Controller role: caller-driven. The caller commits a work item
// (I/O sizes), writes the outbound payload into data_slot(), kick()s one round,
// then collect()s the reply. No internal engine thread.
class CpuControllerSession : public ControllerSession {
  public:
    explicit CpuControllerSession(std::string dev = "rxe0", int gid_idx = 1)
        : base_(std::move(dev), gid_idx)
    {
    }

    int connect(const ConnectInfo &info) override { return base_.connect(info); }
    MemRegion alloc_memory(std::size_t size, MemKind kind) override
    {
        return base_.alloc_memory(size, kind);
    }
    PeerRef exchange_keys(const MemRegion &local) override { return base_.exchange_keys(local); }
    void establish_channel(const ChannelDesc &desc, const MemRegion &local,
                           const PeerRef &peer) override
    {
        base_.establish_channel(desc, local, peer);
    }
    void start() override { base_.start(); }
    int collect(void *const *replies, const std::uint64_t *replies_bytes, std::size_t n) override
    {
        return base_.collect(replies, replies_bytes, n);
    }
    void stop() override { base_.stop(); }
    std::uint64_t last_rtt_ns() const override { return base_.last_rtt_ns(); }

    // ControllerSession interface.
    void commit_work_item(std::uint32_t work_item_idx, std::uint64_t in_bytes,
                          std::uint64_t out_bytes) override
    {
        base_.commit_work_item(work_item_idx, in_bytes, out_bytes);
    }
    int kick(std::uint32_t work_item_idx = 0) override { return base_.kick(work_item_idx); }
    void *data_slot() override { return base_.data_slot(); }

  private:
    // Caller-driven controller over the shared session primitives. run() is unused
    // (no engine thread); start()/stop()/collect() are overridden for the
    // synchronous kick model.
    class Impl : public CpuSessionBase {
      public:
        using CpuSessionBase::CpuSessionBase;
        ~Impl() { stop(); }

        void start() override;
        void stop() override;
        int collect(void *const *replies, const std::uint64_t *replies_bytes,
                    std::size_t n) override;
        std::uint64_t last_rtt_ns() const override { return rtt_ns_; }

        void commit_work_item(std::uint32_t work_item_idx, std::uint64_t in_bytes,
                              std::uint64_t out_bytes);
        int kick(std::uint32_t work_item_idx);
        void *data_slot();

      protected:
        void run(std::stop_token) override {} // unused: controller is caller-driven
        bool oob_listens() const override { return false; }

      private:
        std::uint64_t in_bytes_ = sizeof(common::Payload::value);
        std::uint64_t out_bytes_ = sizeof(common::Payload::value);
        std::uint64_t next_send_ = 0, next_recv_ = 0;
        int signaled_outstanding_ = 0;
        std::uint64_t kick_ns_ = 0, rtt_ns_ = 0;
    };
    Impl base_;
};

} // namespace catalyst::transport::cpu_verbs
