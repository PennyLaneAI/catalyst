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
#include <cstddef>
#include <memory>

#include "CpuSessionBase.hpp"

namespace catalyst::transport::cpu_verbs {

// Coprocessor role: receives messages, runs the coprocessor function, and
// returns the result. The function is bound via set_coprocessor_fn; nullptr
// selects the built-in echo (passthrough self-test).
class CpuCoprocessorSession : public CoprocessorSession {
  public:
    explicit CpuCoprocessorSession(std::string dev = "rxe0", int gid_idx = 1)
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

    void set_coprocessor_fn(CoprocessorFn fn, void *ctx) override;

  private:
    class Impl : public CpuSessionBase {
      public:
        using CpuSessionBase::CpuSessionBase;
        ~Impl() { stop(); }
        CoprocessorFn coproc_fn_ = nullptr; // nullptr -> built-in echo
        void *coproc_ctx_ = nullptr;

      protected:
        void run(std::stop_token st) override;
        bool oob_listens() const override { return true; }
    };
    Impl base_;
};

} // namespace catalyst::transport::cpu_verbs
