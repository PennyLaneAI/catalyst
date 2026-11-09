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

/**
 * @file
 * The CPU coprocessor on the HSB wire (HsbWire.hpp), selected by `wire=hsb` in the cpu_verbs
 * coprocessor's config. The request ring is host memory at IOVA 0, laid out as
 * hsb::ring_from_config says; a pinned thread spins on the seq of the slot HSB writes next, runs
 * the coprocessor function, and sends the reply.
 */

#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>

#include "Frame.hpp"
#include "HsbLink.hpp"
#include "MemoryRegion.hpp"
#include "Transport.hpp"

namespace catalyst::transport::cpu_verbs {

class CpuHsbCoprocessorSession final : public CoprocessorSession {
  public:
    CpuHsbCoprocessorSession(std::string dev, int gid_idx, common::frame::RingGeometry ring,
                             bool imm)
        : link_(std::move(dev), gid_idx, imm), ring_(ring) {}
    ~CpuHsbCoprocessorSession() override { stop(); }

    CpuHsbCoprocessorSession(const CpuHsbCoprocessorSession &) = delete;
    CpuHsbCoprocessorSession &operator=(const CpuHsbCoprocessorSession &) = delete;

    int connect(const ConnectInfo &info) override;
    MemRegion alloc_memory(std::size_t size, MemKind kind) override;
    PeerRef exchange_keys(const MemRegion &local) override;
    void establish_channel(const ChannelDesc &desc, const MemRegion &local,
                           const PeerRef &peer) override;
    void start() override;
    int collect(void *const *replies, const std::uint64_t *replies_bytes, std::size_t n) override;
    void stop() override;

    void set_coprocessor_fn(CoprocessorFn fn, void *ctx) override;

    void set_thread_affinity(int cpu, bool realtime) {
        pin_cpu_ = cpu;
        pin_realtime_ = realtime;
    }

  private:
    void run(std::stop_token st);

    common::HsbLink link_;
    common::frame::RingGeometry ring_;
    std::optional<common::MemoryRegion> ring_mr_; // the ring, at IOVA 0
    std::uint8_t *ring_buf_ = nullptr;

    int pin_cpu_ = -1; // -1 -> leave affinity alone
    bool pin_realtime_ = false;
    CoprocessorFn coproc_fn_ = nullptr; // nullptr -> built-in echo
    void *coproc_ctx_ = nullptr;

    // failed_ (release) publishes error_; collect() acquire-loads it and rethrows.
    std::atomic<bool> failed_{false};
    std::exception_ptr error_;
    std::atomic<std::uint64_t> completed_{0};
    std::atomic<std::uint64_t> last_word_{0};
    std::jthread engine_;
};

} // namespace catalyst::transport::cpu_verbs
