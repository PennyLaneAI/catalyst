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
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

#include "Frame.hpp"
#include "MemoryRegion.hpp"
#include "Transport.hpp"
#include "WireLink.hpp"

namespace catalyst::transport::cpu_verbs {

// Coprocessor role: receives messages, runs the coprocessor function, and returns
// the result. The function is bound via set_coprocessor_fn; nullptr selects the
// built-in echo. Each message's decoder_id is handed to the function, which may
// serve several codes and dispatch on it, or ignore it if it serves only one.
// The link speaks the wire to the controller (WireLink.hpp).
class CpuCoprocessorSession final : public CoprocessorSession {
  public:
    /// On the RC wire, with `ring` the frames both ends agree on.
    explicit CpuCoprocessorSession(std::string dev, int gid_idx,
                                   common::frame::RingGeometry ring = common::frame::BL16_RING);
    /// On the wire `link` speaks (common::make_coprocessor_link).
    explicit CpuCoprocessorSession(std::unique_ptr<common::WireLink> link);
    ~CpuCoprocessorSession() override { stop(); }

    CpuCoprocessorSession(const CpuCoprocessorSession &) = delete;
    CpuCoprocessorSession &operator=(const CpuCoprocessorSession &) = delete;

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
    // The engine loop; runs on engine_.
    void run(std::stop_token st);
    template <common::frame::Mode M, class Link> void serve(Link &link, std::stop_token st);

    std::unique_ptr<common::WireLink> link_;
    common::frame::RingGeometry ring_;
    std::vector<common::MemoryRegion> regions_; // handed out by alloc_memory()
    std::uint8_t *ring_buf_ = nullptr; // the ring requests land in, from establish_channel()

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
