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
#include <thread>

#include "CpuSessionBase.hpp"

namespace catalyst::transport::cpu_verbs {

// Coprocessor role: receives messages, runs the coprocessor function, and returns
// the result. The function is bound via set_coprocessor_fn; nullptr selects the
// built-in echo. Each message's decoder_id is handed to the function, which may
// serve several codes and dispatch on it, or ignore it if it serves only one.
class CpuCoprocessorSession : public CpuSessionBase<CoprocessorSession> {
    using Base = CpuSessionBase<CoprocessorSession>;

  public:
    explicit CpuCoprocessorSession(std::string dev, int gid_idx) : Base(std::move(dev), gid_idx) {}
    ~CpuCoprocessorSession() override { stop(); }

    void start() override;
    int collect(void *const *replies, const std::uint64_t *replies_bytes, std::size_t n) override;
    void stop() override;

    void set_coprocessor_fn(CoprocessorFn fn, void *ctx) override;

    void set_thread_affinity(int cpu, bool realtime) {
        pin_cpu_ = cpu;
        pin_realtime_ = realtime;
    }

  protected:
    bool oob_listens() const override { return true; }

  private:
    // The engine loop; runs on engine_.
    void run(std::stop_token st);

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
