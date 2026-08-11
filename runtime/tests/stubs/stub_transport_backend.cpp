// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// A no-op transport backend for Test_Transport: implements the controller/coprocessor factory ABI
// with sessions whose methods do nothing, so the transport CAPI (create/get_session/registry,
// argument plumbing) can be unit-tested without a NIC.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "Transport.hpp"
#include "TransportBackend.h"

using namespace catalyst::transport;

namespace {

struct StubController : ControllerSession {
    std::uint64_t slot = 0;
    std::uint64_t reply = 0;
    int connect(const ConnectInfo &) override { return 0; }
    MemRegion alloc_memory(std::size_t, MemKind) override { return {}; }
    PeerRef exchange_keys(const MemRegion &) override { return {}; }
    void establish_channel(const ChannelDesc &, const MemRegion &, const PeerRef &) override {}
    void start() override {}
    int collect(void *const *, const std::uint64_t *, std::size_t) override { return 0; }
    void stop() override {}
    void commit_work_item(std::uint32_t, std::uint64_t, std::uint64_t) override {}
    int kick(std::uint32_t) override { return 0; }
    void *data_slot() override { return &slot; }
    void write_data_slot(const void *, std::uint64_t, std::uint32_t) override {}
    void *reply_slot() override { return &reply; }
};

struct StubCoprocessor : CoprocessorSession {
    int connect(const ConnectInfo &) override { return 0; }
    MemRegion alloc_memory(std::size_t, MemKind) override { return {}; }
    PeerRef exchange_keys(const MemRegion &) override { return {}; }
    void establish_channel(const ChannelDesc &, const MemRegion &, const PeerRef &) override {}
    void start() override {}
    int collect(void *const *, const std::uint64_t *, std::size_t) override { return 0; }
    void stop() override {}
    void set_coprocessor_fn(CoprocessorFn, void *) override {}
};

// Stands in for a backend that runs its own message loop (as the GPU one does), so the
// launch-once side of bind-by-symbol is reachable from a test.
struct StubLaunchOnceCoprocessor : StubCoprocessor {
    void set_coprocessor_fn(CoprocessorFn, void *) override {
        throw std::logic_error("stub: per-message binding is not supported by this backend");
    }
    void set_coprocessor_launcher(CoprocessorLauncherFn, void *) override {}
    CoprocConvention coprocessor_fn_convention() const override {
        return CoprocConvention::LaunchOnce;
    }
};

} // namespace

extern "C" ControllerSession *CatalystTransportControllerFactory(const char *) {
    return new StubController();
}

extern "C" CoprocessorSession *CatalystTransportCoprocessorFactory(const char *config) {
    const std::string cfg = config ? config : "";
    if (cfg == "launch_once") {
        return new StubLaunchOnceCoprocessor();
    }
    return new StubCoprocessor();
}
