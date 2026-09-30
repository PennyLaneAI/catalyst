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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <thread>

#include "CpuControllerSession.hpp"
#include "CpuCoprocessorSession.hpp"
#include "WireProtocol.hpp"

#include <catch2/catch_test_macros.hpp>
#include <infiniband/verbs.h>

using namespace catalyst::transport;
using namespace catalyst::transport::cpu_verbs;
using namespace catalyst::transport::common; // DEMO_SYNDROME, REGION_BYTES

static bool have_rxe() {
    int n = 0;
    ibv_device **devs = ibv_get_device_list(&n);
    bool found = false;
    for (int i = 0; i < n; i++) {
        if (std::string(ibv_get_device_name(devs[i])) == "rxe0") {
            found = true;
        }
    }
    if (devs) {
        ibv_free_device_list(devs);
    }
    return found;
}

// A custom coprocessor function (bitwise-invert) to exercise the set_coprocessor_fn
// path with a non-null, non-echo function.
static int invert_fn(const void *in, std::size_t in_len, void *out, std::size_t out_cap,
                     void * /*ctx*/) {
    std::uint64_t v = 0;
    std::memcpy(&v, in, std::min(in_len, sizeof(v)));
    v = ~v;
    const std::size_t n = std::min(out_cap, sizeof(v));
    std::memcpy(out, &v, n);
    return 0;
}

static int failing_fn(const void *, std::size_t, void *, std::size_t, void * /*ctx*/) { return 1; }

TEST_CASE("controller and coprocessor connect: both reach INIT and open the "
          "OOB channel",
          "[cpu_libibverbs]") {
    if (!have_rxe()) {
        SKIP("no rxe0 RDMA device");
    }
    const std::uint16_t port = 18590;
    int coproc_rc = -99;
    std::thread t([&] {
        CpuCoprocessorSession coproc("rxe0", 1);
        ConnectInfo ci{
            .peer = "127.0.0.1",
            .oob_port = port,
        };
        coproc_rc = coproc.connect(ci);
    });
    CpuControllerSession controller("rxe0", 1);
    ConnectInfo ci{
        .peer = "127.0.0.1",
        .oob_port = port,
    };
    int controller_rc = controller.connect(ci);
    t.join();
    REQUIRE(controller_rc == 0);
    REQUIRE(coproc_rc == 0);
}

TEST_CASE("alloc_memory registers host RAM and exchange_keys swaps regions", "[cpu_libibverbs]") {
    if (!have_rxe()) {
        SKIP("no rxe0 RDMA device");
    }
    const std::uint16_t port = 18591;
    const std::size_t SIZE = REGION_BYTES;
    std::uint32_t coproc_rkey = 0;
    std::uint64_t coproc_peer_addr = 0;
    std::uint64_t coproc_peer_size = 0;
    std::thread t([&] {
        CpuCoprocessorSession coproc("rxe0", 1);
        ConnectInfo ci{
            .peer = "127.0.0.1",
            .oob_port = port,
        };
        coproc.connect(ci);
        MemRegion m = coproc.alloc_memory(SIZE, MemKind::CpuRam);
        coproc_rkey = m.rkey;
        PeerRef p = coproc.exchange_keys(m);
        coproc_peer_addr = p.remote_addr;
        coproc_peer_size = p.size;
    });
    CpuControllerSession controller("rxe0", 1);
    ConnectInfo ci{
        .peer = "127.0.0.1",
        .oob_port = port,
    };
    controller.connect(ci);
    MemRegion mine = controller.alloc_memory(SIZE, MemKind::CpuRam);
    PeerRef peer = controller.exchange_keys(mine);
    t.join();

    REQUIRE(mine.addr != nullptr);
    REQUIRE(mine.lkey != 0);
    REQUIRE(mine.rkey != 0);
    REQUIRE(peer.size == SIZE);
    REQUIRE(peer.rkey == coproc_rkey);
    REQUIRE(coproc_peer_addr == reinterpret_cast<std::uint64_t>(mine.addr));
    REQUIRE(coproc_peer_size == SIZE);
}

TEST_CASE("round-trip: coprocessor gets request, controller gets bounced reply",
          "[cpu_libibverbs]") {
    if (!have_rxe()) {
        SKIP("no rxe0 RDMA device");
    }
    const std::uint16_t port = 18593;
    const std::size_t SIZE = REGION_BYTES;
    std::uint64_t coproc_got = 0;
    std::thread t([&] {
        CpuCoprocessorSession coproc("rxe0", 1);
        ConnectInfo ci{
            .peer = "127.0.0.1",
            .oob_port = port,
        };
        coproc.connect(ci);
        MemRegion m = coproc.alloc_memory(SIZE, MemKind::CpuRam);
        PeerRef p = coproc.exchange_keys(m);
        ChannelDesc desc{
            .transport = "rdma",
        };
        coproc.establish_channel(desc, m, p);
        coproc.set_coprocessor_fn(nullptr, nullptr); // built-in echo
        coproc.start();
        void *outs[1] = {&coproc_got};
        std::uint64_t obytes[1] = {sizeof(coproc_got)};
        coproc.collect(outs, obytes, 1);
        coproc.stop();
    });
    CpuControllerSession controller("rxe0", 1);
    ConnectInfo ci{
        .peer = "127.0.0.1",
        .oob_port = port,
    };
    controller.connect(ci);
    MemRegion m = controller.alloc_memory(SIZE, MemKind::CpuRam);
    PeerRef p = controller.exchange_keys(m);
    ChannelDesc desc{
        .transport = "rdma",
    };
    controller.establish_channel(desc, m, p);
    controller.commit_work_item(0, sizeof(std::uint64_t), sizeof(std::uint64_t));
    controller.start();
    const std::uint64_t syndrome = DEMO_SYNDROME;
    std::memcpy(controller.data_slot(), &syndrome, sizeof(syndrome));
    controller.kick(0);
    std::uint64_t controller_got = 0;
    void *outs[1] = {&controller_got};
    std::uint64_t obytes[1] = {sizeof(controller_got)};
    controller.collect(outs, obytes, 1);
    controller.stop();
    t.join();

    REQUIRE(coproc_got == DEMO_SYNDROME);
    REQUIRE(controller_got == DEMO_SYNDROME); // echoed back unchanged
}

TEST_CASE("round-trip with a custom coprocessor function runs on the coprocessor",
          "[cpu_libibverbs]") {
    if (!have_rxe()) {
        SKIP("no rxe0 RDMA device");
    }
    const std::uint16_t port = 18595;
    const std::size_t SIZE = REGION_BYTES;
    std::thread t([&] {
        CpuCoprocessorSession coproc("rxe0", 1);
        ConnectInfo ci{
            .peer = "127.0.0.1",
            .oob_port = port,
        };
        coproc.connect(ci);
        MemRegion m = coproc.alloc_memory(SIZE, MemKind::CpuRam);
        PeerRef p = coproc.exchange_keys(m);
        ChannelDesc desc{
            .transport = "rdma",
        };
        coproc.establish_channel(desc, m, p);
        coproc.set_coprocessor_fn(invert_fn, nullptr);
        coproc.start();
        std::uint64_t got = 0;
        void *outs[1] = {&got};
        std::uint64_t obytes[1] = {sizeof(got)};
        coproc.collect(outs, obytes, 1);
        coproc.stop();
    });
    CpuControllerSession controller("rxe0", 1);
    ConnectInfo ci{
        .peer = "127.0.0.1",
        .oob_port = port,
    };
    controller.connect(ci);
    MemRegion m = controller.alloc_memory(SIZE, MemKind::CpuRam);
    PeerRef p = controller.exchange_keys(m);
    ChannelDesc desc{
        .transport = "rdma",
    };
    controller.establish_channel(desc, m, p);
    controller.commit_work_item(0, sizeof(std::uint64_t), sizeof(std::uint64_t));
    controller.start();
    const std::uint64_t syndrome = DEMO_SYNDROME;
    std::memcpy(controller.data_slot(), &syndrome, sizeof(syndrome));
    controller.kick(0);
    std::uint64_t got = 0;
    void *outs[1] = {&got};
    std::uint64_t obytes[1] = {sizeof(got)};
    controller.collect(outs, obytes, 1);
    controller.stop();
    t.join();

    REQUIRE(got == ~DEMO_SYNDROME); // controller received the coprocessor's result
    REQUIRE(got != DEMO_SYNDROME);  // and it is not a mere echo
}

TEST_CASE("cpu_verbs treats the coprocessor return value as a status code", "[cpu_libibverbs]") {
    if (!have_rxe()) {
        SKIP("no rxe0 RDMA device");
    }
    const std::uint16_t port = 18596;
    const std::size_t SIZE = REGION_BYTES;
    bool coproc_failed = false;
    std::string coproc_error;

    std::thread t([&] {
        try {
            CpuCoprocessorSession coproc("rxe0", 1);
            ConnectInfo ci{
                .peer = "127.0.0.1",
                .oob_port = port,
            };
            coproc.connect(ci);
            MemRegion m = coproc.alloc_memory(SIZE, MemKind::CpuRam);
            PeerRef p = coproc.exchange_keys(m);
            ChannelDesc desc{
                .transport = "rdma",
            };
            coproc.establish_channel(desc, m, p);
            coproc.set_coprocessor_fn(failing_fn, nullptr);
            coproc.start();
            coproc.collect(nullptr, nullptr, 0);
        } catch (const std::exception &e) {
            coproc_failed = true;
            coproc_error = e.what();
        }
    });

    CpuControllerSession controller("rxe0", 1);
    ConnectInfo ci{
        .peer = "127.0.0.1",
        .oob_port = port,
    };
    controller.connect(ci);
    MemRegion m = controller.alloc_memory(SIZE, MemKind::CpuRam);
    PeerRef p = controller.exchange_keys(m);
    ChannelDesc desc{
        .transport = "rdma",
    };
    controller.establish_channel(desc, m, p);
    controller.commit_work_item(0, sizeof(std::uint64_t), sizeof(std::uint64_t));
    controller.start();
    const std::uint64_t syndrome = DEMO_SYNDROME;
    controller.write_data_slot(&syndrome, sizeof(syndrome), 0);
    REQUIRE(controller.kick(0) == 0);
    controller.stop();
    t.join();

    REQUIRE(coproc_failed);
    CHECK(coproc_error.find("status 1") != std::string::npos);
}
