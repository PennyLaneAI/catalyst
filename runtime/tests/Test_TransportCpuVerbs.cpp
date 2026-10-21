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
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "CpuControllerSession.hpp"
#include "CpuCoprocessorSession.hpp"
#include "Frame.hpp"
#include "WireProtocol.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
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
static std::size_t invert_fn(const void *in, std::size_t in_len, std::uint32_t /*fn_id*/, void *out,
                             std::size_t out_cap, void * /*ctx*/) {
    std::uint64_t v = 0;
    std::memcpy(&v, in, std::min(in_len, sizeof(v)));
    v = ~v;
    const std::size_t n = std::min(out_cap, sizeof(v));
    std::memcpy(out, &v, n);
    return n;
}

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

// Replies with each payload byte plus the fn_id, for as many bytes as the reply holds, and
// records the payload length it was handed.
static std::size_t g_blv_len = 0;
static std::size_t add_fn_id(const void *in, std::size_t in_len, std::uint32_t fn_id, void *out,
                             std::size_t out_cap, void * /*ctx*/) {
    g_blv_len = in_len;
    const std::size_t n = std::min(in_len, out_cap);
    for (std::size_t i = 0; i < n; ++i) {
        static_cast<std::uint8_t *>(out)[i] =
            static_cast<std::uint8_t>(static_cast<const std::uint8_t *>(in)[i] + fn_id);
    }
    return n;
}

TEST_CASE("cpu_verbs carries messages larger than 8 B in BLV frames", "[cpu_libibverbs]") {
    if (!have_rxe()) {
        SKIP("no rxe0 RDMA device");
    }
    namespace frame = catalyst::transport::common::frame;
    // 64 B fits a 128 B slot and is sent inline; 1000 B needs a 2 KiB slot, sent from the
    // registered buffer because it exceeds the queue pair's inline capacity.
    for (const std::size_t in_bytes : {std::size_t{64}, std::size_t{1000}}) {
        const std::size_t out_bytes = in_bytes / 2;
        const auto ring = frame::geometry_for(frame::Mode::BLV, in_bytes);
        const auto port = static_cast<std::uint16_t>(in_bytes == 64 ? 18598 : 18599);
        std::thread t([&] {
            CpuCoprocessorSession coproc("rxe0", 1, ring);
            coproc.connect(ConnectInfo{.peer = "127.0.0.1", .oob_port = port});
            MemRegion m = coproc.alloc_memory(ring.region_bytes(), MemKind::CpuRam);
            PeerRef p = coproc.exchange_keys(m);
            coproc.establish_channel(ChannelDesc{.transport = "rdma"}, m, p);
            coproc.set_coprocessor_fn(add_fn_id, nullptr);
            coproc.start();
            std::uint64_t first = 0;
            void *outs[1] = {&first};
            std::uint64_t obytes[1] = {sizeof(first)};
            coproc.collect(outs, obytes, 1);
            coproc.stop();
        });
        CpuControllerSession controller("rxe0", 1, ring);
        controller.connect(ConnectInfo{.peer = "127.0.0.1", .oob_port = port});
        MemRegion m = controller.alloc_memory(ring.region_bytes(), MemKind::CpuRam);
        PeerRef p = controller.exchange_keys(m);
        controller.establish_channel(ChannelDesc{.transport = "rdma"}, m, p);
        controller.commit_work_item(0, in_bytes, out_bytes);
        controller.start();
        std::vector<std::uint8_t> request(in_bytes);
        for (std::size_t i = 0; i < in_bytes; ++i) {
            request[i] = static_cast<std::uint8_t>(i);
        }
        controller.write_data_slot(request.data(), in_bytes, /*decoder_id=*/3);
        controller.kick(0);
        std::vector<std::uint8_t> reply(out_bytes);
        void *outs[1] = {reply.data()};
        std::uint64_t obytes[1] = {out_bytes};
        REQUIRE(controller.collect(outs, obytes, 1) == 0);
        controller.stop();
        t.join();

        CHECK(g_blv_len == in_bytes); // the payload's own length, not a padded frame
        for (std::size_t i = 0; i < out_bytes; ++i) {
            REQUIRE(reply[i] == static_cast<std::uint8_t>(i + 3));
        }
    }
}

// Runs `rounds` CUQR round trips of `bytes` B over rxe0 with the given verb, checking each reply.
static void cuqr_echo(std::size_t bytes, bool write_imm, std::uint16_t port, std::uint32_t rounds,
                      bool cudaq_peer = false, std::uint32_t slots = 256,
                      bool replies_by_send = false) {
    namespace frame = catalyst::transport::common::frame;
    auto ring = frame::geometry_for(frame::Mode::CUQR, bytes);
    ring.slots = slots;
    const common::CuqrConfig cuqr{.write_imm = write_imm, .cudaq_peer = cudaq_peer};
    std::thread t([&] {
        CpuCoprocessorSession coproc(
            "rxe0", 1, ring,
            {.write_imm = write_imm && !replies_by_send, .cudaq_peer = cudaq_peer});
        coproc.connect(ConnectInfo{.peer = "127.0.0.1", .oob_port = port});
        MemRegion m = coproc.alloc_memory(ring.region_bytes(), MemKind::CpuRam);
        PeerRef p = coproc.exchange_keys(m);
        coproc.establish_channel(ChannelDesc{.transport = "rdma"}, m, p);
        coproc.set_coprocessor_fn(add_fn_id, nullptr);
        coproc.start();
        while (coproc.completed() < rounds) {
            std::this_thread::yield();
        }
        coproc.stop();
    });
    CpuControllerSession controller("rxe0", 1, ring, cuqr);
    controller.connect(ConnectInfo{.peer = "127.0.0.1", .oob_port = port});
    MemRegion m = controller.alloc_memory(ring.region_bytes(), MemKind::CpuRam);
    PeerRef p = controller.exchange_keys(m);
    controller.establish_channel(ChannelDesc{.transport = "rdma"}, m, p);
    controller.commit_work_item(0, bytes, bytes);
    controller.start();
    std::vector<std::uint8_t> request(bytes), reply(bytes);
    for (std::uint32_t r = 0; r < rounds; ++r) {
        for (std::size_t i = 0; i < bytes; ++i) {
            request[i] = static_cast<std::uint8_t>(i * 7 + r);
        }
        controller.write_data_slot(request.data(), bytes, /*decoder_id=*/2);
        controller.kick(0);
        void *outs[1] = {reply.data()};
        std::uint64_t obytes[1] = {bytes};
        REQUIRE(controller.collect(outs, obytes, 1) == 0);
        for (std::size_t i = 0; i < bytes; ++i) {
            REQUIRE(reply[i] == static_cast<std::uint8_t>(request[i] + 2));
        }
    }
    controller.stop();
    t.join();
}

TEST_CASE("cpu_verbs carries CUQR frames with SEND and with write-with-immediate",
          "[cpu_libibverbs]") {
    if (!have_rxe()) {
        SKIP("no rxe0 RDMA device");
    }
    std::uint16_t port = 18610;
    for (const bool write_imm : {false, true}) {
        for (const std::size_t bytes :
             {std::size_t{8}, std::size_t{64}, std::size_t{1024}, std::size_t{4096}}) {
            // 300 rounds pass a 256-slot ring, so receive requests must be reposted.
            cuqr_echo(bytes, write_imm, port++, 300);
        }
    }
}

TEST_CASE("cpu_verbs refuses a CUQR peer using the other completion verb", "[cpu_libibverbs]") {
    if (!have_rxe()) {
        SKIP("no rxe0 RDMA device");
    }
    namespace frame = catalyst::transport::common::frame;
    const auto ring = frame::geometry_for(frame::Mode::CUQR, 8);
    std::thread t([&] {
        CpuCoprocessorSession coproc("rxe0", 1, ring, {.write_imm = true});
        coproc.connect(ConnectInfo{.peer = "127.0.0.1", .oob_port = 18630});
        MemRegion m = coproc.alloc_memory(ring.region_bytes(), MemKind::CpuRam);
        CHECK_THROWS_WITH(coproc.exchange_keys(m), Catch::Matchers::ContainsSubstring("cuqr_tx"));
    });
    CpuControllerSession controller("rxe0", 1, ring, {.write_imm = false});
    controller.connect(ConnectInfo{.peer = "127.0.0.1", .oob_port = 18630});
    MemRegion m = controller.alloc_memory(ring.region_bytes(), MemKind::CpuRam);
    CHECK_THROWS_WITH(controller.exchange_keys(m), Catch::Matchers::ContainsSubstring("cuqr_tx"));
    t.join();
}

// Fails fn 7 with the ABI's error value, answers fn 8 with an empty result, adds fn_id otherwise.
static std::size_t fail_on_fn_7(const void *in, std::size_t in_len, std::uint32_t fn_id, void *out,
                                std::size_t out_cap, void *ctx) {
    if (fn_id == 7) {
        return COPROCESSOR_FN_ERROR;
    }
    return fn_id == 8 ? 0 : add_fn_id(in, in_len, fn_id, out, out_cap, ctx);
}

TEST_CASE("a CUQR coprocessor answers a failing function with a status and keeps serving",
          "[cpu_libibverbs]") {
    if (!have_rxe()) {
        SKIP("no rxe0 RDMA device");
    }
    namespace frame = catalyst::transport::common::frame;
    const auto ring = frame::geometry_for(frame::Mode::CUQR, 8);
    std::thread t([&] {
        CpuCoprocessorSession coproc("rxe0", 1, ring);
        coproc.connect(ConnectInfo{.peer = "127.0.0.1", .oob_port = 18640});
        MemRegion m = coproc.alloc_memory(ring.region_bytes(), MemKind::CpuRam);
        PeerRef p = coproc.exchange_keys(m);
        coproc.establish_channel(ChannelDesc{.transport = "rdma"}, m, p);
        coproc.set_coprocessor_fn(fail_on_fn_7, nullptr);
        coproc.start();
        while (coproc.completed() < 3) {
            std::this_thread::yield();
        }
        coproc.stop();
    });
    CpuControllerSession controller("rxe0", 1, ring);
    controller.connect(ConnectInfo{.peer = "127.0.0.1", .oob_port = 18640});
    MemRegion m = controller.alloc_memory(ring.region_bytes(), MemKind::CpuRam);
    PeerRef p = controller.exchange_keys(m);
    controller.establish_channel(ChannelDesc{.transport = "rdma"}, m, p);
    controller.commit_work_item(0, 8, 8);
    controller.start();
    std::uint64_t word = 5, reply = 0;
    void *outs[1] = {&reply};
    std::uint64_t obytes[1] = {8};
    controller.write_data_slot(&word, 8, /*decoder_id=*/7);
    controller.kick(0);
    CHECK_THROWS_WITH(controller.collect(outs, obytes, 1),
                      Catch::Matchers::ContainsSubstring("status 1"));
    controller.write_data_slot(&word, 8, /*decoder_id=*/8); // an empty result is not a failure
    controller.kick(0);
    reply = 0xFFFF;
    REQUIRE(controller.collect(outs, obytes, 1) == 0);
    CHECK(reply == 0);
    controller.write_data_slot(&word, 8, /*decoder_id=*/1); // the coprocessor is still serving
    controller.kick(0);
    REQUIRE(controller.collect(outs, obytes, 1) == 0);
    CHECK(reply == 0x0101010101010106ULL); // add_fn_id adds fn_id 1 to every byte of 5
    controller.stop();
    t.join();
}

TEST_CASE("cpu_verbs talks to a cudaq-realtime peer: UC, rendezvous, zero-based addressing",
          "[cpu_libibverbs]") {
    if (!have_rxe()) {
        SKIP("no rxe0 RDMA device");
    }
    // cudaq-realtime's cpu_roce default: 8 slots of 256 B; 20 rounds wrap the ring twice.
    cuqr_echo(232, /*write_imm=*/false, 18650, 20, /*cudaq_peer=*/true, /*slots=*/8);
    cuqr_echo(232, /*write_imm=*/true, 18651, 20, /*cudaq_peer=*/true, /*slots=*/8);
    // A cudaq-realtime server answers with SEND whichever verb its requests came by.
    cuqr_echo(232, /*write_imm=*/true, 18652, 20, /*cudaq_peer=*/true, /*slots=*/8,
              /*replies_by_send=*/true);
}

TEST_CASE("an idle CUQR coprocessor stops promptly", "[cpu_libibverbs]") {
    if (!have_rxe()) {
        SKIP("no rxe0 RDMA device");
    }
    namespace frame = catalyst::transport::common::frame;
    const auto ring = frame::geometry_for(frame::Mode::CUQR, 8);
    std::thread t([&] {
        CpuControllerSession controller("rxe0", 1, ring);
        controller.connect(ConnectInfo{.peer = "127.0.0.1", .oob_port = 18660});
        MemRegion m = controller.alloc_memory(ring.region_bytes(), MemKind::CpuRam);
        PeerRef p = controller.exchange_keys(m);
        controller.establish_channel(ChannelDesc{.transport = "rdma"}, m, p);
    });
    CpuCoprocessorSession coproc("rxe0", 1, ring);
    coproc.connect(ConnectInfo{.peer = "127.0.0.1", .oob_port = 18660});
    MemRegion m = coproc.alloc_memory(ring.region_bytes(), MemKind::CpuRam);
    PeerRef p = coproc.exchange_keys(m);
    coproc.establish_channel(ChannelDesc{.transport = "rdma"}, m, p);
    t.join();
    coproc.start(); // waits on its receive completion queue; nothing arrives
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const auto t0 = std::chrono::steady_clock::now();
    coproc.stop();
    CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(1));
}

// Echoes a request of 0xEE bytes in full, and only the first half of any other.
static std::size_t half_fn(const void *in, std::size_t in_len, std::uint32_t /*fn_id*/, void *out,
                           std::size_t /*out_cap*/, void * /*ctx*/) {
    const std::size_t n = *static_cast<const std::uint8_t *>(in) == 0xEE ? in_len : in_len / 2;
    std::memcpy(out, in, n);
    return n;
}

TEST_CASE("a short CUQR reply reads as zeros past its length, also after the ring wraps",
          "[cpu_libibverbs]") {
    if (!have_rxe()) {
        SKIP("no rxe0 RDMA device");
    }
    namespace frame = catalyst::transport::common::frame;
    const auto ring = frame::geometry_for(frame::Mode::CUQR, 64);
    constexpr std::uint32_t rounds = 300; // past a 256-slot lap
    std::thread t([&] {
        CpuCoprocessorSession coproc("rxe0", 1, ring);
        coproc.connect(ConnectInfo{.peer = "127.0.0.1", .oob_port = 18670});
        MemRegion m = coproc.alloc_memory(ring.region_bytes(), MemKind::CpuRam);
        PeerRef p = coproc.exchange_keys(m);
        coproc.establish_channel(ChannelDesc{.transport = "rdma"}, m, p);
        coproc.set_coprocessor_fn(half_fn, nullptr);
        coproc.start();
        while (coproc.completed() < rounds) {
            std::this_thread::yield();
        }
        coproc.stop();
    });
    CpuControllerSession controller("rxe0", 1, ring);
    controller.connect(ConnectInfo{.peer = "127.0.0.1", .oob_port = 18670});
    MemRegion m = controller.alloc_memory(ring.region_bytes(), MemKind::CpuRam);
    PeerRef p = controller.exchange_keys(m);
    controller.establish_channel(ChannelDesc{.transport = "rdma"}, m, p);
    controller.commit_work_item(0, 64, 64);
    controller.start();
    std::vector<std::uint8_t> request(64), reply(64);
    for (std::uint32_t r = 0; r < rounds; ++r) {
        // The first lap's replies fill the slot; the second lap's are half as long, so a tail
        // left over from the first lap would show.
        const bool first_lap = r < ring.slots;
        std::memset(request.data(), first_lap ? 0xEE : 0x11, request.size());
        controller.write_data_slot(request.data(), 64, 0);
        controller.kick(0);
        void *outs[1] = {reply.data()};
        std::uint64_t obytes[1] = {64};
        REQUIRE(controller.collect(outs, obytes, 1) == 0);
        for (std::size_t i = 32; i < 64; ++i) {
            REQUIRE(reply[i] == (first_lap ? 0xEE : 0));
        }
    }
    controller.stop();
    t.join();
}

TEST_CASE("a controller talking to a cudaq-realtime peer reports a lost reply",
          "[cpu_libibverbs]") {
    if (!have_rxe()) {
        SKIP("no rxe0 RDMA device");
    }
    namespace frame = catalyst::transport::common::frame;
    // UC drops rather than retries, so a reply that never comes must fail, not hang.
    auto ring = frame::geometry_for(frame::Mode::CUQR, 8);
    ring.slots = 8;
    std::thread t([&] {
        CpuCoprocessorSession coproc("rxe0", 1, ring, {.cudaq_peer = true});
        coproc.connect(ConnectInfo{.peer = "127.0.0.1", .oob_port = 18680});
        MemRegion m = coproc.alloc_memory(ring.region_bytes(), MemKind::CpuRam);
        coproc.exchange_keys(m); // and never started, so nothing answers
        std::this_thread::sleep_for(std::chrono::seconds(8));
    });
    CpuControllerSession controller("rxe0", 1, ring, {.cudaq_peer = true});
    controller.connect(ConnectInfo{.peer = "127.0.0.1", .oob_port = 18680});
    MemRegion m = controller.alloc_memory(ring.region_bytes(), MemKind::CpuRam);
    PeerRef p = controller.exchange_keys(m);
    controller.establish_channel(ChannelDesc{.transport = "rdma"}, m, p);
    controller.commit_work_item(0, 8, 8);
    controller.start();
    std::uint64_t word = 5, reply = 0;
    void *outs[1] = {&reply};
    std::uint64_t obytes[1] = {8};
    controller.write_data_slot(&word, 8, 0);
    controller.kick(0);
    CHECK_THROWS_WITH(controller.collect(outs, obytes, 1),
                      Catch::Matchers::ContainsSubstring("no reply"));
    controller.stop();
    t.join();
}

TEST_CASE("cpu_verbs refuses a CUQR peer with a different slot count", "[cpu_libibverbs]") {
    if (!have_rxe()) {
        SKIP("no rxe0 RDMA device");
    }
    namespace frame = catalyst::transport::common::frame;
    auto small = frame::geometry_for(frame::Mode::CUQR, 8);
    small.slots = 8;
    const auto large = frame::geometry_for(frame::Mode::CUQR, 8);
    std::thread t([&] {
        CpuCoprocessorSession coproc("rxe0", 1, large);
        coproc.connect(ConnectInfo{.peer = "127.0.0.1", .oob_port = 18690});
        MemRegion m = coproc.alloc_memory(large.region_bytes(), MemKind::CpuRam);
        CHECK_THROWS_WITH(coproc.exchange_keys(m), Catch::Matchers::ContainsSubstring("slots"));
    });
    CpuControllerSession controller("rxe0", 1, small);
    controller.connect(ConnectInfo{.peer = "127.0.0.1", .oob_port = 18690});
    MemRegion m = controller.alloc_memory(small.region_bytes(), MemKind::CpuRam);
    CHECK_THROWS_WITH(controller.exchange_keys(m), Catch::Matchers::ContainsSubstring("slots"));
    t.join();
}
