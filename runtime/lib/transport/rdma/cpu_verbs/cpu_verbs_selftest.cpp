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
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "CpuControllerSession.hpp"
#include "CpuCoprocessorSession.hpp"
#include "Frame.hpp"
#include "WireProtocol.hpp"

using namespace catalyst::transport;
using namespace catalyst::transport::cpu_verbs;
using namespace catalyst::transport::common; // REGION_BYTES, DEMO_SYNDROME, Payload

namespace {
// SIGINT/SIGTERM end a coprocessor's serving early, so a test harness can stop it cleanly.
std::atomic<bool> interrupted{false};
extern "C" void on_interrupt(int) { interrupted.store(true); }

// cudaq-realtime's rpc_increment handler: each payload byte plus one.
std::size_t increment(const void *in, std::size_t len, std::uint32_t /*fn_id*/, void *out,
                      std::size_t cap, void * /*ctx*/) {
    const std::size_t n = std::min(len, cap);
    for (std::size_t i = 0; i < n; ++i) {
        static_cast<std::uint8_t *>(out)[i] =
            static_cast<std::uint8_t>(static_cast<const std::uint8_t *>(in)[i] + 1);
    }
    return n;
}
} // namespace

int main(int argc, char **argv) {
    std::string role = "coprocessor", dev = "rxe0", peer = "127.0.0.1";
    int gid = 1;
    std::uint16_t port = 18560;
    // Message sizes (bytes). The controller role feeds these to commit_work_item;
    // both roles size the collect() buffer by --correction-bytes.
    std::uint32_t syndrome_bytes = sizeof(std::uint64_t);
    std::uint32_t correction_bytes = sizeof(std::uint64_t);
    // Rounds the controller runs; with more than one it reports RTT percentiles.
    std::uint32_t rounds = 1;
    // --frame cuqr selects cudaq-realtime's frames, completed with --tx send or write_imm;
    // --peer-mode cudaq talks to a cudaq-realtime peer, whose slot count --frame-slots matches.
    bool cuqr = false, write_imm = false, cudaq_peer = false;
    std::uint32_t frame_slots = 0, frame_slot_bytes = 0;
    // --increment 1: the coprocessor serves (and the controller expects) cudaq-realtime's
    // rpc_increment instead of an echo; --fn-id is the function id requests carry;
    // --serve-seconds is how long a coprocessor serves.
    bool incr = false;
    std::uint32_t fn_id = 0, serve_seconds = 3;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--role") {
            role = v;
        } else if (k == "--dev") {
            dev = v;
        } else if (k == "--gid") {
            gid = std::atoi(v.c_str());
        } else if (k == "--peer") {
            peer = v;
        } else if (k == "--port") {
            port = static_cast<std::uint16_t>(std::atoi(v.c_str()));
        } else if (k == "--syndrome-bytes") {
            syndrome_bytes = static_cast<std::uint32_t>(std::strtoul(v.c_str(), nullptr, 0));
        } else if (k == "--correction-bytes") {
            correction_bytes = static_cast<std::uint32_t>(std::strtoul(v.c_str(), nullptr, 0));
        } else if (k == "--rounds") {
            rounds = static_cast<std::uint32_t>(std::strtoul(v.c_str(), nullptr, 0));
        } else if (k == "--frame") {
            cuqr = v == "cuqr";
        } else if (k == "--tx") {
            write_imm = v == "write_imm";
        } else if (k == "--peer-mode") {
            cudaq_peer = v == "cudaq";
        } else if (k == "--frame-slots") {
            frame_slots = static_cast<std::uint32_t>(std::strtoul(v.c_str(), nullptr, 0));
        } else if (k == "--frame-slot-bytes") {
            frame_slot_bytes = static_cast<std::uint32_t>(std::strtoul(v.c_str(), nullptr, 0));
        } else if (k == "--increment") {
            incr = v == "1";
        } else if (k == "--fn-id") {
            fn_id = static_cast<std::uint32_t>(std::strtoul(v.c_str(), nullptr, 0));
        } else if (k == "--serve-seconds") {
            serve_seconds = static_cast<std::uint32_t>(std::strtoul(v.c_str(), nullptr, 0));
        }
    }
    const bool is_coprocessor = (role == "coprocessor");
    // Both roles derive the same ring from the message sizes, as the compiler's frame keys make
    // them.
    const std::size_t largest = std::max(syndrome_bytes, correction_bytes);
    auto ring = frame::geometry_for(cuqr                            ? frame::Mode::CUQR
                                    : largest <= PAYLOAD_DATA_BYTES ? frame::Mode::BL16
                                                                    : frame::Mode::BLV,
                                    largest);
    if (frame_slots != 0) {
        ring.slots = frame_slots;
    }
    if (frame_slot_bytes != 0) { // a cudaq-realtime server's slot size
        ring.stride = frame_slot_bytes;
    }
    const common::CuqrConfig cuqr_cfg{.write_imm = write_imm, .cudaq_peer = cudaq_peer};

    std::unique_ptr<TransportSession> s;
    CpuCoprocessorSession *coproc = nullptr;
    CpuControllerSession *controller = nullptr;
    if (is_coprocessor) {
        auto up = std::make_unique<CpuCoprocessorSession>(dev, gid, ring, cuqr_cfg);
        coproc = up.get();
        s = std::move(up);
    } else {
        auto up = std::make_unique<CpuControllerSession>(dev, gid, ring, cuqr_cfg);
        controller = up.get();
        s = std::move(up);
    }

    ConnectInfo ci{
        .peer = peer,
        .oob_port = port,
    };
    s->connect(ci);
    MemRegion m = s->alloc_memory(ring.region_bytes(), MemKind::CpuRam);
    PeerRef p = s->exchange_keys(m);
    ChannelDesc desc{
        .transport = "rdma",
    };
    s->establish_channel(desc, m, p);

    // Reply buffer that collect() fills with up to --correction-bytes. Sized to at
    // least 8 B so the echo check below can always read a full 64-bit word: `got`
    // copies the leading 8 bytes of the reply and compares them to DEMO_SYNDROME.
    std::vector<std::uint8_t> corr(std::max<std::size_t>(correction_bytes, sizeof(std::uint64_t)),
                                   0);
    void *outs[1] = {corr.data()};
    std::uint64_t obytes[1] = {correction_bytes};
    if (coproc) {
        coproc->set_coprocessor_fn(incr ? increment : nullptr, nullptr); // else built-in echo
        coproc->start();
        std::signal(SIGINT, on_interrupt);
        std::signal(SIGTERM, on_interrupt);
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(serve_seconds);
        while (!interrupted.load() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::fprintf(stderr, "[coprocessor] served %llu messages\n",
                     static_cast<unsigned long long>(coproc->completed()));
        // A coprocessor's collect() reports the first 8 B of the last request it served.
        std::uint64_t first_bytes[1] = {sizeof(std::uint64_t)};
        coproc->collect(outs, first_bytes, 1);
        coproc->stop();
    } else {
        // Controller: commit a work item sized by --syndrome-bytes/--correction-bytes,
        // write the syndrome into the outbound slot, kick one round, collect the correction.
        controller->commit_work_item(/*work_item_idx=*/0, syndrome_bytes, correction_bytes);
        controller->start();
        std::vector<std::uint8_t> syndrome(std::max<std::size_t>(syndrome_bytes, 8), 0);
        std::memcpy(syndrome.data(), &DEMO_SYNDROME, sizeof(DEMO_SYNDROME));
        std::vector<std::uint64_t> rtt_ns;
        // Past the leading syndrome word, each round carries its own byte pattern, and the
        // echoed reply is compared in full, so a stale or partly written frame is caught.
        const std::size_t echoed = std::min(syndrome_bytes, correction_bytes);
        std::uint32_t mismatched = 0;
        for (std::uint32_t r = 0; r < rounds; ++r) {
            for (std::size_t i = sizeof(DEMO_SYNDROME); i < syndrome_bytes; ++i) {
                syndrome[i] = static_cast<std::uint8_t>(i * 31 + r);
            }
            controller->write_data_slot(syndrome.data(), syndrome_bytes, fn_id);
            controller->kick(0);
            controller->collect(outs, obytes, 1);
            rtt_ns.push_back(controller->last_rtt_ns());
            if (incr) {
                std::vector<std::uint8_t> want(echoed);
                increment(syndrome.data(), echoed, fn_id, want.data(), echoed, nullptr);
                mismatched += std::memcmp(corr.data(), want.data(), echoed) != 0;
            } else {
                mismatched += std::memcmp(corr.data(), syndrome.data(), echoed) != 0;
            }
        }
        controller->stop();
        if (mismatched != 0) {
            std::fprintf(stderr, "[controller] %u of %u replies differ from their request\n",
                         mismatched, rounds);
            return 1;
        }
        if (rounds > 1) {
            std::sort(rtt_ns.begin(), rtt_ns.end());
            const auto pct = [&](double q) {
                return rtt_ns[static_cast<std::size_t>(q * (rtt_ns.size() - 1))];
            };
            std::fprintf(stderr,
                         "[controller] %s %u B in, %u B out, %u rounds: RTT p50=%.2f us p99=%.2f "
                         "us max=%.2f us\n",
                         ring.mode == frame::Mode::BL16 ? "BL16"
                         : ring.mode == frame::Mode::BLV
                             ? "BLV"
                             : (write_imm ? "CUQR/write_imm" : "CUQR/send"),
                         syndrome_bytes, correction_bytes, rounds, pct(0.50) / 1e3, pct(0.99) / 1e3,
                         rtt_ns.back() / 1e3);
        }
    }

    std::uint64_t got = 0;
    std::memcpy(&got, corr.data(), sizeof(got));

    // Built-in echo coprocessor: both roles observe the demo syndrome (incremented by
    // rpc_increment for the controller; the coprocessor reports the request it last served).
    std::uint64_t expect = DEMO_SYNDROME;
    if (incr && !is_coprocessor) {
        increment(&DEMO_SYNDROME, sizeof(expect), 0, &expect, sizeof(expect), nullptr);
    }
    const bool pass = (got == expect);
    std::fprintf(stderr, "[%s] got=0x%llx expect=0x%llx -> %s\n", role.c_str(),
                 static_cast<unsigned long long>(got), static_cast<unsigned long long>(expect),
                 pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
