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

#include <atomic>
#include <cstdint>
#include <cstring>
#include <vector>

#include "Frame.hpp"
#include "GpuLaunchers.hpp"
#include "GpuRuntime.hpp"
#include "HipCheck.hpp"
#include "SteaneLut.hpp"
#include "TransportABI.h"

#include <catch2/catch_test_macros.hpp>
#include <hip/hip_runtime.h>

using namespace catalyst::transport;
using namespace catalyst::transport::common;
using namespace catalyst::transport::coproc;

namespace {
// Runs `launcher` over `rounds` messages in `ring`'s layout, staging each request on the host the
// way a memcpy controller does, and returns the reply frames the kernel wrote.
struct Reply {
    std::vector<std::uint8_t> data;
    std::uint32_t fn_id;
    std::int32_t status;
    std::uint32_t request_id;
};
std::vector<Reply> run_kernel(CoprocessorLauncherFn launcher, frame::RingGeometry ring,
                              const std::vector<std::vector<std::uint8_t>> &requests) {
    GpuRuntime gpu(0);
    std::uint8_t *req_host = nullptr;
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&req_host), ring.region_bytes(),
                            hipHostMallocMapped | hipHostMallocCoherent),
              "hipHostMalloc(requests)");
    std::memset(req_host, 0, ring.region_bytes());
    void *req_dev = nullptr;
    HIP_CHECK(hipHostGetDevicePointer(&req_dev, req_host, 0), "hipHostGetDevicePointer");
    // CUQR frames come with a flag per slot each way, after the reply ring, as gpu_verbs lays them.
    const bool cuqr = ring.mode == frame::Mode::CUQR;
    const std::size_t flag_bytes = cuqr ? 2 * ring.slots * sizeof(std::uint32_t) : 0;
    GpuRuntime::Handoff replies = gpu.alloc_handoff(ring.region_bytes() + flag_bytes);
    auto *rx_host = reinterpret_cast<std::uint32_t *>(replies.host + ring.region_bytes());
    auto *tx_host = rx_host + ring.slots;
    std::memset(rx_host, 0, flag_bytes);

    CoprocLaunchDesc desc{
        .ring = req_dev,
        .ring_slots = ring.slots,
        .handoff = replies.dev,
        .stop = replies.stop_dev,
        .total = requests.size(),
        .stream = gpu.stream(),
        .frame_mode = static_cast<std::uint32_t>(ring.mode),
        .slot_stride = ring.stride,
        .rx_flags = cuqr ? replies.dev + ring.region_bytes() : nullptr,
        .tx_flags =
            cuqr ? replies.dev + ring.region_bytes() + ring.slots * sizeof(std::uint32_t) : nullptr,
    };
    REQUIRE(launcher(&desc, nullptr) == 0);
    std::vector<Reply> out;
    for (std::uint64_t c = 0; c < requests.size(); ++c) {
        const auto expect = static_cast<std::uint32_t>(c + 1);
        std::uint8_t *slot = req_host + ring.slot_offset(c);
        frame::stage(ring, slot, /*fn_id=*/static_cast<std::uint32_t>(c + 5), requests[c].data(),
                     requests[c].size());
        frame::seal(ring, slot, expect);
        const std::size_t s = c & (ring.slots - 1);
        if (cuqr) { // what gpu_verbs does once the receive completion queue shows the request
            std::atomic_ref<std::uint32_t>(rx_host[s]).store(expect, std::memory_order_release);
            while (std::atomic_ref<std::uint32_t>(tx_host[s]).load(std::memory_order_acquire) !=
                   expect) {
            }
        }
        frame::FrameView view;
        while (!frame::poll(ring, replies.host + ring.slot_offset(c), expect, view)) {
        }
        out.push_back({std::vector<std::uint8_t>(view.data, view.data + view.len), view.fn_id,
                       view.status, view.request_id});
    }
    gpu.sync();
    (void)hipHostFree(req_host);
    return out;
}
} // namespace

TEST_CASE("the GPU launchers report the frame layout they are built for") {
    CHECK(gpu_echo_launcher_info()->abi_version >= CATALYST_COPROCESSOR_LAUNCHER_MIN_ABI_VERSION);
    CHECK(gpu_steane_launcher_info()->abi_version >= CATALYST_COPROCESSOR_LAUNCHER_MIN_ABI_VERSION);
}

TEST_CASE("the GPU echo kernel answers 16 B frames with the request's payload and fn_id") {
    std::vector<std::vector<std::uint8_t>> requests;
    for (std::uint8_t r = 0; r < 3; ++r) {
        requests.push_back({r, 1, 2, 3, 4, 5, 6, 7});
    }
    const auto replies = run_kernel(gpu_echo_launcher, frame::BL16_RING, requests);
    for (std::size_t c = 0; c < requests.size(); ++c) {
        CHECK(replies[c].fn_id == c + 5);
        CHECK(replies[c].data == requests[c]);
    }
}

TEST_CASE("the GPU echo kernel answers BLV frames over 8 B, past one lap of the ring") {
    const auto ring = frame::geometry_for(frame::Mode::BLV, 300);
    std::vector<std::vector<std::uint8_t>> requests;
    for (std::uint32_t r = 0; r < ring.slots + 4; ++r) {
        std::vector<std::uint8_t> payload(300 -
                                          r % 7); // lengths vary, so a stale length would show
        for (std::size_t i = 0; i < payload.size(); ++i) {
            payload[i] = static_cast<std::uint8_t>(i * 13 + r);
        }
        requests.push_back(payload);
    }
    const auto replies = run_kernel(gpu_echo_launcher, ring, requests);
    for (std::size_t c = 0; c < requests.size(); ++c) {
        REQUIRE(replies[c].fn_id == c + 5);
        REQUIRE(replies[c].data == requests[c]);
    }
}

TEST_CASE("the GPU Steane kernel decodes an unpacked syndrome in every frame layout") {
    for (const auto ring : {frame::BL16_RING, frame::geometry_for(frame::Mode::BLV, 64),
                            frame::geometry_for(frame::Mode::CUQR, 64)}) {
        std::vector<std::vector<std::uint8_t>> requests;
        for (std::uint8_t packed = 0; packed < (1 << STEANE_CHECKS); ++packed) {
            // Check 0 is the most significant bit of the packed syndrome.
            requests.push_back({static_cast<std::uint8_t>(packed >> 2 & 1U),
                                static_cast<std::uint8_t>(packed >> 1 & 1U),
                                static_cast<std::uint8_t>(packed & 1U), 0, 0, 0, 0, 0});
        }
        const auto replies = run_kernel(gpu_steane_launcher, ring, requests);
        for (std::size_t packed = 0; packed < requests.size(); ++packed) {
            REQUIRE(replies[packed].data.size() >= sizeof(std::int64_t));
            std::int64_t qubit = 0;
            std::memcpy(&qubit, replies[packed].data.data(), sizeof(qubit));
            CHECK(qubit == STEANE_SYNDROME_TO_QUBIT[packed]);
        }
    }
}

TEST_CASE("the GPU echo kernel answers CUQR frames through the slot flags, echoing request ids") {
    const auto ring = frame::geometry_for(frame::Mode::CUQR, 300);
    std::vector<std::vector<std::uint8_t>> requests;
    for (std::uint32_t r = 0; r < ring.slots + 4; ++r) {
        std::vector<std::uint8_t> payload(300 - r % 7); // lengths vary, so a stale one would show
        for (std::size_t i = 0; i < payload.size(); ++i) {
            payload[i] = static_cast<std::uint8_t>(i * 13 + r);
        }
        requests.push_back(payload);
    }
    const auto replies = run_kernel(gpu_echo_launcher, ring, requests);
    for (std::size_t c = 0; c < requests.size(); ++c) {
        REQUIRE(replies[c].status == 0);
        REQUIRE(replies[c].request_id == c + 1); // the request's id, echoed
        REQUIRE(replies[c].data == requests[c]);
    }
}
