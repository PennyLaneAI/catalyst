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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "CpuControllerSession.hpp"
#include "CpuCoprocessorSession.hpp"
#include "WireProtocol.hpp"

using namespace catalyst::transport;
using namespace catalyst::transport::cpu_verbs;
using namespace catalyst::transport::common; // REGION_BYTES, DEMO_SYNDROME, Payload

int main(int argc, char **argv) {
    std::string role = "coprocessor", dev = "rxe0", peer = "127.0.0.1";
    int gid = 1;
    std::uint16_t port = 18560;
    // Message sizes (bytes). The controller role feeds these to commit_work_item;
    // both roles size the collect() buffer by --correction-bytes.
    std::uint32_t syndrome_bytes = sizeof(std::uint64_t);
    std::uint32_t correction_bytes = sizeof(std::uint64_t);
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
        }
    }
    const bool is_coprocessor = (role == "coprocessor");

    std::unique_ptr<TransportSession> s;
    CpuCoprocessorSession *coproc = nullptr;
    CpuControllerSession *controller = nullptr;
    if (is_coprocessor) {
        auto up = std::make_unique<CpuCoprocessorSession>(dev, gid);
        coproc = up.get();
        s = std::move(up);
    } else {
        auto up = std::make_unique<CpuControllerSession>(dev, gid);
        controller = up.get();
        s = std::move(up);
    }

    ConnectInfo ci{
        .peer = peer,
        .oob_port = port,
    };
    s->connect(ci);
    MemRegion m = s->alloc_memory(REGION_BYTES, MemKind::CpuRam);
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
        coproc->set_coprocessor_fn(nullptr, nullptr); // built-in echo
        coproc->start();
        std::this_thread::sleep_for(std::chrono::seconds(3)); // serve ~3 s
        coproc->collect(outs, obytes, 1);
        coproc->stop();
    } else {
        // Controller: commit a work item sized by --syndrome-bytes/--correction-bytes,
        // write the syndrome into the outbound slot, kick one round, collect the correction.
        controller->commit_work_item(/*work_item_idx=*/0, syndrome_bytes, correction_bytes);
        controller->start();
        const std::uint64_t syndrome = DEMO_SYNDROME;
        controller->write_data_slot(&syndrome, syndrome_bytes, /*decoder_id=*/0);
        controller->kick(0);
        controller->collect(outs, obytes, 1);
        controller->stop();
    }

    std::uint64_t got = 0;
    std::memcpy(&got, corr.data(), sizeof(got));

    // Built-in echo coprocessor: both roles observe the demo syndrome.
    const bool pass = (got == DEMO_SYNDROME);
    std::fprintf(stderr, "[%s] got=0x%llx expect=0x%llx -> %s\n", role.c_str(),
                 static_cast<unsigned long long>(got),
                 static_cast<unsigned long long>(DEMO_SYNDROME), pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
