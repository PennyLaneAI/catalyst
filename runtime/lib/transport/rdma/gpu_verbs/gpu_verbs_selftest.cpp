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

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

#include "Frame.hpp"
#include "GpuCoprocessorSession.hpp"
#include "WireProtocol.hpp"

using namespace catalyst::transport;
using namespace catalyst::transport::gpu_verbs;
using namespace catalyst::transport::common; // frame, DEMO_SYNDROME

extern "C" int gpu_echo_launcher(const CoprocLaunchDesc *desc, void *ctx);

int main(int argc, char **argv) {
    // --dev and --gid are required
    std::string dev, peer = "0.0.0.0";
    int gid = -1;
    int gpu_device = 0;
    std::uint16_t port = 18560;
    std::uint32_t bytes = sizeof(std::uint64_t);
    bool cuqr = false, write_imm = false;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--dev") {
            dev = v;
        } else if (k == "--gid") {
            gid = std::atoi(v.c_str());
        } else if (k == "--gpu") {
            gpu_device = std::atoi(v.c_str());
        } else if (k == "--peer") {
            peer = v;
        } else if (k == "--port") {
            port = static_cast<std::uint16_t>(std::atoi(v.c_str()));
        } else if (k == "--bytes") {
            bytes = static_cast<std::uint32_t>(std::strtoul(v.c_str(), nullptr, 0));
        } else if (k == "--frame") {
            cuqr = v == "cuqr";
        } else if (k == "--tx") {
            write_imm = v == "write_imm";
        }
    }
    if (dev.empty() || gid < 0) {
        std::fprintf(stderr,
                     "usage: %s --dev <rdma_device> --gid <gid_index>"
                     " [--gpu <index>] [--peer <ip>] [--port <n>] [--bytes <n>]"
                     " [--frame cuqr [--tx send|write_imm]]\n"
                     "  --dev and --gid are required; the device needs dma-buf MR support\n"
                     "  --bytes is the message size each way (default 8), and must match the\n"
                     "  controller's; over 8 B the frames are BLV\n"
                     "  --frame cuqr uses cudaq-realtime's frames, completed by --tx\n",
                     argv[0]);
        return 2;
    }
    // The same layout cpu_verbs_selftest picks for these message sizes.
    const auto ring = frame::geometry_for(cuqr                          ? frame::Mode::CUQR
                                          : bytes <= PAYLOAD_DATA_BYTES ? frame::Mode::BL16
                                                                        : frame::Mode::BLV,
                                          bytes);
    GpuCoprocessorSession s(dev, gid, gpu_device, ring, write_imm);
    ConnectInfo ci{
        .peer = peer,
        .oob_port = port,
    };
    s.connect(ci);
    MemRegion m = s.alloc_memory(ring.region_bytes(), MemKind::GpuHbm);
    PeerRef p = s.exchange_keys(m);
    ChannelDesc desc{
        .transport = "rdma",
    };
    s.establish_channel(desc, m, p);
    s.set_coprocessor_launcher(gpu_echo_launcher, nullptr);
    s.start();
    std::this_thread::sleep_for(std::chrono::seconds(3));
    std::uint64_t got = 0;
    void *outs[1] = {&got};
    std::uint64_t obytes[1] = {sizeof(got)};
    s.collect(outs, obytes, 1);
    s.stop();
    const std::uint64_t expect = DEMO_SYNDROME;
    const bool pass = (got == expect);
    std::fprintf(stderr, "[coprocessor] got=0x%llx expect=0x%llx -> %s\n",
                 static_cast<unsigned long long>(got), static_cast<unsigned long long>(expect),
                 pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
