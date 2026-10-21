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

// HwhsBackend.cpp - the plugin entry point for the VPK120 HWHS transport backend.

#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include "ConfigParser.hpp"
#include "FpgaBackendConfig.hpp"
#include "HwhsControllerSession.hpp"
#include "TransportBackend.h"

namespace {

namespace common = catalyst::transport::common;
namespace cp = catalyst::transport::common::configparser;

// `key`'s value, `dflt` when absent; throws unless it is one of `allowed`.
std::string pick(const std::string &config, const char *key, const char *dflt,
                 std::initializer_list<std::string_view> allowed) {
    const std::string v{cp::find(config, key).value_or(dflt)};
    for (std::string_view a : allowed) {
        if (v == a) {
            return v;
        }
    }
    std::string want;
    for (std::string_view a : allowed) {
        want += (want.empty() ? "" : "|") + std::string{a};
    }
    throw std::runtime_error("unknown " + std::string{key} + " '" + v + "' (want " + want + ")");
}

// `key` as an unsigned integer (decimal or 0x hex), `dflt` when absent.
std::uint64_t number(const std::string &config, const char *key, std::uint64_t dflt) {
    const auto v = cp::find(config, key);
    return v ? cp::parse_u64(*v, key) : dflt;
}

catalyst::transport::ControllerSession *make_hwhs_controller(const std::string &config) {
    // dev, gid, ring, stride_log2, data_mem and reply_mem, with fpga_verbs' defaults.
    const common::FpgaConfig fpga = common::parse_fpga_config(config);
    std::optional<catalyst::transport::MemKind> sq_kind;
    if (const auto v = cp::find(config, "sq_mem")) {
        sq_kind = common::parse_mem_kind("sq_mem", std::string{*v});
    }
    const bool cpu_poll = pick(config, "reply_poll", "sw", {"sw", "hw"}) == "sw";
    const bool kick_ioctl = pick(config, "kick", "mmap", {"mmap", "ioctl"}) == "ioctl";

    catalyst::transport::hwhs::DemoCfg demo;
    demo.enable = pick(config, "demo", "off", {"off", "hw"}) == "hw";
    demo.freq_num = number(config, "demo_freq", 0);
    demo.cmd_cnt = static_cast<std::uint32_t>(number(config, "demo_cnt", 1));
    demo.syn_depth = static_cast<std::uint32_t>(number(config, "demo_depth", 256));
    demo.bram_pa = number(config, "demo_bram", 0x80000000);
    demo.freq_span = static_cast<std::uint32_t>(number(config, "demo_span", 0));
    demo.lfsr_seed = static_cast<std::uint32_t>(number(config, "demo_seed", 0));
    //   demo_trace=<pa>    physical address the BD maps the RTT trace RAM at.
    //   demo_table=<file>  a byte-for-byte image of the BRAM table
    //   demo_trace_out=<f> where to write the trace
    demo.trace_pa = number(config, "demo_trace", 0x92000000);
    demo.trace_out = std::string{cp::find(config, "demo_trace_out").value_or("")};
    demo.table_path = std::string{cp::find(config, "demo_table").value_or("")};

    return new catalyst::transport::hwhs::HwhsControllerSession(
        fpga.dev, fpga.gid, fpga.ring, fpga.stride_log2, fpga.data_mem, sq_kind, fpga.reply_mem,
        cpu_poll, kick_ioctl, demo);
}

} // namespace

GENERATE_TRANSPORT_CONTROLLER_FACTORY(CatalystTransportController, make_hwhs_controller)
CATALYST_TRANSPORT_BACKEND_CAPS("hwhs", CATALYST_TRANSPORT_ROLE_BIT_CONTROLLER,
                                CATALYST_TRANSPORT_MEM_BIT(CpuRam) |
                                    CATALYST_TRANSPORT_MEM_BIT(Ddr) |
                                    CATALYST_TRANSPORT_MEM_BIT(Other),
                                0, 8, 8, 1)
