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

#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#include "HwhsControllerSession.hpp"
#include "TransportBackend.h"
#include "WireProtocol.hpp"

namespace {

std::string cfg_get(const std::string &config, const std::string &key, const std::string &dflt) {
    const std::string needle = key + "=";
    std::size_t pos = 0;
    while (pos <= config.size()) {
        std::size_t semi = config.find(';', pos);
        std::size_t len = (semi == std::string::npos) ? std::string::npos : semi - pos;
        std::string tok = config.substr(pos, len);
        if (tok.rfind(needle, 0) == 0) {
            return tok.substr(needle.size());
        }
        if (semi == std::string::npos) {
            break;
        }
        pos = semi + 1;
    }
    return dflt;
}

std::optional<catalyst::transport::MemKind> mem_kind_opt(const std::string &config,
                                                         const char *key) {
    std::string v = cfg_get(config, key, "");
    if (v.empty()) {
        return std::nullopt;
    }
    if (v == "ps") {
        return catalyst::transport::MemKind::CpuRam;
    }
    if (v == "pl") {
        return catalyst::transport::MemKind::Ddr;
    }
    if (v == "bram") {
        return catalyst::transport::MemKind::Other;
    }
    throw std::runtime_error("unknown mem placement '" + v + "' (want ps|pl|bram) for key " + key);
}

catalyst::transport::ControllerSession *make_hwhs_controller(const std::string &config) {
    std::string dev = cfg_get(config, "dev", "xib_0");
    int gid = std::stoi(cfg_get(config, "gid", "1"));
    // Defaults to the wire protocol's ring size: the peer's slot index is cursor % K_RING_SLOTS.
    auto ring = static_cast<std::uint32_t>(std::stoul(
        cfg_get(config, "ring", std::to_string(catalyst::transport::common::K_RING_SLOTS))));
    auto stride_log2 = static_cast<std::uint32_t>(std::stoul(cfg_get(config, "stride_log2", "6")));
    auto data_kind = mem_kind_opt(config, "data_mem");
    auto sq_kind = mem_kind_opt(config, "sq_mem");
    auto reply_kind = mem_kind_opt(config, "reply_mem");

    std::string rp = cfg_get(config, "reply_poll", "sw");
    if (rp != "sw" && rp != "hw") {
        throw std::runtime_error("unknown reply_poll '" + rp + "' (want sw|hw)");
    }
    bool cpu_poll = (rp == "sw");
    std::string kk = cfg_get(config, "kick", "mmap");
    if (kk != "mmap" && kk != "ioctl") {
        throw std::runtime_error("unknown kick '" + kk + "' (want mmap|ioctl)");
    }
    bool kick_ioctl = (kk == "ioctl");

    catalyst::transport::hwhs::DemoCfg demo;
    std::string dm = cfg_get(config, "demo", "off");
    if (dm != "off" && dm != "hw") {
        throw std::runtime_error("unknown demo '" + dm + "' (want off|hw)");
    }
    demo.enable = (dm == "hw");
    demo.freq_num = std::stoull(cfg_get(config, "demo_freq", "0"));
    demo.cmd_cnt = static_cast<std::uint32_t>(std::stoul(cfg_get(config, "demo_cnt", "1")));
    demo.syn_depth = static_cast<std::uint32_t>(std::stoul(cfg_get(config, "demo_depth", "256")));
    demo.bram_pa = std::stoull(cfg_get(config, "demo_bram", "0x80000000"), nullptr, 0);
    demo.freq_span =
        static_cast<std::uint32_t>(std::stoul(cfg_get(config, "demo_span", "0"), nullptr, 0));
    demo.lfsr_seed =
        static_cast<std::uint32_t>(std::stoul(cfg_get(config, "demo_seed", "0"), nullptr, 0));
    //   demo_trace=<pa>    physical address the BD maps the RTT trace RAM at.
    //   demo_table=<file>  a byte-for-byte image of the BRAM table
    //   demo_trace_out=<f> where to write the trace
    demo.trace_pa = std::stoull(cfg_get(config, "demo_trace", "0x92000000"), nullptr, 0);
    demo.trace_out = cfg_get(config, "demo_trace_out", "");
    demo.table_path = cfg_get(config, "demo_table", "");

    return new catalyst::transport::hwhs::HwhsControllerSession(
        dev, gid, ring, stride_log2, data_kind, sq_kind, reply_kind, cpu_poll, kick_ioctl, demo);
}

} // namespace

GENERATE_TRANSPORT_CONTROLLER_FACTORY(CatalystTransportController, make_hwhs_controller)
