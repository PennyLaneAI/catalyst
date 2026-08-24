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
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include "ConfigParser.hpp"
#include "Transport.hpp"
#include "WireProtocol.hpp"

namespace rdma::devices::fpga_verbs {

// Construction parameters parsed from a backend factory `config` string
// ("key=value;..."). Connection parameters (peer/oob_port) are not here — they
// arrive later via connect(). Recognised keys: `dev`, `gid`, `ring`,
// `stride_log2`, `data_mem`, `reply_mem`.
struct FpgaConfig {
    // RDMA device name.
    std::string dev = "xib_0";
    // GID index.
    int gid = 1;
    // Ring slots.
    std::uint32_t ring = static_cast<std::uint32_t>(catalyst::transport::common::K_RING_SLOTS);
    // Slot stride, log2 (6 = 64 B).
    std::uint32_t stride_log2 = 6;
    // Request-ring placement; unset lets the session choose.
    std::optional<catalyst::transport::MemKind> data_mem;
    // Reply-ring placement; unset lets the session choose.
    std::optional<catalyst::transport::MemKind> reply_mem;
};

// `ps` (PS DDR), `pl` (PL DDR) and `bram` are the placements the board's
// allocator exposes.
inline catalyst::transport::MemKind parse_mem_kind(std::string_view key, const std::string &val) {
    if (val == "ps") {
        return catalyst::transport::MemKind::CpuRam;
    }
    if (val == "pl") {
        return catalyst::transport::MemKind::Ddr;
    }
    if (val == "bram") {
        return catalyst::transport::MemKind::Other;
    }
    throw std::runtime_error("unknown mem placement '" + val + "' (want ps|pl|bram) for key " +
                             std::string(key));
}

inline FpgaConfig parse_fpga_config(const std::string &config) {
    namespace cp = catalyst::transport::common::configparser;
    FpgaConfig cfg;
    cp::for_each_kv(config, [&](std::string_view key, std::string_view val) {
        if (key == "dev") {
            cfg.dev = std::string(val);
        } else if (key == "gid") {
            cfg.gid = cp::parse_index(val, "gid");
        } else if (key == "ring") {
            cfg.ring = static_cast<std::uint32_t>(cp::parse_index(val, "ring"));
        } else if (key == "stride_log2") {
            cfg.stride_log2 = static_cast<std::uint32_t>(cp::parse_index(val, "stride_log2"));
        } else if (key == "data_mem") {
            cfg.data_mem = parse_mem_kind(key, std::string(val));
        } else if (key == "reply_mem") {
            cfg.reply_mem = parse_mem_kind(key, std::string(val));
        }
    });
    return cfg;
}

} // namespace rdma::devices::fpga_verbs
