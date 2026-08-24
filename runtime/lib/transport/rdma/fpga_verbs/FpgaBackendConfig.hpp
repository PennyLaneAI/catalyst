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
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

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
    FpgaConfig cfg;
    for (std::size_t pos = 0; pos < config.size();) {
        const std::size_t sep = config.find(';', pos);
        const std::size_t end = (sep == std::string::npos) ? config.size() : sep;
        const std::string_view tok(config.data() + pos, end - pos);
        if (const std::size_t eq = tok.find('='); eq != std::string_view::npos) {
            const std::string_view key = tok.substr(0, eq);
            const std::string val(tok.substr(eq + 1));
            if (key == "dev") {
                cfg.dev = val;
            } else if (key == "gid") {
                cfg.gid = std::atoi(val.c_str());
            } else if (key == "ring") {
                cfg.ring = static_cast<std::uint32_t>(std::strtoul(val.c_str(), nullptr, 10));
            } else if (key == "stride_log2") {
                cfg.stride_log2 =
                    static_cast<std::uint32_t>(std::strtoul(val.c_str(), nullptr, 10));
            } else if (key == "data_mem") {
                cfg.data_mem = parse_mem_kind(key, val);
            } else if (key == "reply_mem") {
                cfg.reply_mem = parse_mem_kind(key, val);
            }
        }
        if (sep == std::string::npos) {
            break;
        }
        pos = sep + 1;
    }
    return cfg;
}

} // namespace rdma::devices::fpga_verbs
