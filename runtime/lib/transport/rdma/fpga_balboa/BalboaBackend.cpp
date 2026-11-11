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

// BalboaBackend.cpp - the plugin entry point for the VPK120 controller behind BALBOA (hwhs-balboa).
//
// Config keys, all optional:
//   reply_to_us=<n>     the engine gives up on a reply after this long (default 1 s)
//   pacer=off|hw        hw: the PL pacer issues every round (BalboaControllerSession.hpp)
//   pacer_freq, pacer_cnt, pacer_depth, pacer_span, pacer_seed, pacer_trace, pacer_trace_out,
//   pacer_table         as the hwhs backend's keys of the same name

#include <cstdint>
#include <stdexcept>
#include <string>

#include "BalboaControllerSession.hpp"
#include "ConfigParser.hpp"
#include "TransportBackend.h"

namespace {

namespace cp = catalyst::transport::common::configparser;

// `key` as an unsigned integer (decimal or 0x hex), `dflt` when absent.
std::uint64_t number(const std::string &config, const char *key, std::uint64_t dflt) {
    const auto v = cp::find(config, key);
    return v ? cp::parse_u64(*v, key) : dflt;
}

catalyst::transport::ControllerSession *make_balboa_controller(const std::string &config) {
    const std::string pacer_mode{cp::find(config, "pacer").value_or("off")};
    if (pacer_mode != "off" && pacer_mode != "hw") {
        throw std::runtime_error("unknown pacer '" + pacer_mode + "' (want off|hw)");
    }
    catalyst::transport::balboa::PacerCfg pacer;
    pacer.enable = pacer_mode == "hw";
    pacer.freq = number(config, "pacer_freq", 0);
    pacer.cnt = static_cast<std::uint32_t>(number(config, "pacer_cnt", 1));
    pacer.depth = static_cast<std::uint32_t>(number(config, "pacer_depth", 256));
    pacer.span = static_cast<std::uint32_t>(number(config, "pacer_span", 0));
    pacer.seed = static_cast<std::uint32_t>(number(config, "pacer_seed", 0));
    // The trace RAM is fixed on this board, so pacer_trace can only confirm where it is.
    if (number(config, "pacer_trace", HWHS_PHYS_TRACE) != HWHS_PHYS_TRACE) {
        throw std::runtime_error("pacer_trace: the trace RAM is at 0x92000000 on this board");
    }
    pacer.trace_out = std::string{cp::find(config, "pacer_trace_out").value_or("")};
    pacer.table_path = std::string{cp::find(config, "pacer_table").value_or("")};

    return new catalyst::transport::balboa::BalboaControllerSession(
        number(config, "reply_to_us", 1'000'000), pacer);
}

} // namespace

GENERATE_TRANSPORT_CONTROLLER_FACTORY(CatalystTransportController, make_balboa_controller)
CATALYST_TRANSPORT_BACKEND_CAPS("balboa_hwhs", CATALYST_TRANSPORT_ROLE_BIT_CONTROLLER,
                                CATALYST_TRANSPORT_MEM_BIT(CpuRam) |
                                    CATALYST_TRANSPORT_MEM_BIT(Other),
                                CATALYST_TRANSPORT_FRAME_BL16, 0, 8, 8, 1)
