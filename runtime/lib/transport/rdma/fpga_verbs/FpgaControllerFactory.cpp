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

// Plugin entry point for the FPGA-verbs controller backend. The runtime dlopen's
// this .so and resolves CatalystTransportControllerFactory (see TransportBackend.h).

#include <string>

#include "FpgaBackendConfig.hpp"
#include "FpgaControllerSession.hpp"
#include "TransportBackend.h"

namespace {
catalyst::transport::ControllerSession *make_fpga_controller(const std::string &config)
{
    const auto cfg = rdma::devices::fpga_verbs::parse_fpga_config(config);
    return new rdma::devices::fpga_verbs::FpgaControllerSession(
        cfg.dev, cfg.gid, cfg.ring, cfg.stride_log2, cfg.data_mem, cfg.reply_mem);
}
} // namespace

GENERATE_TRANSPORT_CONTROLLER_FACTORY(CatalystTransportController, make_fpga_controller)
