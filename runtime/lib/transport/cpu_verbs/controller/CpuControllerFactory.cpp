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

// Plugin entry point for the CPU-verbs controller backend. The runtime dlopen's
// this .so and resolves CatalystTransportControllerFactory (see TransportBackend.h).

#include <string>

#include "BackendConfig.hpp"
#include "CpuControllerSession.hpp"
#include "TransportBackend.h"

namespace {
catalyst::transport::ControllerSession *make_cpu_controller(const std::string &config) {
    const auto cfg = catalyst::transport::common::parse_backend_config(config);
    return new catalyst::transport::cpu_verbs::CpuControllerSession(cfg.dev, cfg.gid);
}
} // namespace

GENERATE_TRANSPORT_CONTROLLER_FACTORY(CatalystTransportController, make_cpu_controller)
