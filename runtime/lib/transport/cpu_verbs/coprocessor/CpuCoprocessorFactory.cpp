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

// Plugin entry point for the CPU-verbs coprocessor backend. A loader dlopen's
// this .so and resolves CatalystTransportCoprocessorFactory (see TransportBackend.h).
// The coprocessor function is bound after construction via set_coprocessor_fn()
// (by the coprocessor-side harness); a freshly built session defaults to the
// built-in echo.

#include <string>

#include "CpuBackendConfig.hpp"
#include "CpuCoprocessorSession.hpp"
#include "TransportBackend.h"

namespace {
catalyst::transport::CoprocessorSession *make_cpu_coprocessor(const std::string &config)
{
    const auto cfg = catalyst::transport::cpu_verbs::parse_cpu_config(config);
    return new catalyst::transport::cpu_verbs::CpuCoprocessorSession(cfg.dev, cfg.gid);
}
} // namespace

GENERATE_TRANSPORT_COPROCESSOR_FACTORY(CatalystTransportCoprocessor, make_cpu_coprocessor)
