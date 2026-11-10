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
// built-in echo. The link speaks the wire the config names with `wire=` (WireLink.hpp).

#include <string>

#include "BackendConfig.hpp"
#include "CpuCoprocessorSession.hpp"
#include "TransportBackend.h"
#include "WireLink.hpp"

namespace {
catalyst::transport::CoprocessorSession *make_cpu_coprocessor(const std::string &config) {
    namespace common = catalyst::transport::common;
    namespace cpu_verbs = catalyst::transport::cpu_verbs;
    const int cpu = common::configparser::parse_optional_index(config, "cpu_pin", -1);
    const bool rt = common::configparser::parse_optional_index(config, "rt", 0) != 0;
    auto *session =
        new cpu_verbs::CpuCoprocessorSession(common::make_coprocessor_link(config, "cpu_verbs"));
    session->set_thread_affinity(cpu, rt);
    return session;
}
} // namespace

GENERATE_TRANSPORT_COPROCESSOR_FACTORY(CatalystTransportCoprocessor, make_cpu_coprocessor)
CATALYST_TRANSPORT_BACKEND_CAPS("cpu_verbs", CATALYST_TRANSPORT_ROLE_BIT_COPROCESSOR,
                                CATALYST_TRANSPORT_MEM_BIT(CpuRam),
                                CATALYST_TRANSPORT_FRAME_BL16 | CATALYST_TRANSPORT_FRAME_BLV,
                                CATALYST_TRANSPORT_CONV_BIT(PerMessage), UINT32_MAX, UINT32_MAX, 1)
