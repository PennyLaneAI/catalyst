// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Plugin entry point for the memcpy controller backend. The runtime dlopen's
// this .so and resolves CatalystTransportControllerFactory (see TransportBackend.h).

#include <string>

#include "CpuControllerSession.hpp"
#include "TransportBackend.h"

namespace {
catalyst::transport::ControllerSession *make_local_controller(const std::string &config) {
    return new catalyst::transport::memcpy::CpuControllerSession(config);
}
} // namespace

GENERATE_TRANSPORT_CONTROLLER_FACTORY(CatalystTransportController, make_local_controller)
CATALYST_TRANSPORT_BACKEND_CAPS("memcpy", CATALYST_TRANSPORT_ROLE_BIT_CONTROLLER,
                                CATALYST_TRANSPORT_MEM_BIT(CpuRam),
                                CATALYST_TRANSPORT_FRAME_BL16 | CATALYST_TRANSPORT_FRAME_BLV, 0,
                                UINT32_MAX, UINT32_MAX, 1)
