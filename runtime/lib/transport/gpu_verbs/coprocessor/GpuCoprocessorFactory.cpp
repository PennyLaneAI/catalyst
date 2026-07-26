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

/**
 * Plugin entry point for the GPU-verbs coprocessor backend. A loader dlopen's
 * this .so and resolves CatalystTransportCoprocessorFactory.
 * The coprocessor function is bound after construction via
 * set_coprocessor_launcher(): a launch-once function invoked at start() to
 * launch the on-device decode kernel (the kernel itself lives in
 * GpuLaunchers.hip); nullptr selects the built-in echo launcher. This backend
 * supports only the launcher convention; binding a per-message function via
 * set_coprocessor_fn throws (the base class default), so a mis-bind fails at
 * bind time rather than hanging at run time.
 */

#include <string>

#include "GpuBackendConfig.hpp"
#include "GpuCoprocessorSession.hpp"
#include "TransportBackend.h"

namespace {
catalyst::transport::CoprocessorSession *make_gpu_coprocessor(const std::string &config)
{
    const auto cfg = catalyst::transport::gpu_verbs::parse_gpu_config(config);
    return new catalyst::transport::gpu_verbs::GpuCoprocessorSession(cfg.dev, cfg.gid);
}
} // namespace

GENERATE_TRANSPORT_COPROCESSOR_FACTORY(CatalystTransportCoprocessor, make_gpu_coprocessor)
