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

#include "BackendConfig.hpp"
#include "GpuCoprocessorSession.hpp"
#include "TransportBackend.h"

namespace {
catalyst::transport::CoprocessorSession *make_gpu_coprocessor(const std::string &config) {
    const auto cfg = catalyst::transport::common::parse_backend_config(config);
    // `gpu` is optional and defaults to device 0
    const int gpu_device = catalyst::transport::common::configparser::parse_optional_index(
        config, "gpu", /*fallback=*/0);
    auto *session =
        new catalyst::transport::gpu_verbs::GpuCoprocessorSession(cfg.dev, cfg.gid, gpu_device);
    session->set_thread_affinity(
        catalyst::transport::common::configparser::parse_optional_index(config, "cpu_pin", -1),
        catalyst::transport::common::configparser::parse_optional_index(config, "rt", 0) != 0);
    return session;
}
} // namespace

GENERATE_TRANSPORT_COPROCESSOR_FACTORY(CatalystTransportCoprocessor, make_gpu_coprocessor)
