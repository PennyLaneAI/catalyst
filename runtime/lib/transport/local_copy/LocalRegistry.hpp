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

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "Transport.hpp"

namespace catalyst::transport::local_copy {

class LocalCpuControllerSession;

// A process-local rendezvous between a controller and its coprocessor. Both sides look up the
// same EndpointPair by (peer, oob_port) at connect(); the coprocessor registers `run_once` so
// the controller's kick() can drive the coprocessor synchronously in-process.
struct EndpointPair {
    using RunOnce = std::function<std::size_t(const void *req, std::size_t req_bytes,
                                              std::uint32_t decoder_id, void *reply,
                                              std::size_t reply_cap)>;

    LocalCpuControllerSession *controller = nullptr;
    RunOnce run_once;
};

inline auto acquire_endpoint_pair(const ConnectInfo &info) -> std::shared_ptr<EndpointPair> {
    static std::mutex mu;
    // Process-wide weak_ptr registry so dead pairs can disappear once the respective
    // controller/coprocessor are destroyed.
    static std::unordered_map<std::string, std::weak_ptr<EndpointPair>> pairs;

    const std::string key = info.peer + ":" + std::to_string(info.oob_port);
    std::lock_guard<std::mutex> lock(mu);

    if (auto it = pairs.find(key); it != pairs.end()) {
        if (auto pair = it->second.lock()) {
            return pair;
        }
    }

    auto pair = std::make_shared<EndpointPair>();
    pairs[key] = pair;
    return pair;
}

} // namespace catalyst::transport::local_copy
