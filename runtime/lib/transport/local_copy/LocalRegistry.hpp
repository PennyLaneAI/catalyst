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

// Same-process rendezvous keyed by (peer, oob_port). The coprocessor binds `run_once`; the
// controller's kick() drives it inline. `mu` serializes kick against coprocessor teardown.
struct EndpointPair {
    using RunOnce = std::function<std::size_t(const void *req, std::size_t req_bytes,
                                              std::uint32_t decoder_id, void *reply,
                                              std::size_t reply_cap)>;

    std::mutex mu;
    LocalCpuControllerSession *controller = nullptr;
    RunOnce run_once;
};

inline auto acquire_endpoint_pair(const ConnectInfo &info) -> std::shared_ptr<EndpointPair> {
    static std::mutex mu;
    static std::unordered_map<std::string, std::weak_ptr<EndpointPair>> pairs;

    const std::string key = info.peer + ":" + std::to_string(info.oob_port);
    std::lock_guard<std::mutex> lock(mu);

    // Prune expired entries so churning keys don't accumulate.
    for (auto it = pairs.begin(); it != pairs.end();) {
        if (it->second.expired()) {
            it = pairs.erase(it);
        } else {
            ++it;
        }
    }

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
