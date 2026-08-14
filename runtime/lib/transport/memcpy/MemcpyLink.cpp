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

// Definition of the process-global memcpy link registry. This lives in its own shared library
// so all memcpy plugins dlopen'd into a process share one instance of the static map, letting
// a controller and coprocessor loaded from separate .so files find each other by
// (peer, oob_port).

#include "MemcpyLink.hpp"

#include <stdexcept>
#include <string>
#include <unordered_map>

#include "BackendConfig.hpp"

namespace catalyst::transport::memcpy {

auto parse_pair_key(std::string_view config) -> std::string {
    std::string out;
    common::backendconfig::for_each_config_kv(config, [&](std::string_view k, std::string_view v) {
        if (k == "pair") {
            out.assign(v);
        }
    });
    return out;
}

auto acquire_memcpy_link(std::string_view pair_key) -> std::shared_ptr<MemcpyLink> {
    if (pair_key.empty()) {
        throw std::runtime_error(
            "memcpy: missing session pair key in backend config (expected 'pair=<key>')");
    }

    static std::mutex mu;
    static std::unordered_map<std::string, std::weak_ptr<MemcpyLink>> links;

    const std::string key(pair_key);
    std::lock_guard<std::mutex> lock(mu);

    // Prune expired entries so churning keys don't accumulate.
    for (auto it = links.begin(); it != links.end();) {
        if (it->second.expired()) {
            it = links.erase(it);
        } else {
            ++it;
        }
    }

    if (auto it = links.find(key); it != links.end()) {
        if (auto link = it->second.lock()) {
            return link;
        }
    }

    auto link = std::make_shared<MemcpyLink>();
    links[key] = link;
    return link;
}

} // namespace catalyst::transport::memcpy
