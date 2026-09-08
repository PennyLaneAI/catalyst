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

// Definitions for the process-global memcpy link registry. This TU is compiled into its own
// shared library (`libmemcpy_cpu_impl.so`) so the controller and coprocessor plugins, though
// dlopen'd separately, share one instance of the static map and rendezvous on the compiler-
// emitted session pair key.

#include "MemcpyLink.hpp"

#include <string>
#include <unordered_map>

#include "ConfigParser.hpp"
#include "Error.hpp"

namespace catalyst::transport::memcpy {

auto parse_pair_key(std::string_view config) -> std::string {
    std::string out;
    common::configparser::for_each_kv(config, [&](std::string_view k, std::string_view v) {
        if (k == "pair") {
            out.assign(v);
        }
    });
    return out;
}

auto acquire_memcpy_link(std::string_view pair_key) -> std::shared_ptr<MemcpyLink> {
    TP_CHECK(!pair_key.empty(), "Missing pair key in config (expected 'pair=<key>')");

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
