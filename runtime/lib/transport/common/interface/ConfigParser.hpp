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

#pragma once
#include <charconv>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

#include "Error.hpp"

// Backend-agnostic parser for "key=value;..." config strings. No RDMA/HIP dependencies —
// usable by every transport backend (memcpy included).

namespace catalyst::transport::common::configparser {

/**
 * @brief Split a "key=value;..." config string and invoke `fn` for each entry.
 *
 * Entries are separated by ';' and split on their first '=', so a value may itself contain '='.
 *
 * @tparam Fn Callable as `fn(std::string_view key, std::string_view value)`.
 * @param config The config string, e.g. "dev=rxe0;gid=1".
 * @param fn Invoked once per entry, in the order the entries appear.
 */
template <class Fn> void for_each_kv(std::string_view config, Fn fn) {
    while (!config.empty()) {
        const std::size_t sep = config.find(';');
        const std::string_view tok =
            config.substr(0, sep == std::string_view::npos ? config.size() : sep);
        if (const std::size_t eq = tok.find('='); eq != std::string_view::npos) {
            fn(tok.substr(0, eq), tok.substr(eq + 1));
        }
        if (sep == std::string_view::npos) {
            break;
        }
        config.remove_prefix(sep + 1);
    }
}

/// Parse a non-negative integer from `val`; throw with a message naming `key` on any error.
inline int parse_index(std::string_view val, const char *key) {
    int out = 0;
    const char *last = val.data() + val.size();
    const auto res = std::from_chars(val.data(), last, out);
    if (res.ec != std::errc{} || res.ptr != last || out < 0) {
        char buf[192];
        std::snprintf(buf, sizeof(buf), "config: '%s' must be a non-negative integer, got '%.*s'",
                      key, static_cast<int>(val.size()), val.data());
        throw TransportError(buf);
    }
    return out;
}

/// Look up an optional non-negative integer key in a `key=value;...` config string.
/// Absent key yields `fallback`; a present but malformed one throws.
inline int parse_optional_index(const std::string &config, const char *key, int fallback) {
    int out = fallback;
    for_each_kv(config, [&](std::string_view k, std::string_view val) {
        if (k == key) {
            out = parse_index(val, key);
        }
    });
    return out;
}

} // namespace catalyst::transport::common::configparser
