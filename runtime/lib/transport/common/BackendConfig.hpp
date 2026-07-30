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
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include "Error.hpp"

namespace catalyst::transport::common {

// Construction parameters shared by the verbs-based backends, parsed from a
// backend factory `config` string ("key=value;..."). Connection parameters
// (peer/oob_port) are not here — they arrive later via connect(). Both keys are
// required and have no defaults, e.g. "dev=rxe0;gid=1".
struct BackendConfig {
    std::string dev; // RDMA device name
    int gid;         // GID index on that device's port
};

namespace backendconfig {

/**
 * @brief Split a "key=value;..." config string and invoke `fn` for each entry.
 *
 * Entries are separated by ';' and split on their first '=', so a value may
 * itself contain '='.
 *
 * @tparam Fn Callable as `fn(std::string_view key, std::string_view value)`.
 * @param config The config string, e.g. "dev=rxe0;gid=1".
 * @param fn Invoked once per entry, in the order the entries appear.
 */
template <class Fn> void for_each_config_kv(std::string_view config, Fn fn)
{
    while (!config.empty()) {
        const std::size_t sep = config.find(';');
        const std::string_view tok = config.substr(0, sep);
        if (const std::size_t eq = tok.find('='); eq != std::string_view::npos) {
            fn(tok.substr(0, eq), tok.substr(eq + 1));
        }
        if (sep == std::string_view::npos) {
            break;
        }
        config.remove_prefix(sep + 1);
    }
}

// Parse a non-negative integer, throwing on anything malformed.
inline int parse_index(std::string_view val, const char *key)
{
    int out = 0;
    const char *last = val.data() + val.size();
    const auto res = std::from_chars(val.data(), last, out);
    RDMA_CHECK(res.ec == std::errc{} && res.ptr == last && out >= 0,
               "config: '%s' must be a non-negative integer, got '%.*s'", key,
               static_cast<int>(val.size()), val.data());
    return out;
}

} // namespace backendconfig

/**
 * @brief Parse the `dev` and `gid` keys out of a backend `config` string.
 *
 * Unknown keys are ignored, so a backend may carry extra keys of its own and
 * pull them out with parse_optional_index(). Both keys here are mandatory: an
 * absent, empty or malformed value throws exception.
 *
 * @param config The config string, e.g. "dev=rxe0;gid=1".
 * @return The parsed device name and GID index.
 */
inline BackendConfig parse_backend_config(const std::string &config)
{
    std::string dev;
    int gid = 0;
    bool have_dev = false, have_gid = false;
    backendconfig::for_each_config_kv(config, [&](std::string_view key, std::string_view val) {
        if (key == "dev") {
            RDMA_CHECK(!val.empty(), "config: 'dev' must not be empty");
            dev = std::string(val);
            have_dev = true;
        }
        else if (key == "gid") {
            gid = backendconfig::parse_index(val, "gid");
            have_gid = true;
        }
    });
    RDMA_CHECK(have_dev && have_gid,
               "config requires both 'dev' and 'gid' (e.g. \"dev=rxe0;gid=1\"), got \"%s\"",
               config.c_str());
    return BackendConfig{std::move(dev), gid};
}

/**
 * @brief Look up an optional non-negative integer key in a backend `config`
 * string.
 *
 * For backend-specific keys that have a safe default, unlike `dev`/`gid`. An
 * absent key yields `fallback`; a present but malformed one throws, so a typo
 * fails loudly instead of silently taking the default.
 *
 * @param config The factory config string.
 * @param key Key to look for, e.g. "gpu".
 * @param fallback Value to return when the key is absent.
 * @return The parsed value, or `fallback` if the key is not present.
 */
inline int parse_optional_index(const std::string &config, const char *key, int fallback)
{
    int out = fallback;
    backendconfig::for_each_config_kv(config, [&](std::string_view k, std::string_view val) {
        if (k == key) {
            out = backendconfig::parse_index(val, key);
        }
    });
    return out;
}

} // namespace catalyst::transport::common
