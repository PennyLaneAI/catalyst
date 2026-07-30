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

/**
 * @brief Parse the `dev` and `gid` keys out of a backend `config` string.
 *
 * Unknown keys are ignored, so a backend may carry extra keys of its own and
 * parse them separately. Both recognised keys are mandatory: an absent, empty
 * or malformed value throws rather than falling back to a built-in device.
 *
 * @param config The factory config string, e.g. "dev=rxe0;gid=1".
 * @param backend Backend name used in error messages, e.g. "cpu_verbs".
 * @return The parsed device name and GID index.
 */
inline BackendConfig parse_backend_config(const std::string &config, const char *backend)
{
    // Parsed into locals so a rejected config never yields a partially built
    // BackendConfig.
    std::string dev;
    int gid = 0;
    bool have_dev = false, have_gid = false;
    for (std::size_t pos = 0; pos < config.size();) {
        const std::size_t sep = config.find(';', pos);
        const std::size_t end = (sep == std::string::npos) ? config.size() : sep;
        const std::string_view tok(config.data() + pos, end - pos);
        if (const std::size_t eq = tok.find('='); eq != std::string_view::npos) {
            const std::string_view key = tok.substr(0, eq);
            const std::string_view val = tok.substr(eq + 1);
            if (key == "dev") {
                RDMA_CHECK(!val.empty(), "%s config: 'dev' must not be empty", backend);
                dev = std::string(val);
                have_dev = true;
            }
            else if (key == "gid") {
                const char *last = val.data() + val.size();
                const auto res = std::from_chars(val.data(), last, gid);
                RDMA_CHECK(res.ec == std::errc{} && res.ptr == last && gid >= 0,
                           "%s config: 'gid' must be a non-negative integer, got '%.*s'", backend,
                           static_cast<int>(val.size()), val.data());
                have_gid = true;
            }
        }
        if (sep == std::string::npos) {
            break;
        }
        pos = sep + 1;
    }
    RDMA_CHECK(have_dev && have_gid,
               "%s config requires both 'dev' and 'gid' (e.g. \"dev=rxe0;gid=1\"), got \"%s\"",
               backend, config.c_str());
    return BackendConfig{std::move(dev), gid};
}

} // namespace catalyst::transport::common
