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
#include <string>
#include <string_view>
#include <utility>

#include "ConfigParser.hpp"
#include "Error.hpp"

namespace catalyst::transport::common {

// Construction parameters shared by the verbs-based backends, parsed from a backend factory
// `config` string ("key=value;..."). Connection parameters (peer/oob_port) are not here —
// they arrive later via connect(). Both keys are required and have no defaults, e.g.
// "dev=rxe0;gid=1".
struct BackendConfig {
    std::string dev; // RDMA device name
    int gid;         // GID index on that device's port
};

/**
 * @brief Parse the `dev` and `gid` keys out of a backend `config` string.
 *
 * Unknown keys are ignored, so a backend may carry extra keys of its own and pull them out
 * with configparser::parse_optional_index(). Both keys here are mandatory: an absent, empty
 * or malformed value throws.
 *
 * @param config The config string, e.g. "dev=rxe0;gid=1".
 * @return The parsed device name and GID index.
 */
inline BackendConfig parse_backend_config(const std::string &config) {
    std::string dev;
    int gid = 0;
    bool have_dev = false, have_gid = false;
    configparser::for_each_kv(config, [&](std::string_view key, std::string_view val) {
        if (key == "dev") {
            TP_CHECK(!val.empty(), "config: 'dev' must not be empty");
            dev = std::string(val);
            have_dev = true;
        } else if (key == "gid") {
            gid = configparser::parse_index(val, "gid");
            have_gid = true;
        }
    });
    TP_CHECK(have_dev && have_gid,
             "config requires both 'dev' and 'gid' (e.g. \"dev=rxe0;gid=1\"), got \"%s\"",
             config.c_str());
    return BackendConfig{std::move(dev), gid};
}

} // namespace catalyst::transport::common
