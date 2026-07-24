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
#include <cstdlib>
#include <string>
#include <string_view>

namespace catalyst::transport::cpu_verbs {

// Construction parameters parsed from a backend factory `config` string
// ("key=value;..."). Connection parameters (peer/oob_port) are not here — they
// arrive later via connect(). Recognised keys: `dev`, `gid`.
struct CpuConfig {
    std::string dev = "rxe0";
    int gid = 1;
};

inline CpuConfig parse_cpu_config(const std::string &config)
{
    CpuConfig cfg;
    for (std::size_t pos = 0; pos < config.size();) {
        const std::size_t sep = config.find(';', pos);
        const std::size_t end = (sep == std::string::npos) ? config.size() : sep;
        const std::string_view tok(config.data() + pos, end - pos);
        if (const std::size_t eq = tok.find('='); eq != std::string_view::npos) {
            const std::string_view key = tok.substr(0, eq);
            const std::string val(tok.substr(eq + 1));
            if (key == "dev")
                cfg.dev = val;
            else if (key == "gid")
                cfg.gid = std::atoi(val.c_str());
        }
        if (sep == std::string::npos)
            break;
        pos = sep + 1;
    }
    return cfg;
}

} // namespace catalyst::transport::cpu_verbs
