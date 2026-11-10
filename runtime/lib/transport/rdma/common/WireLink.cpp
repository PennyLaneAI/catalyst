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

#include "WireLink.hpp"

#include "Error.hpp"
#include "RcLink.hpp"

namespace catalyst::transport::common {

std::unique_ptr<WireLink> make_coprocessor_link(const std::string &config, const char *backend,
                                                QpShape fwd_shape) {
    const BackendConfig cfg = parse_backend_config(config);
    switch (wire_from_config(config)) {
    case Wire::RC:
        return std::make_unique<RcLink>(cfg.dev, cfg.gid, backend, RcLink::Side::Coprocessor,
                                        frame::geometry_from_config(config), fwd_shape);
    }
    TP_FAIL("%s: unknown wire", backend);
}

} // namespace catalyst::transport::common
