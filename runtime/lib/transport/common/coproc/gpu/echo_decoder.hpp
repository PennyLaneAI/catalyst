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
#include <cstdint>

#include "DeviceCopy.hpp"

namespace catalyst::transport::coproc {

// Passthrough self-test: replies with the request's payload, cut to the reply's capacity.
__device__ inline std::uint32_t echo_decode(const std::uint8_t *in, std::uint32_t len,
                                            std::uint32_t /*fn_id*/, std::uint8_t *out,
                                            std::uint32_t cap, std::uint32_t lane,
                                            std::uint32_t lanes) {
    const std::uint32_t n = len < cap ? len : cap;
    device_copy(out, in, n, lane, lanes);
    return n;
}

} // namespace catalyst::transport::coproc
