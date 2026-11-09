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
#include "SteaneLut.hpp"

namespace catalyst::transport::coproc {

/**
 * @brief Decode one Steane syndrome to an error qubit index.
 *
 * @param in The syndrome, one byte per check, byte `i` carrying check `i` in its low bit (a
 *           `memref<?xi1>` lowers to one byte per element, so the checks arrive unpacked).
 * @param len Bytes in `in`; at least STEANE_CHECKS.
 * @param out Receives the error qubit index as an `int64_t`, or -1 for no error.
 * @param cap Bytes `out` holds; at least 8.
 * @param lane This thread's index in the group making the call.
 * @param lanes The group's size; each thread decodes, and they share the write.
 * @return 8, or 0 if `out` is too short. A syndrome shorter than STEANE_CHECKS decodes to -1.
 */
__device__ inline std::uint32_t steane_decode(const std::uint8_t *in, std::uint32_t len,
                                              std::uint32_t /*fn_id*/, std::uint8_t *out,
                                              std::uint32_t cap, std::uint32_t lane,
                                              std::uint32_t lanes) {
    if (cap < sizeof(std::int64_t)) {
        return 0;
    }
    std::uint32_t qubit = 0xFU; // no error
    if (len >= STEANE_CHECKS) {
        std::uint32_t packed = 0;
        for (int i = 0; i < STEANE_CHECKS; ++i) {
            packed = (packed << 1U) | static_cast<std::uint32_t>(in[i] & 1U);
        }
        qubit = (STEANE_TABLE_PACKED >> (packed * 4U)) & 0xFU;
    }
    const std::int64_t correction = (qubit == 0xFU) ? -1 : static_cast<std::int64_t>(qubit);
    device_copy(out, reinterpret_cast<const std::uint8_t *>(&correction), sizeof(correction), lane,
                lanes);
    return sizeof(correction);
}

} // namespace catalyst::transport::coproc
