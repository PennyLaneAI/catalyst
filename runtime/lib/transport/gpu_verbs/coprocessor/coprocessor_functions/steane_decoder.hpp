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
namespace catalyst::transport::gpu_verbs {

constexpr int STEANE_CHECKS = 3;

/**
 * @brief Syndrome to error qubit index for the [[7,1,3]] Steane code; -1 is no
 * error.
 *
 * Indexed by the 3 checks packed with check 0 as the most significant bit.
 * For compatibility with existing Steane decoder.
 */
__device__ constexpr std::int64_t STEANE_SYNDROME_TO_QUBIT[1 << STEANE_CHECKS] = {-1, 6, 4, 5,
                                                                                  0,  3, 1, 2};

/**
 * @brief Decode one Steane syndrome to an error qubit index.
 *
 * @param syndrome Ring word holding one byte per check, byte `i` carrying check
 *                 `i` in its low bit (a `memref<?xi1>` lowers to one byte per
 *                 element, so the checks arrive unpacked).
 * @return The error qubit index, or -1 for no error.
 */
__device__ inline std::int64_t steane_decode(std::uint64_t syndrome)
{
    std::uint32_t packed = 0;
    for (int i = 0; i < STEANE_CHECKS; ++i) {
        packed = (packed << 1U) | static_cast<std::uint32_t>((syndrome >> (8 * i)) & 1U);
    }
    return STEANE_SYNDROME_TO_QUBIT[packed];
}

} // namespace catalyst::transport::gpu_verbs
