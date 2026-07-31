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
constexpr std::int64_t STEANE_SYNDROME_TO_QUBIT[1 << STEANE_CHECKS] = {-1, 6, 4, 5, 0, 3, 1, 2};

/**
 * @brief STEANE_SYNDROME_TO_QUBIT packed into a single 32-bit word, 4 bits per
 * entry; 0xF encodes the -1 (no error) sentinel. Packed for reduced memory access.
 */
constexpr std::uint32_t pack_steane_table()
{
    std::uint32_t packed = 0;
    for (int i = 0; i < (1 << STEANE_CHECKS); ++i) {
        const std::int64_t qubit = STEANE_SYNDROME_TO_QUBIT[i];
        packed |= (qubit < 0 ? 0xFU : static_cast<std::uint32_t>(qubit)) << (4 * i);
    }
    return packed;
}
constexpr std::uint32_t STEANE_TABLE_PACKED = pack_steane_table();
static_assert(STEANE_TABLE_PACKED == 0x2130546FU,
              "packed table must match STEANE_SYNDROME_TO_QUBIT");

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
    const std::uint32_t qubit = (STEANE_TABLE_PACKED >> (packed * 4U)) & 0xFU;
    return (qubit == 0xFU) ? -1 : static_cast<std::int64_t>(qubit);
}

} // namespace catalyst::transport::gpu_verbs
