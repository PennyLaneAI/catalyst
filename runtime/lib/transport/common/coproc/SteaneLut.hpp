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

// Shared LUT (look-up table) for the [[7,1,3]] Steane code, used by both the CPU
// CoprocessorFn (steane_decoder_fn.cpp) and the GPU __device__ decode
// (steane_decoder.hpp). Kept here as a single source of truth so the two decoders
// can't drift.

#pragma once
#include <cstdint>

namespace catalyst::transport::coproc {

/// Number of syndrome check bits per shot for the [[7,1,3]] Steane code.
constexpr int STEANE_CHECKS = 3;

/// Syndrome → error qubit index (0..6); -1 marks "no error".
/// Indexed by the 3 checks packed with check 0 as the most significant bit.
constexpr std::int64_t STEANE_SYNDROME_TO_QUBIT[1 << STEANE_CHECKS] = {-1, 6, 4, 5, 0, 3, 1, 2};

/// STEANE_SYNDROME_TO_QUBIT packed into a single 32-bit word, 4 bits per entry;
/// 0xF encodes the -1 (no error) sentinel. Compact form for constant-memory access
/// on the GPU.
constexpr std::uint32_t pack_steane_table() {
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

} // namespace catalyst::transport::coproc
