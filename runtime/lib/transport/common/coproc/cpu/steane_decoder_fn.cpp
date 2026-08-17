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
/**
 * @file
 * A reference CoprocessorFn implementing a [[7,1,3]] Steane-code decode.
 */

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "SteaneLut.hpp"

// The shared LUT (STEANE_CHECKS, STEANE_SYNDROME_TO_QUBIT) lives in common/coproc/SteaneLut.hpp
// so the CPU and GPU decoders can't disagree on the table.
using catalyst::transport::coproc::STEANE_CHECKS;
using catalyst::transport::coproc::STEANE_SYNDROME_TO_QUBIT;

/**
 * @brief A hard-coded [[7,1,3]] Steane-code decode exposed as a CoprocessorFn
 *
 * @param in Syndrome measurements, one byte per check, byte `i` holding check
 *           `i` in its low bit (a `memref<?xi1>` lowers to one byte per element).
 * @param in_len Length of the inbound syndrome, in bytes.
 * @param out Buffer for the outbound error qubit index.
 * @param out_cap Capacity of the outbound buffer, in bytes.
 * @param ctx Opaque context (unused).
 * @return Bytes written to @p out, or 0 if the buffers are too small, which the
 *         caller reports as a failed round.
 *
 * @note `in_len` is the payload capacity rather than the syndrome length, which
 * the wire does not carry; a decoder therefore has to know its own code's check
 * count.
 */
// `in` is the request frame; its leading bytes are the syndrome. The frame's
// decoder_id (a uint32 at byte offset 8) is not read: the [[7,1,3]] Steane code has
// Hx == Hz, so one table serves both the X and Z checks. A code whose matrices differ
// would read that field and switch on it here.
extern "C" std::size_t steane_coprocessor(const void *in, std::size_t in_len, void *out,
                                          std::size_t out_cap, void * /*ctx*/) {
    if (in == nullptr || out == nullptr || in_len < STEANE_CHECKS ||
        out_cap < sizeof(std::int64_t)) {
        return 0;
    }
    const auto *checks = static_cast<const std::uint8_t *>(in);
    std::uint32_t syndrome = 0;
    for (std::size_t i = 0; i < STEANE_CHECKS; ++i) {
        syndrome = (syndrome << 1U) | (checks[i] & 1U);
    }
    const std::int64_t err_idx = STEANE_SYNDROME_TO_QUBIT[syndrome];
    std::memcpy(out, &err_idx, sizeof(err_idx));
    return sizeof(err_idx);
}
