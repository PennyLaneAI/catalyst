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

/**
 * @brief A hard-coded [[7,1,3]] Steane-code decode, exposed as a CoprocessorFn
 * (see Transport.hpp) — the general "run this on the coprocessor" contract that
 * supersedes the old decoder-plugin ABI.
 *
 * @note The FTQC (Fault-Tolerant Quantum Computing) compilation pipeline
 * dispatches either X-check or Z-check syndromes independently per call, so each
 * call carries a single 3-bit check. This may be unified in future iterations.
 *
 * @param in Pointer to the inbound syndrome measurements.
 * @param in_len Length of the inbound syndrome, in bytes.
 * @param out Pointer to the outbound correction buffer.
 * @param out_cap Capacity of the outbound buffer, in bytes.
 * @param ctx Opaque context (unused).
 * @return Number of bytes written to @p out.
 */
extern "C" std::size_t steane_coprocessor(const void *in, std::size_t in_len, void *out,
                                          std::size_t out_cap, void * /*ctx*/)
{
    std::uint64_t syndrome = 0;
    std::memcpy(&syndrome, in, in_len < sizeof(syndrome) ? in_len : sizeof(syndrome));
    // One 3-bit check (X or Z) per call; the nonzero index selects the single
    // corrected qubit (0 => no error).
    const std::uint32_t check = syndrome & 0x7u;
    const std::uint64_t correction = static_cast<std::uint64_t>(check ? (1u << (check - 1)) : 0u);
    const std::size_t nb = out_cap < sizeof(correction) ? out_cap : sizeof(correction);
    std::memset(out, 0, out_cap);
    std::memcpy(out, &correction, nb);
    return nb;
}
