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
 * @file steane_plugin.cpp
 * Defines a decoder for the [[7,1,3]] Steane code.
 */

#include <cstddef>
#include <cstdint>
#include <cstring>

/**
 * @brief A hard-coded decoder for the [[7,1,3]] Steane code with a static
 * Tanner graph mapping.
 *
 * @note The FTQC (Fault-Tolerant Quantum Computing) compilation
 * pipeline processes and dispatches either X-check or Z-check syndromes
 *       independently per execution call, so each call carries a single 3-bit
 *       check. This behavior may be unified in future pipeline iterations.
 *
 * @param ctx Pointer to a common::Context instance.
 * @param in Pointer to inbound syndrome measurements
 * @param in_len The length of the inbound syndrome/message.
 * @param out Pointer to the index of the detected error qubit.
 * @param out_len The length of the outbound message.
 * side.
 */
extern "C" void decode(void * /*ctx*/, const void *in, std::size_t in_len, void *out,
                       std::size_t out_len)
{
    std::uint64_t syndrome = 0;
    std::memcpy(&syndrome, in, in_len < sizeof(syndrome) ? in_len : sizeof(syndrome));
    // One 3-bit check (X or Z) per call; the nonzero index selects the single
    // corrected qubit (0 => no error).
    const std::uint32_t check = syndrome & 0x7u;
    const std::uint64_t correction = static_cast<std::uint64_t>(check ? (1u << (check - 1)) : 0u);
    std::memset(out, 0, out_len);
    std::memcpy(out, &correction, out_len < sizeof(correction) ? out_len : sizeof(correction));
}
