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

__device__ inline std::uint64_t steane_decode(std::uint64_t syndrome)
{
    // One 3-bit check (X or Z) per call; the nonzero index selects the single
    // corrected qubit (0 => no error).
    const std::uint32_t check = syndrome & 0x7u;
    return static_cast<std::uint64_t>(check ? (1u << (check - 1)) : 0u);
}

} // namespace catalyst::transport::gpu_verbs
