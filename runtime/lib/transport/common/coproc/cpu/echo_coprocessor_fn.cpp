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
 * A CoprocessorFn that returns the request unchanged, for timing the transport by itself.
 */

#include <cstddef>
#include <cstdint>
#include <cstring>

/**
 * @brief Copy the request's data area straight back as the reply.
 *
 * @param in Request frame; its leading bytes are the data area.
 * @param in_len Bytes readable at @p in.
 * @param out Reply data area.
 * @param out_cap Bytes writable at @p out.
 * @param ctx Unused.
 * @return 0 on success, nonzero if either buffer is missing.
 */
extern "C" int echo_coprocessor(const void *in, std::size_t in_len, void *out, std::size_t out_cap,
                                void * /*ctx*/) {
    if ((in_len != 0 && in == nullptr) || (out_cap != 0 && out == nullptr)) {
        return 1;
    }
    const std::size_t n = in_len < out_cap ? in_len : out_cap;
    if (n != 0) {
        std::memcpy(out, in, n);
    }
    return 0;
}
