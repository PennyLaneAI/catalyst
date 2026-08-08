// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstddef>
#include <cstdint>

namespace catalyst::transport::local_copy {

// Header stored at the front of the coprocessor's local request region. The request payload bytes
// follow immediately after this header.
struct LocalRequestHeader {
    std::uint64_t bytes = 0;
    std::uint32_t decoder_id = 0;
};

// Header stored at the front of the controller's local reply region. The reply payload bytes
// follow immediately after this header.
struct LocalReplyHeader {
    std::uint64_t bytes = 0;
};

inline constexpr std::size_t kLocalRequestHeaderBytes = sizeof(LocalRequestHeader);
inline constexpr std::size_t kLocalReplyHeaderBytes = sizeof(LocalReplyHeader);

} // namespace catalyst::transport::local_copy
