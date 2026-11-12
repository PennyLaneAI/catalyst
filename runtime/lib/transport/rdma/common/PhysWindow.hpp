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
 * Maps a window of the board's physical address space, such as an FPGA engine's registers: through
 * the UIO device whose first map starts there, else through /dev/mem.
 */

#pragma once
#include <cstddef>
#include <cstdint>

namespace catalyst::transport::common {

/// Map `len` bytes at physical address `phys`. `who` starts every error message.
void *map_phys_window(std::uint64_t phys, std::size_t len, const char *who);

/// Unmap a window that map_phys_window returned; null is ignored.
void unmap_phys_window(volatile void *p, std::size_t len);

} // namespace catalyst::transport::common
