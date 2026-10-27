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

#include <hip/hip_runtime.h> // int4

namespace catalyst::transport::coproc {

/// Copy `n` bytes on the device, shared among `lanes` threads that all make the same call, this
/// one being `lane`. Each copies its 16 B chunks where both ends are 16 B aligned (8 B chunks
/// where they are 8 B aligned, as a CUQR payload at byte 24 is), then its share of the remaining
/// bytes. A group of one copies everything itself.
///
/// Sharing matters because each load from the uncached request ring takes about 0.5 us, and one
/// thread waits for its loads one after another; a group issues them together and waits about
/// once. Measured on an MI210 behind a ConnectX-7 (NIC loopback, gpu_verbs echo, round-trip
/// p50), one thread against 32: 16 B 17.4 vs 6.3 us, 256 B 18.8 vs 7.7 us, 1 KiB 49 vs 8.9 us.
__device__ inline void device_copy(std::uint8_t *dst, const std::uint8_t *src, std::uint32_t n,
                                   std::uint32_t lane, std::uint32_t lanes) {
    if (lanes == 1 && n <= 16) {
        __builtin_memcpy(dst, src, n); // a short copy between locals stays in registers
        return;
    }
    std::uint32_t done = 0;
    const auto align =
        reinterpret_cast<std::uintptr_t>(dst) | reinterpret_cast<std::uintptr_t>(src);
    if (n > 16 && (align & 15U) == 0) {
        done = n & ~15U;
        for (std::uint32_t i = lane * 16; i < done; i += lanes * 16) {
            *reinterpret_cast<int4 *>(dst + i) = *reinterpret_cast<const int4 *>(src + i);
        }
    } else if (n > 16 && (align & 7U) == 0) {
        done = n & ~7U;
        for (std::uint32_t i = lane * 8; i < done; i += lanes * 8) {
            *reinterpret_cast<std::uint64_t *>(dst + i) =
                *reinterpret_cast<const std::uint64_t *>(src + i);
        }
    }
    for (std::uint32_t i = done + lane; i < n; i += lanes) {
        dst[i] = src[i];
    }
}

} // namespace catalyst::transport::coproc
