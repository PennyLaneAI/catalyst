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

namespace catalyst::transport::common {

/**
 * @struct QpInfo
 * @brief Connection metadata exchanged out-of-band to connect remote Queue
 * Pairs.
 */
struct QpInfo {
    std::uint32_t qpn;    // Unique Queue Pair Number hardware identifier.
    std::uint32_t psn;    // Starting Packet Sequence Number for flow validation.
    std::uint8_t gid[16]; // 128-bit Global Identifier routing address.
};

/**
 * @struct HandshakeMsg
 * @brief Message exchanged once over the OOB TCP socket, after the MR exists,
 * so QP identity and MR handle are swapped together.
 */
struct HandshakeMsg {
    QpInfo fwd;             // forward QP (controller -> coprocessor)
    QpInfo bwd;             // backward QP (coprocessor -> controller)
    std::uint64_t mr_vaddr; // where the peer writes into us
    std::uint32_t mr_rkey;
    std::uint32_t mtu_enum; // ibv_mtu enum
};

} // namespace catalyst::transport::common
