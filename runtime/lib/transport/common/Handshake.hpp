#pragma once
#include <cstdint>

namespace rdma::devices::common {

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

} // namespace rdma::devices::common
