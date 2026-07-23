#pragma once
#include <cstddef>
#include <cstdint>

namespace rdma::devices::common {

// Ring size, identical on both sides; power of two so index is a mask.
inline constexpr std::size_t K_RING_SLOTS = 256;

// Selective-signalling stride (flow control on the pipelined send paths).
inline constexpr std::uint32_t SIGNAL_EVERY = 64;

// Validation salt: packed into Payload.value low bits in the RTT self-test.
inline constexpr std::uint32_t SALT = 0xC0DE1515u;

// Demo/loopback payload the controller ships each shot (stand-in for a real
// measurement outcome). Its (echo) decode is what the self-test checks.
inline constexpr std::uint64_t DEMO_SYNDROME = 0x0123456789ABCDEFull;

// 16 B wire frame.
// Note: this application payload size is unrelated to the network MTU (the QP's
// max packet size, negotiated at RTR). The size here is far below any MTU, so
// each transfer is always a single packet.
#pragma pack(push, 1)
struct Payload {
    std::uint64_t value;
    std::uint32_t seq_num;
    std::uint32_t pad;
};
#pragma pack(pop)
static_assert(sizeof(Payload) == 16, "Payload must be exactly 16 B");

// Some controller DMA engine requires 64-B aligned. Rings are
// 64-B-strided slots; only the leading Payload (16 B) is transferred per slot.
struct alignas(64) PayloadSlot {
    Payload p;
    std::uint8_t pad_[48];
};
static_assert(sizeof(PayloadSlot) == 64, "PayloadSlot must be exactly 64 B");
static_assert(alignof(PayloadSlot) == 64, "PayloadSlot must be 64-B aligned");

// Receive ring is K_RING_SLOTS PayloadSlots; the peer writes slot[cursor %
// K_RING_SLOTS]. K_RING_SLOTS must be a power of two.
inline constexpr std::size_t REGION_BYTES = K_RING_SLOTS * sizeof(PayloadSlot);

} // namespace rdma::devices::common
