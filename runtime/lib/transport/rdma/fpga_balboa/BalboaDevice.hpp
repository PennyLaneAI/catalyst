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
 * The VPK120's hwhs-balboa controller and the BALBOA RoCE v2 stack behind it, from the A72. A C++
 * port of backline-roce's sw/lib/hwhs/hwhs.c.
 *
 * Every window is mapped through generic-uio (the device whose map0 starts at the window's
 * physical address) or, failing that, /dev/mem; either way the process needs root. BALBOA takes
 * its address and the QP through hwhs registers (NET_CTRL, QP_CTX, CONN_CTX).
 */

#pragma once
#include <cstddef>
#include <cstdint>

#include "hwhs_regs.h"

namespace catalyst::transport::balboa {

/// One RC connection between BALBOA and a coprocessor.
struct Connection {
    std::uint32_t remote_qpn; // the coprocessor's QP
    std::uint32_t psn;        // the PSN of both directions
    std::uint32_t rkey;       // of the coprocessor's ring
    std::uint64_t ring_iova;  // its slot 0
    std::uint32_t remote_ip;  // its RoCE IPv4, host byte order
};

class BalboaDevice {
  public:
    // BALBOA's address after a reset (lynx_pkg.sv), the same as HSB host port 0's in the HSB build
    static constexpr std::uint32_t FPGA_IP = 0xC0A80002u; // 192.168.0.2
    static constexpr std::uint8_t FPGA_MAC[6] = {0x02, 0x93, 0xc6, 0x48, 0x4d, 0x8e};
    // Coyote's rdma_flow reads the flow-control and retransmission pointers of QP 0 whatever the
    // QP, so QP 0 is the one it tracks correctly.
    static constexpr std::uint32_t FPGA_QPN = 0;
    // UDP source port of BALBOA's packets; the destination is always 4791.
    static constexpr std::uint16_t UDP_SPORT = 49152;

    /// Map every window and check the platform: throws if one is missing or the clock is off.
    BalboaDevice();
    ~BalboaDevice();
    BalboaDevice(const BalboaDevice &) = delete;
    BalboaDevice &operator=(const BalboaDevice &) = delete;

    std::uint32_t rd(std::uint32_t off) const { return regs_[off / 4]; }
    void wr(std::uint32_t off, std::uint32_t v) { regs_[off / 4] = v; }
    std::uint32_t clk_hz() const { return clk_hz_; }
    std::uint32_t build_id() const { return rd(HWHS_BUILD_ID); }

    volatile std::uint8_t *slot(std::uint32_t idx) {
        return slot_ + std::size_t{idx} * HWHS_SLOT_BYTES;
    }
    volatile std::uint8_t *reply(std::uint32_t idx) {
        return reply_ + std::size_t{idx} * HWHS_SLOT_BYTES;
    }
    volatile std::uint8_t *demo() { return demo_; }
    /// RTT trace entry `i`, in net clock cycles; 0xFFFF is saturated.
    std::uint16_t trace(std::uint32_t i) const {
        const std::uint32_t w = trace_[i / 2];
        return static_cast<std::uint16_t>((i & 1) ? (w >> 16) : w);
    }

    /// Hold BALBOA in reset for a moment (QP state, PSNs, ARP table) and wait for it to come back.
    void stack_reset(int timeout_ms);
    /// Give BALBOA its IPv4 (host byte order) and MAC.
    void set_addr(std::uint32_t ip, const std::uint8_t mac[6]);
    /// Ask BALBOA's ARP server to resolve `ip`; false if no ARP reply came in time.
    bool arp(std::uint32_t ip, int timeout_ms);
    /// Write the QP and connection contexts of `c` on FPGA_QPN and push them to BALBOA.
    void connect(const Connection &c);

    /// 64 B to or from a device window, as eight 64-bit accesses (device memory takes no
    /// unaligned or wider accesses, which memcpy may use).
    static void copy_to_dev64(volatile std::uint8_t *dst, const void *src);
    static void copy_from_dev64(void *dst, const volatile std::uint8_t *src, std::size_t bytes);
    /// Order the stores above before the stores below, all the way to the device.
    static void io_barrier();

  private:
    void net_push(std::uint32_t bits);
    void release() noexcept;

    volatile std::uint32_t *regs_ = nullptr;     // hwhs registers
    volatile std::uint8_t *slot_ = nullptr;      // slot RAM: request payloads
    volatile std::uint8_t *reply_ = nullptr;     // reply RAM
    volatile std::uint32_t *trace_ = nullptr;    // RTT trace, two 16-bit entries per word
    volatile std::uint8_t *demo_ = nullptr;      // transport_demo table
    volatile std::uint32_t *gpio_id_ = nullptr;  // ch1 BUILD_ID, ch2 net clock Hz
    volatile std::uint32_t *gpio_ctl_ = nullptr; // ch1 [0] BALBOA reset, ch2 status
    std::uint32_t clk_hz_ = 0;
};

} // namespace catalyst::transport::balboa
