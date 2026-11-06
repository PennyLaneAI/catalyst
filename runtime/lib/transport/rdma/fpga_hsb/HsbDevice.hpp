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
 * The VPK120's hwhs-hsb controller and the HSB behind it, from the A72. A C++ port of
 * backline-hsb's sw/lib/hwhs/hwhs.c.
 *
 * Every window is mapped through generic-uio (the device whose map0 starts at the window's
 * physical address) or, failing that, /dev/mem; either way the process needs root. Registers are
 * plain loads and stores: there is no kernel module and no ioctl. HSB's own registers are reached
 * through ECB packets that hwhs's mailbox puts on HSB host port 1, so HSB itself is unmodified.
 */

#pragma once
#include <cstddef>
#include <cstdint>

#include "hwhs_regs.h"

namespace catalyst::transport::hsb {

/// Where HSB writes the requests: the coprocessor's ring (HsbWire.hpp's HostInfo).
struct DataPlane {
    std::uint32_t qpn;
    std::uint32_t rkey;
    std::uint64_t ring_iova;
    std::uint32_t ring_pitch; // a multiple of 128
    std::uint32_t ring_slots;
    std::uint32_t host_ip; // host byte order
    std::uint8_t host_mac[6];
    std::uint32_t frame_bytes; // per request
};

class HsbDevice {
  public:
    // HSB host port addresses, as wired in backline-hsb's hw/rtl/top/backline_hsb_top.sv
    static constexpr std::uint8_t HSB_MAC0[6] = {0x02, 0x93, 0xc6, 0x48, 0x4d, 0x8e}; // 100G
    static constexpr std::uint8_t HSB_MAC1[6] = {0x02, 0x93, 0xc6, 0x48, 0x4d, 0x8f}; // ECB
    static constexpr std::uint8_t PS_MAC[6] = {0x02, 0x93, 0xc6, 0x48, 0x4d, 0x90};   // the A72
    static constexpr std::uint32_t PS_IP = 0xC0A800FEu;                               // .0.254
    static constexpr std::uint16_t PS_PORT = 12345;

    /// Map every window and check the platform: throws if one is missing or the clock is off.
    HsbDevice();
    ~HsbDevice();
    HsbDevice(const HsbDevice &) = delete;
    HsbDevice &operator=(const HsbDevice &) = delete;

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
    /// RTT trace entry `i`, in hif cycles; 0xFFFF is saturated.
    std::uint16_t trace(std::uint32_t i) const {
        const std::uint32_t w = trace_[i / 2];
        return static_cast<std::uint16_t>((i & 1) ? (w >> 16) : w);
    }

    /// HSB registers over ECB; both throw on a timeout or an ECB error code.
    std::uint32_t ecb_read(std::uint32_t addr);
    void ecb_write(std::uint32_t addr, std::uint32_t val);

    /// Pulse HSB's reset (which clears its slot pointer and PSN) and wait for init_done.
    void hsb_reset(int timeout_ms);
    /// Program HSB data plane 0 / sensor 0 / sensor-TX 0 to write requests into `dp`.
    void hsb_configure(const DataPlane &dp);
    /// Stop HSB sending (VP mask off). Best effort, for teardown.
    void hsb_stop() noexcept;

    /// 64 B to or from a device window, as eight 64-bit accesses (device memory takes no
    /// unaligned or wider accesses, which memcpy may use).
    static void copy_to_dev64(volatile std::uint8_t *dst, const void *src);
    static void copy_from_dev64(void *dst, const volatile std::uint8_t *src, std::size_t bytes);
    /// Order the stores above before the stores below, all the way to the device.
    static void io_barrier();

  private:
    void ecb_xfer(std::uint8_t cmd, std::uint32_t addr, const std::uint32_t *wval,
                  std::uint32_t *rval);
    void release() noexcept;

    volatile std::uint32_t *regs_ = nullptr;     // hwhs registers + ECB mailbox
    volatile std::uint8_t *slot_ = nullptr;      // slot RAM: request payloads
    volatile std::uint8_t *reply_ = nullptr;     // reply RAM
    volatile std::uint32_t *trace_ = nullptr;    // RTT trace, two 16-bit entries per word
    volatile std::uint8_t *demo_ = nullptr;      // transport_demo table
    volatile std::uint32_t *gpio_id_ = nullptr;  // ch1 BUILD_ID, ch2 hif Hz
    volatile std::uint32_t *gpio_ctl_ = nullptr; // ch1 [0] HSB reset, ch2 status
    std::uint32_t clk_hz_ = 0;
    std::uint16_t ecb_seq_ = 1;
};

} // namespace catalyst::transport::hsb
