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

#include "HsbDevice.hpp"

#include <cstdio>
#include <cstring>
#include <ctime>
#include <exception>

#include "Error.hpp"
#include "PhysWindow.hpp"
#include "hsb_regs.h"

#include <unistd.h>

namespace catalyst::transport::hsb {

namespace {

constexpr int ECB_TIMEOUT_MS = 100;

std::uint64_t now_ns() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ull +
           static_cast<std::uint64_t>(ts.tv_nsec);
}

void put_be16(std::uint8_t *p, std::uint32_t v) {
    p[0] = static_cast<std::uint8_t>(v >> 8);
    p[1] = static_cast<std::uint8_t>(v);
}
void put_be32(std::uint8_t *p, std::uint32_t v) {
    p[0] = static_cast<std::uint8_t>(v >> 24);
    p[1] = static_cast<std::uint8_t>(v >> 16);
    p[2] = static_cast<std::uint8_t>(v >> 8);
    p[3] = static_cast<std::uint8_t>(v);
}
std::uint32_t get_be16(const std::uint8_t *p) { return (std::uint32_t{p[0]} << 8) | p[1]; }
std::uint32_t get_be32(const std::uint8_t *p) {
    return (std::uint32_t{p[0]} << 24) | (std::uint32_t{p[1]} << 16) | (std::uint32_t{p[2]} << 8) |
           p[3];
}

std::uint16_t ip_checksum(const std::uint8_t *hdr, std::size_t n) {
    std::uint32_t s = 0;
    for (std::size_t i = 0; i < n; i += 2) {
        s += get_be16(hdr + i);
    }
    while (s >> 16) {
        s = (s & 0xFFFF) + (s >> 16);
    }
    return static_cast<std::uint16_t>(~s);
}

// Ethernet / IPv4 / UDP / ECB request, padded to the 60-byte minimum (a MAC would add the FCS;
// HSB host ports carry frames without it). Returns its length.
std::size_t ecb_frame(std::uint8_t *f, std::uint8_t cmd, std::uint16_t seq, std::uint32_t addr,
                      const std::uint32_t *val) {
    const std::size_t ecb = 10 + (val ? 4 : 0); // cmd, flags, seq, 2 reserved, addr [, value]
    const std::size_t udp = 8 + ecb;
    const std::size_t ip = 20 + udp;
    const std::size_t n = 14 + ip;

    std::memset(f, 0, HWHS_ECB_BUF_BYTES);
    std::memcpy(f + 0, HsbDevice::HSB_MAC1, 6);
    std::memcpy(f + 6, HsbDevice::PS_MAC, 6);
    f[12] = 0x08;
    f[13] = 0x00;

    std::uint8_t *iph = f + 14;
    iph[0] = 0x45;
    put_be16(iph + 2, static_cast<std::uint32_t>(ip));
    iph[6] = 0x40; // DF
    iph[8] = 64;   // TTL
    iph[9] = 17;   // UDP
    put_be32(iph + 12, HsbDevice::PS_IP);
    put_be32(iph + 16, HSB_DEFAULT_IP + 1); // host port 1
    put_be16(iph + 10, ip_checksum(iph, 20));

    std::uint8_t *udph = iph + 20;
    put_be16(udph + 0, HsbDevice::PS_PORT);
    put_be16(udph + 2, HSB_ECB_UDP_PORT);
    put_be16(udph + 4, static_cast<std::uint32_t>(udp)); // checksum 0: not used over IPv4

    std::uint8_t *e = udph + 8;
    e[0] = cmd;
    e[1] = HSB_ECB_FLAG_ACK_REQUEST;
    put_be16(e + 2, seq);
    put_be32(e + 6, addr);
    if (val) {
        put_be32(e + 10, *val);
    }
    return n < 60 ? 60 : n;
}

} // namespace

void HsbDevice::io_barrier() {
#if defined(__aarch64__)
    __asm__ __volatile__("dsb sy" ::: "memory");
#else
    __sync_synchronize();
#endif
}

void HsbDevice::copy_to_dev64(volatile std::uint8_t *dst, const void *src) {
    std::uint64_t buf[HWHS_SLOT_BYTES / 8];
    std::memcpy(buf, src, sizeof(buf));
    volatile std::uint64_t *d = reinterpret_cast<volatile std::uint64_t *>(dst);
    for (std::size_t i = 0; i < HWHS_SLOT_BYTES / 8; ++i) {
        d[i] = buf[i];
    }
}

void HsbDevice::copy_from_dev64(void *dst, const volatile std::uint8_t *src, std::size_t bytes) {
    std::uint64_t buf[HWHS_SLOT_BYTES / 8];
    const volatile std::uint64_t *s = reinterpret_cast<const volatile std::uint64_t *>(src);
    const std::size_t words = (bytes + 7) / 8;
    for (std::size_t i = 0; i < words && i < HWHS_SLOT_BYTES / 8; ++i) {
        buf[i] = s[i];
    }
    std::memcpy(dst, buf, bytes < HWHS_SLOT_BYTES ? bytes : HWHS_SLOT_BYTES);
}

HsbDevice::HsbDevice() {
    try {
        regs_ = static_cast<volatile std::uint32_t *>(
            common::map_phys_window(HWHS_PHYS_REGS, HWHS_SIZE_REGS, "hsb"));
        slot_ = static_cast<volatile std::uint8_t *>(
            common::map_phys_window(HWHS_PHYS_SLOT, HWHS_SIZE_SLOT, "hsb"));
        reply_ = static_cast<volatile std::uint8_t *>(
            common::map_phys_window(HWHS_PHYS_REPLY, HWHS_SIZE_REPLY, "hsb"));
        trace_ = static_cast<volatile std::uint32_t *>(
            common::map_phys_window(HWHS_PHYS_TRACE, HWHS_SIZE_TRACE, "hsb"));
        pacer_ = static_cast<volatile std::uint8_t *>(
            common::map_phys_window(HWHS_PHYS_DEMO, HWHS_SIZE_DEMO, "hsb"));
        gpio_id_ = static_cast<volatile std::uint32_t *>(
            common::map_phys_window(HWHS_PHYS_GPIO_ID, HWHS_SIZE_GPIO, "hsb"));
        gpio_ctl_ = static_cast<volatile std::uint32_t *>(
            common::map_phys_window(HWHS_PHYS_GPIO_CTL, HWHS_SIZE_GPIO, "hsb"));

        // The GPIO is in the always-on pl0 domain and hwhs needs the hif clock: check it before
        // touching hwhs, so a dead clock reports instead of hanging the bus.
        TP_CHECK(gpio_ctl_[GPIO2_DATA / 4] & CTL_IN_CLK_LOCKED,
                 "hsb: platform clock wizard not locked (axi_gpio_ctl ch2 = 0x%08x): is this the "
                 "backline-hsb bitstream?",
                 gpio_ctl_[GPIO2_DATA / 4]);
        clk_hz_ = rd(HWHS_CLK_HZ);
        TP_CHECK(clk_hz_ != 0 && clk_hz_ == gpio_id_[GPIO2_DATA / 4],
                 "hsb: CLK_HZ %u does not match the platform (%u): wrong bitstream?", clk_hz_,
                 gpio_id_[GPIO2_DATA / 4]);
    } catch (...) {
        release(); // a throwing constructor gets no destructor call
        throw;
    }
}

HsbDevice::~HsbDevice() { release(); }

void HsbDevice::release() noexcept {
    common::unmap_phys_window(regs_, HWHS_SIZE_REGS);
    common::unmap_phys_window(slot_, HWHS_SIZE_SLOT);
    common::unmap_phys_window(reply_, HWHS_SIZE_REPLY);
    common::unmap_phys_window(trace_, HWHS_SIZE_TRACE);
    common::unmap_phys_window(pacer_, HWHS_SIZE_DEMO);
    common::unmap_phys_window(gpio_id_, HWHS_SIZE_GPIO);
    common::unmap_phys_window(gpio_ctl_, HWHS_SIZE_GPIO);
    regs_ = nullptr;
    slot_ = reply_ = pacer_ = nullptr;
    trace_ = gpio_id_ = gpio_ctl_ = nullptr;
}

void HsbDevice::ecb_xfer(std::uint8_t cmd, std::uint32_t addr, const std::uint32_t *wval,
                         std::uint32_t *rval) {
    std::uint8_t f[HWHS_ECB_BUF_BYTES];
    const std::uint16_t seq = ecb_seq_++;
    const std::size_t n = ecb_frame(f, cmd, seq, addr, wval);
    const char *what = wval ? "write" : "read";

    TP_CHECK(rd(HWHS_STATUS) & HWHS_ST_HSB_INIT_DONE, "hsb: HSB is not initialized (in reset?)");
    if (rd(HWHS_ECB_STATUS) & HWHS_ECB_ST_RX_VALID) { // a stale reply
        wr(HWHS_ECB_CTRL, HWHS_ECB_RELEASE);
    }
    for (std::size_t i = 0; i < (n + 3) / 4; ++i) {
        const std::uint32_t w = std::uint32_t{f[4 * i]} | (std::uint32_t{f[4 * i + 1]} << 8) |
                                (std::uint32_t{f[4 * i + 2]} << 16) |
                                (std::uint32_t{f[4 * i + 3]} << 24);
        wr(HWHS_ECB_TX_BUF + 4 * static_cast<std::uint32_t>(i), w);
    }
    wr(HWHS_ECB_TX_LEN, static_cast<std::uint32_t>(n));
    io_barrier();
    wr(HWHS_ECB_CTRL, HWHS_ECB_SEND);

    const std::uint64_t deadline = now_ns() + std::uint64_t{ECB_TIMEOUT_MS} * 1'000'000ull;
    for (;;) {
        const std::uint32_t st = rd(HWHS_ECB_STATUS);
        if (st & HWHS_ECB_ST_RX_VALID) {
            std::uint8_t r[HWHS_ECB_BUF_BYTES];
            const std::size_t len = HWHS_ECB_ST_RX_LEN(st);
            for (std::size_t i = 0; i < (len + 3) / 4 && i < HWHS_ECB_BUF_BYTES / 4; ++i) {
                const std::uint32_t w = rd(HWHS_ECB_RX_BUF + 4 * static_cast<std::uint32_t>(i));
                r[4 * i] = static_cast<std::uint8_t>(w);
                r[4 * i + 1] = static_cast<std::uint8_t>(w >> 8);
                r[4 * i + 2] = static_cast<std::uint8_t>(w >> 16);
                r[4 * i + 3] = static_cast<std::uint8_t>(w >> 24);
            }
            wr(HWHS_ECB_CTRL, HWHS_ECB_RELEASE);

            // reply: cmd', flags, seq, code, 0, addr, value, latched seq (after Eth/IPv4/UDP)
            const std::uint8_t *e = r + 42;
            if (len < 42 + 6 || get_be16(e + 2) != seq) {
                continue; // not ours: keep waiting
            }
            TP_CHECK(e[4] == HSB_ECB_RESP_SUCCESS, "hsb: ECB %s 0x%08x: response code 0x%02x", what,
                     addr, e[4]);
            if (rval) {
                TP_CHECK(len >= 42 + 14, "hsb: ECB read 0x%08x: short reply (%zu bytes)", addr,
                         len);
                *rval = get_be32(e + 10);
            }
            return;
        }
        TP_CHECK(now_ns() <= deadline,
                 "hsb: ECB %s 0x%08x: no reply in %d ms (ECB_STATUS 0x%08x, dropped %u, "
                 "overflow %u)",
                 what, addr, ECB_TIMEOUT_MS, st, rd(HWHS_ECB_DROP), rd(HWHS_ECB_OVF));
    }
}

std::uint32_t HsbDevice::ecb_read(std::uint32_t addr) {
    std::uint32_t v = 0;
    ecb_xfer(HSB_ECB_RD_DWORD, addr, nullptr, &v);
    return v;
}

void HsbDevice::ecb_write(std::uint32_t addr, std::uint32_t val) {
    ecb_xfer(HSB_ECB_WR_DWORD, addr, &val, nullptr);
}

void HsbDevice::hsb_reset(int timeout_ms) {
    gpio_ctl_[GPIO_DATA / 4] = CTL_OUT_HSB_RESET;
    ::usleep(1000);
    gpio_ctl_[GPIO_DATA / 4] = 0;

    const std::uint64_t deadline = now_ns() + static_cast<std::uint64_t>(timeout_ms) * 1'000'000ull;
    while (!(gpio_ctl_[GPIO2_DATA / 4] & CTL_IN_HSB_INIT_DONE)) {
        TP_CHECK(now_ns() <= deadline, "hsb: HSB init_done did not come back (ctl status 0x%08x)",
                 gpio_ctl_[GPIO2_DATA / 4]);
        ::usleep(100);
    }
}

void HsbDevice::hsb_configure(const DataPlane &dp) {
    const std::uint32_t hif = HSB_HIF(0);
    const std::uint32_t vp = HSB_VP(0);

    TP_CHECK(dp.ring_pitch != 0 && dp.ring_pitch % HSB_PAGE_SIZE == 0 && dp.ring_slots != 0 &&
                 dp.ring_slots <= 4096,
             "hsb: bad ring (pitch %u, slots %u)", dp.ring_pitch, dp.ring_slots);
    const std::uint32_t version = ecb_read(HSB_IP_VERSION) & 0xFFFF;
    TP_CHECK(version >= HSB_IP_VERSION_MIN, "hsb: HSB IP version 0x%04x < 0x%04x", version,
             HSB_IP_VERSION_MIN);

    // The VP stays off while its registers change.
    ecb_write(hif + HSB_DP_VP_MASK, ecb_read(hif + HSB_DP_VP_MASK) & ~1u);

    ecb_write(hif + HSB_DP_PACKET_SIZE, 8); // 8 x 128 B: one packet per message
    ecb_write(hif + HSB_DP_PACKET_UDP_PORT, HSB_DATA_SOURCE_UDP_PORT);
    ecb_write(vp + HSB_DP_BUFFER_LENGTH, dp.frame_bytes);
    ecb_write(vp + HSB_DP_HOST_MAC_LOW, (std::uint32_t{dp.host_mac[2]} << 24) |
                                            (std::uint32_t{dp.host_mac[3]} << 16) |
                                            (std::uint32_t{dp.host_mac[4]} << 8) | dp.host_mac[5]);
    ecb_write(vp + HSB_DP_HOST_MAC_HIGH, (std::uint32_t{dp.host_mac[0]} << 8) | dp.host_mac[1]);
    ecb_write(vp + HSB_DP_HOST_IP, dp.host_ip);
    ecb_write(vp + HSB_DP_HOST_UDP_PORT, HSB_ROCE_UDP_PORT | HSB_MD_DISABLE | HSB_IMM_DISABLE);
    ecb_write(vp + HSB_DP_VLAN, 0);
    ecb_write(vp + HSB_DP_QP, dp.qpn & 0xFFFFFF);
    ecb_write(vp + HSB_DP_RKEY, dp.rkey);
    ecb_write(vp + HSB_DP_PAGE_LSB, static_cast<std::uint32_t>(dp.ring_iova));
    ecb_write(vp + HSB_DP_PAGE_MSB, static_cast<std::uint32_t>(dp.ring_iova >> 32));
    ecb_write(vp + HSB_DP_PAGE_INC, dp.ring_pitch / HSB_PAGE_SIZE);
    ecb_write(vp + HSB_DP_MAX_BUFF, dp.ring_slots - 1);

    // Sensor-TX 0 holds its output until it is 3/4 full unless the thresholds are 0; the 0 -> 1
    // edge of ctrl bit 2 latches them.
    ecb_write(HSB_STX0 + HSB_STX_AFULL_THRESH, 0);
    ecb_write(HSB_STX0 + HSB_STX_AEMPTY_THRESH, 0);
    ecb_write(HSB_STX0 + HSB_STX_CTRL, 0x1);
    ecb_write(HSB_STX0 + HSB_STX_CTRL, 0x5);

    ecb_write(hif + HSB_DP_VP_MASK, ecb_read(hif + HSB_DP_VP_MASK) | 1u);
}

void HsbDevice::hsb_stop() noexcept {
    try {
        const std::uint32_t mask = HSB_HIF(0) + HSB_DP_VP_MASK;
        ecb_write(mask, ecb_read(mask) & ~1u);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "[hsb] %s\n", e.what());
    }
}

} // namespace catalyst::transport::hsb
