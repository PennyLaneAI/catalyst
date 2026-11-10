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

#include "BalboaDevice.hpp"

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <sys/mman.h>

#include "Error.hpp"

#include <fcntl.h>
#include <unistd.h>

namespace catalyst::transport::balboa {

namespace {

// NET_CTRL pushes are taken in a few cycles; the timeout only catches a stack held in reset.
constexpr int NET_PUSH_TIMEOUT_MS = 100;

std::uint64_t now_ns() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ull +
           static_cast<std::uint64_t>(ts.tv_nsec);
}

bool read_sysfs_u64(const std::string &path, std::uint64_t *v) {
    std::FILE *f = std::fopen(path.c_str(), "r");
    if (f == nullptr) {
        return false;
    }
    const bool ok = std::fscanf(f, "%" SCNx64, v) == 1;
    std::fclose(f);
    return ok;
}

// Map a physical window: the UIO device whose map0 starts at `phys`, else /dev/mem.
void *map_window(std::uint64_t phys, std::size_t len) {
    for (int i = 0; i < 64; ++i) {
        const std::string dir = "/sys/class/uio/uio" + std::to_string(i) + "/maps/map0/";
        std::uint64_t addr = 0, size = 0;
        if (!read_sysfs_u64(dir + "addr", &addr) || addr != phys ||
            !read_sysfs_u64(dir + "size", &size) || size < len) {
            continue;
        }
        const std::string dev = "/dev/uio" + std::to_string(i);
        const int fd = ::open(dev.c_str(), O_RDWR | O_SYNC);
        TP_CHECK(fd >= 0, "balboa: %s (0x%" PRIx64 "): %s (the controller needs root)", dev.c_str(),
                 phys, std::strerror(errno));
        void *p = ::mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ::close(fd);
        TP_CHECK(p != MAP_FAILED, "balboa: mmap %s: %s", dev.c_str(), std::strerror(errno));
        return p;
    }
    // No UIO device covers the window (device tree not applied?)
    const int fd = ::open("/dev/mem", O_RDWR | O_SYNC);
    TP_CHECK(fd >= 0, "balboa: no UIO device at 0x%" PRIx64 " and /dev/mem: %s", phys,
             std::strerror(errno));
    void *p =
        ::mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, static_cast<off_t>(phys));
    ::close(fd);
    TP_CHECK(p != MAP_FAILED, "balboa: mmap /dev/mem at 0x%" PRIx64 ": %s", phys,
             std::strerror(errno));
    return p;
}

void unmap(volatile void *p, std::size_t len) {
    if (p != nullptr) {
        ::munmap(const_cast<void *>(p), len);
    }
}

// `width` bits of `v` into the 192-bit context `w` at bit `lsb`.
void put_bits(std::uint32_t w[6], unsigned lsb, unsigned width, std::uint64_t v) {
    for (unsigned i = 0; i < width; ++i) {
        const unsigned b = lsb + i;
        if ((v >> i) & 1u) {
            w[b / 32] |= 1u << (b % 32);
        }
    }
}

// BALBOA takes an IPv4 address with its first octet in the low byte.
std::uint32_t ip_first_octet_low(std::uint32_t ip) {
    return ((ip & 0xFFu) << 24) | ((ip & 0xFF00u) << 8) | ((ip >> 8) & 0xFF00u) | (ip >> 24);
}

} // namespace

void BalboaDevice::io_barrier() {
#if defined(__aarch64__)
    __asm__ __volatile__("dsb sy" ::: "memory");
#else
    __sync_synchronize();
#endif
}

void BalboaDevice::copy_to_dev64(volatile std::uint8_t *dst, const void *src) {
    std::uint64_t buf[HWHS_SLOT_BYTES / 8];
    std::memcpy(buf, src, sizeof(buf));
    volatile std::uint64_t *d = reinterpret_cast<volatile std::uint64_t *>(dst);
    for (std::size_t i = 0; i < HWHS_SLOT_BYTES / 8; ++i) {
        d[i] = buf[i];
    }
}

void BalboaDevice::copy_from_dev64(void *dst, const volatile std::uint8_t *src, std::size_t bytes) {
    std::uint64_t buf[HWHS_SLOT_BYTES / 8];
    const volatile std::uint64_t *s = reinterpret_cast<const volatile std::uint64_t *>(src);
    const std::size_t words = (bytes + 7) / 8;
    for (std::size_t i = 0; i < words && i < HWHS_SLOT_BYTES / 8; ++i) {
        buf[i] = s[i];
    }
    std::memcpy(dst, buf, bytes < HWHS_SLOT_BYTES ? bytes : HWHS_SLOT_BYTES);
}

BalboaDevice::BalboaDevice() {
    try {
        regs_ = static_cast<volatile std::uint32_t *>(map_window(HWHS_PHYS_REGS, HWHS_SIZE_REGS));
        slot_ = static_cast<volatile std::uint8_t *>(map_window(HWHS_PHYS_SLOT, HWHS_SIZE_SLOT));
        reply_ = static_cast<volatile std::uint8_t *>(map_window(HWHS_PHYS_REPLY, HWHS_SIZE_REPLY));
        trace_ =
            static_cast<volatile std::uint32_t *>(map_window(HWHS_PHYS_TRACE, HWHS_SIZE_TRACE));
        pacer_ = static_cast<volatile std::uint8_t *>(map_window(HWHS_PHYS_DEMO, HWHS_SIZE_DEMO));
        gpio_id_ =
            static_cast<volatile std::uint32_t *>(map_window(HWHS_PHYS_GPIO_ID, HWHS_SIZE_GPIO));
        gpio_ctl_ =
            static_cast<volatile std::uint32_t *>(map_window(HWHS_PHYS_GPIO_CTL, HWHS_SIZE_GPIO));

        // The GPIO is in the always-on pl0 domain and hwhs needs the net clock: check it before
        // touching hwhs, so a dead clock reports instead of hanging the bus.
        TP_CHECK(gpio_ctl_[GPIO2_DATA / 4] & CTL_IN_CLK_LOCKED,
                 "balboa: platform clock wizard not locked (axi_gpio_ctl ch2 = 0x%08x): is this "
                 "the backline-roce bitstream?",
                 gpio_ctl_[GPIO2_DATA / 4]);
        clk_hz_ = rd(HWHS_CLK_HZ);
        TP_CHECK(clk_hz_ != 0 && clk_hz_ == gpio_id_[GPIO2_DATA / 4],
                 "balboa: CLK_HZ %u does not match the platform (%u): wrong bitstream?", clk_hz_,
                 gpio_id_[GPIO2_DATA / 4]);
    } catch (...) {
        release(); // a throwing constructor gets no destructor call
        throw;
    }
}

BalboaDevice::~BalboaDevice() { release(); }

void BalboaDevice::release() noexcept {
    unmap(regs_, HWHS_SIZE_REGS);
    unmap(slot_, HWHS_SIZE_SLOT);
    unmap(reply_, HWHS_SIZE_REPLY);
    unmap(trace_, HWHS_SIZE_TRACE);
    unmap(pacer_, HWHS_SIZE_DEMO);
    unmap(gpio_id_, HWHS_SIZE_GPIO);
    unmap(gpio_ctl_, HWHS_SIZE_GPIO);
    regs_ = nullptr;
    slot_ = reply_ = pacer_ = nullptr;
    trace_ = gpio_id_ = gpio_ctl_ = nullptr;
}

void BalboaDevice::stack_reset(int timeout_ms) {
    gpio_ctl_[GPIO_DATA / 4] = CTL_OUT_STACK_RESET;
    ::usleep(1000);
    gpio_ctl_[GPIO_DATA / 4] = 0;

    const std::uint64_t deadline = now_ns() + std::uint64_t(timeout_ms) * 1'000'000ull;
    while (!(rd(HWHS_STATUS) & HWHS_ST_STACK_UP)) {
        TP_CHECK(now_ns() < deadline,
                 "balboa: BALBOA did not come out of reset (ctl status 0x%08x)",
                 gpio_ctl_[GPIO2_DATA / 4]);
        ::usleep(100);
    }
}

void BalboaDevice::net_push(std::uint32_t bits) {
    wr(HWHS_NET_CTRL, bits);
    const std::uint64_t deadline = now_ns() + std::uint64_t(NET_PUSH_TIMEOUT_MS) * 1'000'000ull;
    while (rd(HWHS_NET_PEND) & bits) {
        TP_CHECK(now_ns() < deadline, "balboa: BALBOA did not take NET_CTRL 0x%x (NET_PEND 0x%x)",
                 bits, rd(HWHS_NET_PEND));
    }
}

void BalboaDevice::set_addr(std::uint32_t ip, const std::uint8_t mac[6]) {
    wr(HWHS_LOCAL_IP, ip);
    wr(HWHS_LOCAL_MAC_LO, (std::uint32_t{mac[2]} << 24) | (std::uint32_t{mac[3]} << 16) |
                              (std::uint32_t{mac[4]} << 8) | mac[5]);
    wr(HWHS_LOCAL_MAC_HI, (std::uint32_t{mac[0]} << 8) | mac[1]);
    net_push(HWHS_NET_SET_IP | HWHS_NET_SET_MAC);
}

bool BalboaDevice::arp(std::uint32_t ip, int timeout_ms) {
    const std::uint32_t arp_rx = HWHS_NET_STATS + 4 * HWHS_STAT_ARP_RX;
    const std::uint32_t rx0 = rd(arp_rx);
    wr(HWHS_ARP_IP, ip);
    net_push(HWHS_NET_ARP);

    // BALBOA keeps the answer to itself; an ARP frame coming in is the sign it has it.
    const std::uint64_t deadline = now_ns() + std::uint64_t(timeout_ms) * 1'000'000ull;
    while (rd(arp_rx) == rx0) {
        if (now_ns() >= deadline) {
            return false;
        }
        ::usleep(100);
    }
    ::usleep(1000); // the reply reaches the ARP table a few cycles after the counter
    return true;
}

// The six 32-bit words of each context (word i = bits [32i+31:32i]):
//   rdma_qp_ctx_t   [31:0] new_state (0) [55:32] qp_num [79:56] remote_psn [103:80] local_psn
//                   [135:104] r_key [183:136] vaddr
//   rdma_qp_conn_t  [15:0] local_qpn [39:16] remote_qpn [167:40] remote_ip_address
//                   [183:168] remote_udp_port
// BALBOA takes remote_psn as the PSN it sends with and local_psn as the one it expects, so one
// PSN for both directions keeps the two from being mixed up.
void BalboaDevice::connect(const Connection &c) {
    std::uint32_t q[6] = {}, n[6] = {};
    put_bits(q, 32, 24, FPGA_QPN);
    put_bits(q, 56, 24, c.psn);
    put_bits(q, 80, 24, c.psn);
    put_bits(q, 104, 32, c.rkey);
    put_bits(q, 136, 48, c.ring_iova);
    put_bits(n, 0, 16, FPGA_QPN);
    put_bits(n, 16, 24, c.remote_qpn);
    for (unsigned i = 0; i < 4; ++i) { // the IPv4 address four times
        put_bits(n, 40 + 32 * i, 32, ip_first_octet_low(c.remote_ip));
    }
    put_bits(n, 168, 16, UDP_SPORT);
    for (unsigned i = 0; i < 6; ++i) {
        wr(HWHS_QP_CTX + 4 * i, q[i]);
        wr(HWHS_CONN_CTX + 4 * i, n[i]);
    }
    net_push(HWHS_NET_QP_PUSH | HWHS_NET_CONN_PUSH);
}

} // namespace catalyst::transport::balboa
