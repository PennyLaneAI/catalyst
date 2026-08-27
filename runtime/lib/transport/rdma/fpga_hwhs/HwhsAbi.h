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

// ERNIC™ is a trademark of Advanced Micro Devices, Inc.

// HWHS ABI

#pragma once

#include <cstdint>
#include <sys/ioctl.h>

// It was defined in the `umm_export.h` of VPK120 sysroots
// We copy it here to avoid depending on that sysroot.
enum {
    XMEM_PL_DDR = 1,
    XMEM_PS_DDR = 2,
    XMEM_BRAM = 5,
    XMEM_REPLY_BRAM = 6,
};
#define XMEM_INVALID_VIRT_ADD 0
#define XMEM_IS_INVALID_VADDR(va) ((va) == XMEM_INVALID_VIRT_ADD)

// The following structs are defined in `hw_hs.h` of the HWHS Linux kernel driver.
namespace catalyst::transport::hwhs {

struct hh_round_cfg {
    std::uint32_t qp_num, qp_cnt, payload_len, pattern;
    std::uint64_t data_buf, sq_base;
    std::uint32_t sq_depth, flags, reply_to, timer_qp_sel, run_wqe_idx;
    std::uint64_t reply_buf;
    std::uint32_t reply_seq_off;
    std::uint32_t reply_stride_log2;
};

struct hh_wqe_cfg {
    std::uint32_t idx, opcode, xfer_len, rkey, va_lsb, va_msb;
    std::uint64_t local_offset;
    std::uint32_t wrid, qp_ref, ring_en, ring_len, stride_local_log2, stride_remote_log2;
};

struct hh_status_rd {
    std::uint32_t status, round_cnt;
    std::uint64_t rtt;
    std::uint32_t last_rd;
};

struct hh_mr_rebase {
    std::uint32_t rkey;
    std::uint32_t _pad;
    std::uint64_t bram_pa;
};

struct hh_qpctx_cfg {
    std::uint32_t ctx_idx;           // QP-context slot (0..HH_NUM_CTX-1)
    std::uint32_t qp_num;            // physical ERNIC™ QP for this context
    std::uint64_t sq_base;           // SQ ring base address (DDR/BRAM)
    std::uint32_t sq_depth;          // SQ ring depth (entries)
    std::uint64_t reply_buf;         // per-QP reply buffer base address
    std::uint32_t flags;             // HH_FLAG_*
    std::uint32_t reply_to;          // per-QP reply timeout, core cycles (0 = wait forever)
    std::uint32_t reply_seq_off;     // per-QP: seq_num's byte offset in a reply slot. A round
                                     // is done when the slot reads that round's sequence there.
    std::uint32_t reply_stride_log2; // per-QP reply slot stride (log2 bytes)
    std::uint32_t rebind;            // 1 = (re)create QP: zero sq_pi + ERNIC SQPI; 0 = update-only
};

// Pacer / transport-demo configuration. Mirrors struct hh_demo_cfg in the driver's
// hw_hs.h -- the layout is part of the ioctl number (via _IOW's sizeof), so it must
// match byte for byte.
struct hh_demo_cfg {
    std::uint64_t freq_num;  // cycles between sends inside transport_demo
    std::uint32_t cmd_cnt;   // syndromes per run (syn_sum)
    std::uint32_t syn_depth; // syndrome table size in BYTES (multiple of 64)
    std::uint32_t freq_span; // jitter mask; interval uniform over [freq, freq+span]
    std::uint32_t lfsr_seed; // 0 = engine default; set it to reproduce a jittered run
};

// Mirrors the driver's struct hh_trace_status. cnt is the number of 16-bit entries in the
// trace RAM, each a completed round's rtt in CORE CLOCKS -- convert with the real clock,
// do not bake in 5 ns.
struct hh_trace_status {
    std::uint32_t cnt;
    std::uint32_t full; // filled and stopped: rounds after entry cnt were not recorded
    std::uint32_t sat;  // an entry was clamped to 0xFFFF, so it is a floor not a value
    std::uint32_t _pad;
};

struct hh_demo_status {
    std::uint32_t done, err_cnt, cmd_cnt, syn_depth;
    std::uint64_t freq_num;
};

enum hw_hsk_mod_fn {
    HH_FN_CONFIG = 0,
    HH_FN_WQE,
    HH_FN_START,
    HH_FN_ABORT,
    HH_FN_RESET,
    HH_FN_READ_STATUS,
    HH_FN_SET_PATTERN,
    HH_FN_SET_WQE_IDX,
    HH_FN_QPCTX_COMMIT,
    HH_FN_MR_REBASE,  // 9
    HH_FN_DEMO_CFG,   // 10
    HH_FN_DEMO_START, // 11
    HH_FN_DEMO_STOP,  // 12
    HH_FN_DEMO_STATUS,
    HH_FN_TRACE_ARM,
    HH_FN_TRACE_STATUS, // 13
};

#define HH_MAGIC_NUM 'H'
#define HH_CONFIG _IOW(HH_MAGIC_NUM, HH_FN_CONFIG, struct catalyst::transport::hwhs::hh_round_cfg)
#define HH_WQE_COMMIT _IOW(HH_MAGIC_NUM, HH_FN_WQE, struct catalyst::transport::hwhs::hh_wqe_cfg)
#define HH_START _IO(HH_MAGIC_NUM, HH_FN_START)
#define HH_ABORT _IO(HH_MAGIC_NUM, HH_FN_ABORT)
#define HH_RESET _IO(HH_MAGIC_NUM, HH_FN_RESET)
#define HH_READ_STATUS                                                                             \
    _IOR(HH_MAGIC_NUM, HH_FN_READ_STATUS, struct catalyst::transport::hwhs::hh_status_rd)
#define HH_SET_WQE_IDX _IOW(HH_MAGIC_NUM, HH_FN_SET_WQE_IDX, std::uint32_t)
#define HH_QPCTX_COMMIT                                                                            \
    _IOW(HH_MAGIC_NUM, HH_FN_QPCTX_COMMIT, struct catalyst::transport::hwhs::hh_qpctx_cfg)
#define HH_MR_REBASE _IOW(HH_MAGIC_NUM, 9, struct catalyst::transport::hwhs::hh_mr_rebase)
#define HH_DEMO_CFG                                                                                \
    _IOW(HH_MAGIC_NUM, HH_FN_DEMO_CFG, struct catalyst::transport::hwhs::hh_demo_cfg)
#define HH_DEMO_START_RUN _IO(HH_MAGIC_NUM, HH_FN_DEMO_START)
#define HH_DEMO_STOP_RUN _IO(HH_MAGIC_NUM, HH_FN_DEMO_STOP)
#define HH_DEMO_STATUS                                                                             \
    _IOR(HH_MAGIC_NUM, HH_FN_DEMO_STATUS, struct catalyst::transport::hwhs::hh_demo_status)
#define HH_TRACE_ARM _IO(HH_MAGIC_NUM, HH_FN_TRACE_ARM)
#define HH_TRACE_STATUS                                                                            \
    _IOR(HH_MAGIC_NUM, HH_FN_TRACE_STATUS, struct catalyst::transport::hwhs::hh_trace_status)

// Flags definitions
constexpr std::uint32_t HH_FLAG_REPOST_WQE = 1u << 0; // re-post the WQE each round (default on)
constexpr std::uint32_t HH_FLAG_REPLY_POLL = 1u << 3; // detect the reply by POLLING reply_buf

// State definitions
constexpr std::uint32_t HH_ST_STATE_MASK = 0xFu;
constexpr std::uint32_t HH_ST_DONE = 1u << 5;
constexpr std::uint32_t HH_ST_TIMEOUT = 1u << 7;
constexpr std::uint32_t HH_ST_AXI_ERR = 1u << 8;

// Opcode definitions
constexpr std::uint32_t HH_OP_WRITE = 0;

// Control registers exposed by the driver's mmap() of the 4 KB AXI4-Lite window on /dev/xib0
constexpr std::size_t HH_REG_WIN_SIZE = 0x1000;  // 4 KB control/status window
constexpr std::uint32_t HH_REG_CTRL = 0x00;      // control register offset
constexpr std::uint32_t HH_CTRL_START = 1u << 0; // W1P: start one round

// Demo mode control
constexpr std::uint32_t HH_REG_DEMO_CTRL = 0xA8;
constexpr std::uint32_t HH_DEMO_START = 1u << 0; // W1P: arm one demo round
constexpr std::uint32_t HH_DEMO_RESET = 1u << 1; // level: clear the demo counters

constexpr const char *kHwhsDev = "/dev/xib0";
constexpr std::uint32_t kWqeBytes = 64u;
constexpr double kClkMhz = 200.0;

} // namespace catalyst::transport::hwhs
