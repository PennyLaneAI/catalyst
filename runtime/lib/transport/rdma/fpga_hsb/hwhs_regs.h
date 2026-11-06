/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright 2026 Xanadu Quantum Technologies Inc. */
/* Vendored from backline-hsb common/include/hwhs_regs.h, which is the copy to edit; keep the two in sync. */
/*
 * hwhs_regs.h - VPK120 PS address map and the hwhs-hsb register map.
 * Mirrors hw/rtl/hwhs/hwhs_regs.v and hw/bd/platform.tcl.
 */
#ifndef HWHS_REGS_H
#define HWHS_REGS_H

/* ---- PS physical windows (platform BD) ---------------------------------- */
#define HWHS_PHYS_REGS        0xA4000000u  /* hwhs registers + ECB mailbox, 64 KB */
#define HWHS_PHYS_MRMAC       0xA4040000u  /* MRMAC AXI-Lite, 64 KB */
#define HWHS_PHYS_GPIO_ID     0xA4050000u  /* ch1 BUILD_ID, ch2 hif clock in Hz */
#define HWHS_PHYS_GT_RST_MASK 0xA4060000u  /* AMD: GT reset-done inputs */
#define HWHS_PHYS_GT_CTRL     0xA4070000u  /* AMD: line rate, GT reset, loopback, main cursor */
#define HWHS_PHYS_GT_DEBUG    0xA4080000u  /* AMD: rx/tx datapath reset, pre/post cursor */
#define HWHS_PHYS_GPIO_CTL    0xA4090000u  /* ch1 out [0]=HSB reset, ch2 in = status */
#define HWHS_PHYS_DEMO        0x80000000u  /* transport_demo table, 1 MB (LPD) */
#define HWHS_PHYS_SLOT        0x91000000u  /* slot RAM, 64 KB (LPD) */
#define HWHS_PHYS_TRACE       0x92000000u  /* RTT trace RAM, 2 MB (LPD) */
#define HWHS_PHYS_REPLY       0x93000000u  /* reply RAM, 64 KB (LPD) */

#define HWHS_SIZE_REGS   0x10000u
#define HWHS_SIZE_MRMAC  0x10000u
#define HWHS_SIZE_GPIO   0x10000u
#define HWHS_SIZE_DEMO   0x100000u
#define HWHS_SIZE_SLOT   0x10000u
#define HWHS_SIZE_TRACE  0x200000u
#define HWHS_SIZE_REPLY  0x10000u

#define HWHS_SLOT_BYTES  64u          /* one 512-bit beat per request / reply */
#define HWHS_SLOTS       1024u        /* slot RAM and reply RAM entries */
#define HWHS_TRACE_ENTRIES (1u << 20) /* 16-bit RTT entries */

/* AXI GPIO registers */
#define GPIO_DATA   0x0u   /* channel 1 */
#define GPIO_TRI    0x4u
#define GPIO2_DATA  0x8u   /* channel 2 */
#define GPIO2_TRI   0xCu

/* axi_gpio_ctl */
#define CTL_OUT_HSB_RESET     (1u << 0)     /* ch1: hold HSB in reset */
#define CTL_IN_CLK_LOCKED     (1u << 0)     /* ch2: platform clock wizard locked */
#define CTL_IN_HSB_INIT_DONE  (1u << 1)
#define CTL_IN_HSB_IN_RESET   (1u << 2)
#define CTL_IN_GT_RST_DONE(x) (((x) >> 4) & 0xFu)

/* ---- hwhs-hsb registers ------------------------------------------------- */
#define HWHS_CTRL         0x000u  /* W1P [0] start; RW [1] abort, [2] soft_reset */
#define HWHS_STATUS       0x004u
#define HWHS_RING_IDX     0x008u
#define HWHS_NEXT_SEQ     0x00Cu
#define HWHS_RING_LEN     0x010u
#define HWHS_FLAGS        0x014u  /* [0] seq_stamp */
#define HWHS_LAST_SEQ     0x018u
#define HWHS_DONE_CNT     0x01Cu  /* rounds finished (reply or timeout) */
#define HWHS_REPLY_TO     0x030u  /* hif cycles, 0 = forever */
#define HWHS_ROUND_CNT    0x034u
#define HWHS_RTT_LO       0x038u
#define HWHS_RTT_HI       0x03Cu
#define HWHS_CLK_HZ       0x040u
#define HWHS_STALE_CNT    0x044u
#define HWHS_TIMEOUT_CNT  0x048u
#define HWHS_TX_CNT       0x04Cu
#define HWHS_RX_CNT       0x050u
#define HWHS_HSB_STATUS   0x064u  /* [0] init_done [1] in reset [7:4] GT reset done */
#define HWHS_SEQ_OFF      0x074u
#define HWHS_LAST_RD      0x078u
#define HWHS_BUILD_ID     0x07Cu
#define HWHS_ECB_CTRL     0x080u  /* W1P [0] send, [1] release */
#define HWHS_ECB_STATUS   0x084u  /* [0] tx_busy [1] rx_valid [2] rx_trunc [24:16] rx_len */
#define HWHS_ECB_TX_LEN   0x088u
#define HWHS_ECB_PORT     0x08Cu
#define HWHS_ECB_RX_CNT   0x090u
#define HWHS_ECB_DROP     0x094u
#define HWHS_ECB_OVF      0x098u
#define HWHS_DEMO_CTRL    0x0A8u  /* W1P [0] start_demo; RW [1] reset_demo */
#define HWHS_FREQ_LO      0x0ACu
#define HWHS_FREQ_HI      0x0B0u
#define HWHS_DEMO_STS     0x0B4u  /* [0] demo_done */
#define HWHS_CMD_CNT      0x0B8u
#define HWHS_STS1         0x0BCu  /* demo_err */
#define HWHS_SYN_DEPTH    0x0C0u
#define HWHS_FREQ_SPAN    0x0C4u
#define HWHS_LFSR_SEED    0x0C8u
#define HWHS_TRACE_CTRL   0x0CCu  /* [0] clear */
#define HWHS_TRACE_CNT    0x0D0u
#define HWHS_TRACE_STS    0x0D4u  /* [0] full [1] saturated */
#define HWHS_ECB_TX_BUF   0x400u  /* 64 words */
#define HWHS_ECB_RX_BUF   0x500u  /* 64 words */
#define HWHS_ECB_BUF_BYTES 256u

#define HWHS_CTRL_START       (1u << 0)
#define HWHS_CTRL_ABORT       (1u << 1)
#define HWHS_CTRL_SOFT_RESET  (1u << 2)

#define HWHS_ST_STATE(s)      ((s) & 0xFu)
#define HWHS_ST_BUSY          (1u << 4)
#define HWHS_ST_DONE          (1u << 5)
#define HWHS_ST_REPLY_SEEN    (1u << 6)
#define HWHS_ST_TIMEOUT       (1u << 7)
#define HWHS_ST_HSB_INIT_DONE (1u << 16)
#define HWHS_ST_DEMO_RUN      (1u << 17)
#define HWHS_ST_ECB_RX_VALID  (1u << 18)
#define HWHS_ST_DONE_CNT(s)   ((s) >> 24)   /* DONE_CNT[7:0] */

#define HWHS_ECB_SEND         (1u << 0)
#define HWHS_ECB_RELEASE      (1u << 1)
#define HWHS_ECB_ST_TX_BUSY   (1u << 0)
#define HWHS_ECB_ST_RX_VALID  (1u << 1)
#define HWHS_ECB_ST_RX_TRUNC  (1u << 2)
#define HWHS_ECB_ST_RX_LEN(s) (((s) >> 16) & 0x1FFu)

#define HWHS_DEMO_START       (1u << 0)
#define HWHS_DEMO_RESET       (1u << 1)

#endif /* HWHS_REGS_H */
