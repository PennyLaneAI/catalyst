/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright 2026 Xanadu Quantum Technologies Inc. */
/* Vendored from backline-roce common/include/hwhs_regs.h, which is the copy to edit; keep the two in sync. */
/*
 * hwhs_regs.h - VPK120 PS address map and the hwhs-balboa register map.
 * Mirrors hw/rtl/hwhs/hwhs_regs.v and hw/bd/platform.tcl.
 */
#ifndef HWHS_REGS_H
#define HWHS_REGS_H

/* ---- PS physical windows (platform BD) ---------------------------------- */
#define HWHS_PHYS_REGS        0xA4000000u  /* hwhs registers, 64 KB */
#define HWHS_PHYS_MRMAC       0xA4040000u  /* MRMAC AXI-Lite, 64 KB */
#define HWHS_PHYS_GPIO_ID     0xA4050000u  /* ch1 BUILD_ID, ch2 net clock in Hz */
#define HWHS_PHYS_GT_RST_MASK 0xA4060000u  /* AMD: GT reset-done inputs */
#define HWHS_PHYS_GT_CTRL     0xA4070000u  /* AMD: line rate, GT reset, loopback, main cursor */
#define HWHS_PHYS_GT_DEBUG    0xA4080000u  /* AMD: rx/tx datapath reset, pre/post cursor */
#define HWHS_PHYS_GPIO_CTL    0xA4090000u  /* ch1 out [0]=BALBOA reset, ch2 in = status */
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
#define CTL_OUT_STACK_RESET   (1u << 0)     /* ch1: hold BALBOA in reset */
#define CTL_IN_CLK_LOCKED     (1u << 0)     /* ch2: platform clock wizard locked */
#define CTL_IN_STACK_UP       (1u << 1)
#define CTL_IN_STACK_RESET    (1u << 2)
#define CTL_IN_GT_RST_DONE(x) (((x) >> 4) & 0xFu)

/* ---- hwhs-balboa registers ---------------------------------------------- */
#define HWHS_CTRL         0x000u  /* W1P [0] start; RW [1] abort, [2] soft_reset */
#define HWHS_STATUS       0x004u
#define HWHS_RING_IDX     0x008u
#define HWHS_NEXT_SEQ     0x00Cu
#define HWHS_RING_LEN     0x010u
#define HWHS_FLAGS        0x014u  /* [0] seq_stamp */
#define HWHS_LAST_SEQ     0x018u
#define HWHS_DONE_CNT     0x01Cu  /* rounds finished (reply or timeout) */
#define HWHS_REPLY_TO     0x030u  /* net clock cycles, 0 = forever */
#define HWHS_ROUND_CNT    0x034u
#define HWHS_RTT_LO       0x038u
#define HWHS_RTT_HI       0x03Cu
#define HWHS_CLK_HZ       0x040u
#define HWHS_STALE_CNT    0x044u
#define HWHS_TIMEOUT_CNT  0x048u
#define HWHS_TX_CNT       0x04Cu  /* requests posted to BALBOA */
#define HWHS_RX_CNT       0x050u  /* WRITEs from the coprocessor */
#define HWHS_NET_STATUS   0x064u  /* [0] stack up [1] held in reset [7:4] GT reset done */
#define HWHS_SEQ_OFF      0x074u
#define HWHS_LAST_RD      0x078u
#define HWHS_BUILD_ID     0x07Cu
#define HWHS_NET_CTRL     0x080u  /* W1P [0] set_ip [1] set_mac [2] arp_lookup [3] qp_push [4] conn_push */
#define HWHS_NET_PEND     0x084u  /* the same bits: not yet taken by BALBOA */
#define HWHS_LOCAL_IP     0x088u
#define HWHS_LOCAL_MAC_LO 0x08Cu
#define HWHS_LOCAL_MAC_HI 0x090u
#define HWHS_ARP_IP       0x094u
#define HWHS_LOCAL_QPN    0x098u  /* {vfid[3:0], pid[5:0]}, 0 (see hwhs.h) */
#define HWHS_REQ_LEN      0x09Cu  /* 1..64 */
#define HWHS_RMT_BASE_LO  0x0A0u
#define HWHS_RMT_BASE_HI  0x0A4u
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
#define HWHS_RMT_STRIDE   0x0D8u  /* log2 of the coprocessor ring's slot stride */
#define HWHS_ACK_CNT      0x0DCu
#define HWHS_WR_CNT       0x0E0u
#define HWHS_RD_CNT       0x0E4u
#define HWHS_RETX_WR      0x0E8u
#define HWHS_RETX_RD      0x0ECu
#define HWHS_NET_STATS    0x0F0u  /* HWHS_NET_STATS_N words, HWHS_STAT_* order */
#define HWHS_QP_CTX       0x140u  /* 6 words: rdma_qp_ctx_t */
#define HWHS_CONN_CTX     0x160u  /* 6 words: rdma_qp_conn_t */

#define HWHS_CTRL_START       (1u << 0)
#define HWHS_CTRL_ABORT       (1u << 1)
#define HWHS_CTRL_SOFT_RESET  (1u << 2)

#define HWHS_ST_STATE(s)      ((s) & 0xFu)
#define HWHS_ST_BUSY          (1u << 4)
#define HWHS_ST_DONE          (1u << 5)
#define HWHS_ST_REPLY_SEEN    (1u << 6)
#define HWHS_ST_TIMEOUT       (1u << 7)
#define HWHS_ST_POST_BUSY     (1u << 8)
#define HWHS_ST_STACK_UP      (1u << 16)
#define HWHS_ST_DEMO_RUN      (1u << 17)
#define HWHS_ST_DONE_CNT(s)   ((s) >> 24)   /* DONE_CNT[7:0] */

#define HWHS_NET_SET_IP       (1u << 0)
#define HWHS_NET_SET_MAC      (1u << 1)
#define HWHS_NET_ARP          (1u << 2)
#define HWHS_NET_QP_PUSH      (1u << 3)
#define HWHS_NET_CONN_PUSH    (1u << 4)

#define HWHS_DEMO_START       (1u << 0)
#define HWHS_DEMO_RESET       (1u << 1)

/* NET_STATS words (network_stack m_net_stats) */
enum {
    HWHS_STAT_RX_FRAMES,
    HWHS_STAT_TX_FRAMES,
    HWHS_STAT_ARP_RX,
    HWHS_STAT_ARP_TX,
    HWHS_STAT_ICMP_RX,
    HWHS_STAT_ICMP_TX,
    HWHS_STAT_ROCE_RX,
    HWHS_STAT_ROCE_TX,
    HWHS_STAT_IB_RX,
    HWHS_STAT_IB_TX,
    HWHS_STAT_PSN_DROP,
    HWHS_STAT_RETRANS,
    HWHS_STAT_STREAM_DOWN,
    HWHS_NET_STATS_N
};

#endif /* HWHS_REGS_H */
