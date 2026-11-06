/* SPDX-License-Identifier: Apache-2.0 */
/* Copyright 2026 Xanadu Quantum Technologies Inc. */
/* Vendored from backline-hsb common/include/hsb_regs.h, which is the copy to edit; keep the two in sync. */
/*
 * hsb_regs.h - the HSB (EA rev 0x2609) registers and ECB protocol bits we use.
 *
 * Addresses are HSB's APB map as reached through ECB. They come from NVIDIA's host
 * library (src/hololink/core/{hololink.hpp,data_channel.hpp,enumerator.cpp}) and the
 * RTL (fpga/nv_hsb_ip/reg_map/regmap_pkg.sv, dp_pkt/dp_pkt_top.sv,
 * sensor_tx/tx_stream_buffer.sv) on EA_CUDAQ_10_26 a4b2059.
 */
#ifndef HSB_REGS_H
#define HSB_REGS_H

/* ---- ECB (UDP 8192) -------------------------------------------------------- */
#define HSB_ECB_UDP_PORT          8192u
#define HSB_ECB_WR_DWORD          0x04u
#define HSB_ECB_WR_BLOCK          0x09u
#define HSB_ECB_RD_DWORD          0x14u
#define HSB_ECB_RD_BLOCK          0x19u
#define HSB_ECB_FLAG_ACK_REQUEST  0x01u
#define HSB_ECB_FLAG_SEQ_CHECK    0x02u
#define HSB_ECB_RESP_SUCCESS      0x00u
/* request:  cmd, flags, seq(be16), 0, 0, addr(be32) [, value(be32)]
 * reply:    cmd', flags, seq(be16), code, 0, addr(be32), value(be32), latched_seq(be16) */

/* ---- device ---------------------------------------------------------------- */
#define HSB_IP_VERSION            0x00000080u  /* {BUILD_REV[15:0], HOLOLINK_REV} */
#define HSB_IP_VERSION_MIN        0x2602u      /* NVIDIA's host refuses older (data_channel.cpp) */
#define HSB_DEFAULT_IP            0xC0A80002u  /* host port i = 192.168.0.2 + i (bootp.sv) */

/* ---- data plane 0 (host port 0) --------------------------------------------- */
#define HSB_HIF(dp)               (0x02000300u + 0x10000u * (dp))
#define HSB_DP_PACKET_SIZE        0x04u        /* in 128 B pages */
#define HSB_DP_PACKET_UDP_PORT    0x08u        /* UDP source port of data packets */
#define HSB_DP_VP_MASK            0x0Cu        /* bit per virtual port */
#define HSB_DATA_SOURCE_UDP_PORT  12288u       /* hololink.hpp DATA_SOURCE_UDP_PORT */

/* ---- sensor (virtual port) 0 ----------------------------------------------- */
#define HSB_SIF(s)                (0x01000000u + 0x10000u * (s))
#define HSB_VP(vp)                (0x00001000u + 0x40u * (vp))
#define HSB_DP_QP                 0x00u        /* destination QPN (bit 24: 1722B/COE, keep 0) */
#define HSB_DP_RKEY               0x04u
#define HSB_DP_PAGE_LSB           0x08u        /* ring base, byte address [31:0] */
#define HSB_DP_PAGE_MSB           0x0Cu        /* ring base [63:32] */
#define HSB_DP_PAGE_INC           0x10u        /* slot pitch in 128 B pages */
#define HSB_DP_MAX_BUFF           0x14u        /* last slot index (first = 0) */
#define HSB_DP_BUFFER_LENGTH      0x18u        /* bytes per frame = per message */
#define HSB_DP_HOST_MAC_LOW       0x20u        /* MAC bytes 2..5 */
#define HSB_DP_HOST_MAC_HIGH      0x24u        /* MAC bytes 0..1 */
#define HSB_DP_HOST_IP            0x28u
#define HSB_DP_HOST_UDP_PORT      0x2Cu        /* [15:0] port, [16] md_disable */
#define HSB_DP_VLAN               0x34u
#define HSB_MD_DISABLE            (1u << 16)   /* one packet per frame, no metadata packet */
#define HSB_ROCE_UDP_PORT         4791u
#define HSB_PAGE_SIZE             128u

/* ---- sensor-TX 0 (the reply path) ------------------------------------------ */
#define HSB_STX0                  0x01200000u  /* sensor-TX 0 (NVIDIA's RFSoC init_reg) */
#define HSB_STX_CTRL              0x00u        /* [0] output enable, [2] latch thresholds on 0->1 */
#define HSB_STX_AFULL_THRESH      0x04u
#define HSB_STX_AEMPTY_THRESH     0x08u
/* The coprocessor's reply SEND goes to QP 2 (sensor-TX port 0: dest_qp[8:1] - 1). */
#define HSB_STX_QPN(port)         (((port) + 1u) << 1)

#endif /* HSB_REGS_H */
