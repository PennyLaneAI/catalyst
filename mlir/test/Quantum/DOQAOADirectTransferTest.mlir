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

// Tests for --doqaoa-direct-transfer (Phase 3, Task 6).
//
// DT prefix: full 6-pass pipeline through doqaoa-direct-transfer.
// RUN: quantum-opt %s \
// RUN:   --doqaoa-landscape-overlap \
// RUN:   --doqaoa-bias-shift \
// RUN:   --doqaoa-representative-selection \
// RUN:   --doqaoa-training-schedule \
// RUN:   --doqaoa-shared-buffer \
// RUN:   --doqaoa-direct-transfer \
// RUN:   --split-input-file 2>/dev/null | FileCheck %s --check-prefix=DT

// NOWARM prefix: threshold=100.0 forces all transfers to direct copy.
// RUN: quantum-opt %s \
// RUN:   --doqaoa-landscape-overlap \
// RUN:   '--doqaoa-bias-shift=bias-threshold=100.0' \
// RUN:   '--doqaoa-representative-selection=bias-threshold=100.0' \
// RUN:   --doqaoa-training-schedule \
// RUN:   --doqaoa-shared-buffer \
// RUN:   --doqaoa-direct-transfer \
// RUN:   --split-input-file 2>/dev/null | FileCheck %s --check-prefix=NOWARM

// ALLWARM prefix: threshold=0.0 forces all transfers to warm-start.
// RUN: quantum-opt %s \
// RUN:   --doqaoa-landscape-overlap \
// RUN:   '--doqaoa-bias-shift=bias-threshold=0.0' \
// RUN:   '--doqaoa-representative-selection=bias-threshold=0.0' \
// RUN:   --doqaoa-training-schedule \
// RUN:   --doqaoa-shared-buffer \
// RUN:   '--doqaoa-direct-transfer' \
// RUN:   --split-input-file 2>/dev/null | FileCheck %s --check-prefix=ALLWARM

// GUARD prefix: run without shared-buffer pass → bufSize defaults to 2.
// RUN: quantum-opt %s \
// RUN:   --doqaoa-direct-transfer \
// RUN:   --split-input-file 2>&1 | FileCheck %s --check-prefix=GUARD

// ============================================================
// Test 1: 4-cycle m=1, small bias → direct copy.
//   sp0 = rep (mode 0), sp1 = direct copy (mode 1).
//   With default threshold=0.3: deltaB is small → direct copy.
//   dt_direct_count = 1, dt_warmstart_count = 0.
// ============================================================

// DT-LABEL: func @cycle_direct
// DT:       quantum.freeze_partition
// DT-SAME:  dt_direct_count = 1 : i32
// DT-SAME:  dt_warmstart_count = 0 : i32
// DT-SAME:  param_byte_count = 16 : i32

// NOWARM-LABEL: func @cycle_direct
// NOWARM:   quantum.freeze_partition
// NOWARM-SAME: dt_direct_count = 1 : i32
// NOWARM-SAME: dt_warmstart_count = 0 : i32

// ALLWARM-LABEL: func @cycle_direct
// ALLWARM:  quantum.freeze_partition
// ALLWARM-SAME: dt_direct_count = 1 : i32
// ALLWARM-SAME: dt_warmstart_count = 0 : i32

// GUARD-LABEL: func @cycle_direct
// GUARD:    quantum.freeze_partition
// GUARD-SAME: dt_direct_count

func.func @cycle_direct(%p_rep : !quantum.params) {
    %p = quantum.freeze_partition {
        hotspot_count   = 1 : i32,
        hotspot_indices = array<i32: 0>,
        h_quad = #quantum.dense_graph<4, dense<[
            [ 0.0, -0.5,  0.0, -0.5],
            [-0.5,  0.0, -0.5,  0.0],
            [ 0.0, -0.5,  0.0, -0.5],
            [-0.5,  0.0, -0.5,  0.0]]> : tensor<4x4xf64>>,
        h_lin  = dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf64>
    } : !quantum.partition<4, 1>

    %t1 = quantum.bias_transfer(%p_rep : !quantum.params) {
        B_rep = 0.0 : f64,
        B_target = 0.05 : f64,
        threshold = 0.3 : f64
    } : !quantum.params

    func.return
}

// -----

// ============================================================
// Test 2: Complete graph K4, m=1, large bias → warm-start.
//   sp0 = rep (mode 0), sp1 = warm-start (mode 2).
//   With default threshold=0.3 and large deltaB: warm-start.
//   dt_direct_count = 0, dt_warmstart_count = 1.
// ============================================================

// DT-LABEL: func @k4_warmstart_transfer
// DT:       quantum.freeze_partition
// DT-SAME:  dt_direct_count = 0 : i32
// DT-SAME:  dt_warmstart_count = 1 : i32

// NOWARM-LABEL: func @k4_warmstart_transfer
// NOWARM:   quantum.freeze_partition
// NOWARM-SAME: dt_direct_count = 0 : i32

// ALLWARM-LABEL: func @k4_warmstart_transfer
// ALLWARM:  quantum.freeze_partition
// ALLWARM-SAME: dt_warmstart_count = 1 : i32

func.func @k4_warmstart_transfer(%p_rep : !quantum.params) {
    %p = quantum.freeze_partition {
        hotspot_count   = 1 : i32,
        hotspot_indices = array<i32: 0>,
        h_quad = #quantum.dense_graph<4, dense<[
            [ 0.0, -1.0, -1.0, -1.0],
            [-1.0,  0.0, -1.0, -1.0],
            [-1.0, -1.0,  0.0, -1.0],
            [-1.0, -1.0, -1.0,  0.0]]> : tensor<4x4xf64>>,
        h_lin  = dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf64>
    } : !quantum.partition<4, 1>

    %t1 = quantum.bias_transfer(%p_rep : !quantum.params) {
        B_rep = 0.0 : f64,
        B_target = 0.8 : f64,
        threshold = 0.3 : f64
    } : !quantum.params

    func.return
}

// -----

// ============================================================
// Test 3: m=2, 4 sub-problems, mixed direct + warm-start.
//   sp0 = rep, sp1 = copy, sp2 = warmstart, sp3 = copy.
//   2 bias_transfer ops with different deltaB values.
// ============================================================

// DT-LABEL: func @mixed_transfers
// DT:       quantum.freeze_partition
// DT-SAME:  param_byte_count = 16 : i32

func.func @mixed_transfers(%p_rep : !quantum.params) {
    %p = quantum.freeze_partition {
        hotspot_count   = 2 : i32,
        hotspot_indices = array<i32: 0, 1>,
        h_quad = #quantum.dense_graph<6, dense<[
            [ 0.0, -0.5, -0.5,  0.0,  0.0,  0.0],
            [-0.5,  0.0, -0.5, -0.5,  0.0,  0.0],
            [-0.5, -0.5,  0.0,  0.0, -0.5,  0.0],
            [ 0.0, -0.5,  0.0,  0.0, -0.5, -0.5],
            [ 0.0,  0.0, -0.5, -0.5,  0.0, -0.5],
            [ 0.0,  0.0,  0.0, -0.5, -0.5,  0.0]]> : tensor<6x6xf64>>,
        h_lin  = dense<[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]> : tensor<6xf64>
    } : !quantum.partition<6, 2>

    // small deltaB → direct copy
    %t1 = quantum.bias_transfer(%p_rep : !quantum.params) {
        B_rep = 0.1 : f64,
        B_target = 0.15 : f64,
        threshold = 0.3 : f64
    } : !quantum.params

    // large deltaB → warm-start
    %t2 = quantum.bias_transfer(%p_rep : !quantum.params) {
        B_rep = 0.1 : f64,
        B_target = 0.7 : f64,
        threshold = 0.3 : f64
    } : !quantum.params

    func.return
}
