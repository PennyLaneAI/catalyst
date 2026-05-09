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

// Full 3-pass pipeline: landscape-overlap → bias-shift → representative-selection.
// Check that is_representative, transfer_modes, direct_copy_count, warm_start_count
// are written.
// RUN: quantum-opt %s \
// RUN:   --doqaoa-landscape-overlap \
// RUN:   --doqaoa-bias-shift \
// RUN:   --doqaoa-representative-selection \
// RUN:   --split-input-file 2>/dev/null | FileCheck %s --check-prefix=ATTR

// MODES prefix: bias-threshold=0.0 forces all non-reps into warm-start (mode 2).
// RUN: quantum-opt %s \
// RUN:   --doqaoa-landscape-overlap \
// RUN:   --doqaoa-bias-shift \
// RUN:   '--doqaoa-representative-selection=bias-threshold=0.0' \
// RUN:   --split-input-file 2>/dev/null | FileCheck %s --check-prefix=MODES

// Guard: run representative-selection ALONE (no bias-shift attrs) → warning fires.
// RUN: quantum-opt %s \
// RUN:   --doqaoa-representative-selection \
// RUN:   --split-input-file 2>&1 | FileCheck %s --check-prefix=GUARD

// ============================================================
// Test 1: 4-cycle m=1, K=1.
//   sub-problem 0: representative → mode 0
//   sub-problem 1: bias_shifts[1]=0.0 < 0.3 → mode 1 (direct copy)
//   direct_copy_count = 1, warm_start_count = 0
// ============================================================

// ATTR-LABEL: func @cycle_rep_selection
// ATTR:       quantum.freeze_partition
// ATTR-SAME:  direct_copy_count = 1 : i32
// ATTR-SAME:  is_representative = array<i32: 1, 0>
// ATTR-SAME:  transfer_modes = array<i32: 0, 1>
// ATTR-SAME:  warm_start_count = 0 : i32

// MODES-LABEL: func @cycle_rep_selection
// MODES:       quantum.freeze_partition
// MODES-SAME:  direct_copy_count = 0 : i32
// MODES-SAME:  transfer_modes = array<i32: 0, 2>
// MODES-SAME:  warm_start_count = 1 : i32

// GUARD: warning: doqaoa-representative-selection: missing bias-shift attributes

func.func @cycle_rep_selection() {
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
    func.return
}

// -----

// ============================================================
// Test 2: Complete graph K4, m=1, K=1.
//   Symmetric graph → both landscapes identical → bias_shifts=[0,0].
//   sub-problem 0: representative → mode 0
//   sub-problem 1: mode 1 (direct copy)
// ============================================================

// ATTR-LABEL: func @complete_k4_rep_selection
// ATTR:       quantum.freeze_partition
// ATTR-SAME:  direct_copy_count = 1 : i32
// ATTR-SAME:  is_representative = array<i32: 1, 0>
// ATTR-SAME:  transfer_modes = array<i32: 0, 1>
// ATTR-SAME:  warm_start_count = 0 : i32

// MODES-LABEL: func @complete_k4_rep_selection
// MODES:       quantum.freeze_partition
// MODES-SAME:  transfer_modes = array<i32: 0, 2>
// MODES-SAME:  warm_start_count = 1 : i32

// GUARD: warning: doqaoa-representative-selection: missing bias-shift attributes

func.func @complete_k4_rep_selection() {
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
    func.return
}

// -----

// ============================================================
// Test 3: Biased Ising m=1. Strong linear bias → two different B_k.
//   bias_shifts[1] > 0 but still < 0.3 → mode 1 (direct copy) with threshold=0.3.
//   With threshold=0.0 → mode 2 (warm start).
// ============================================================

// ATTR-LABEL: func @biased_ising_rep_selection
// ATTR:       quantum.freeze_partition
// ATTR-SAME:  direct_copy_count = 1 : i32
// ATTR-SAME:  is_representative = array<i32: 1, 0>
// ATTR-SAME:  transfer_modes = array<i32: 0, 1>
// ATTR-SAME:  warm_start_count = 0 : i32

// MODES-LABEL: func @biased_ising_rep_selection
// MODES:       quantum.freeze_partition
// MODES-SAME:  transfer_modes = array<i32: 0, 2>
// MODES-SAME:  warm_start_count = 1 : i32

// GUARD: warning: doqaoa-representative-selection: missing bias-shift attributes

func.func @biased_ising_rep_selection() {
    %p = quantum.freeze_partition {
        hotspot_count   = 1 : i32,
        hotspot_indices = array<i32: 0>,
        h_quad = #quantum.dense_graph<4, dense<[
            [0.0, -0.5, 0.0, 0.0],
            [-0.5, 0.0, -0.5, 0.0],
            [0.0, -0.5, 0.0, -0.5],
            [0.0, 0.0, -0.5, 0.0]]> : tensor<4x4xf64>>,
        h_lin  = dense<[0.1, 0.0, 0.0, -0.1]> : tensor<4xf64>
    } : !quantum.partition<4, 1>
    func.return
}

// -----

// ============================================================
// Test 4: 4-cycle m=2, K=2. 4 sub-problems, 2 clusters.
//   representatives = [k_c0, k_c1].  Both reps get mode 0.
//   Non-reps with small ΔB → mode 1; with threshold=0 → mode 2.
//   direct_copy_count + warm_start_count = 2  (4 total - 2 reps).
// ============================================================

// ATTR-LABEL: func @cycle_m2_rep_selection
// ATTR:       quantum.freeze_partition
// ATTR-SAME:  is_representative
// ATTR-SAME:  transfer_modes
// ATTR-SAME:  warm_start_count

// MODES-LABEL: func @cycle_m2_rep_selection
// MODES:       quantum.freeze_partition
// MODES-SAME:  is_representative
// MODES-SAME:  transfer_modes
// MODES-SAME:  warm_start_count

// GUARD: warning: doqaoa-representative-selection: missing bias-shift attributes

func.func @cycle_m2_rep_selection() {
    %p = quantum.freeze_partition {
        hotspot_count   = 2 : i32,
        hotspot_indices = array<i32: 1, 2>,
        h_quad = #quantum.dense_graph<4, dense<[
            [ 0.0, -0.5,  0.0, -0.5],
            [-0.5,  0.0, -0.5,  0.0],
            [ 0.0, -0.5,  0.0, -0.5],
            [-0.5,  0.0, -0.5,  0.0]]> : tensor<4x4xf64>>,
        h_lin  = dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf64>
    } : !quantum.partition<4, 2>
    func.return
}

// -----

// ============================================================
// Test 5: Sparse path graph m=2.  Check attrs appear regardless of sparse format.
// ============================================================

// ATTR-LABEL: func @sparse_rep_selection
// ATTR:       quantum.freeze_partition
// ATTR-SAME:  direct_copy_count
// ATTR-SAME:  is_representative
// ATTR-SAME:  transfer_modes
// ATTR-SAME:  warm_start_count

// MODES-LABEL: func @sparse_rep_selection
// MODES:       quantum.freeze_partition
// MODES-SAME:  is_representative
// MODES-SAME:  transfer_modes

// GUARD: warning: doqaoa-representative-selection: missing bias-shift attributes

func.func @sparse_rep_selection() {
    %p = quantum.freeze_partition {
        hotspot_count   = 2 : i32,
        hotspot_indices = array<i32: 1, 2>,
        h_quad = #quantum.sparse_graph<4, 3,
            [0, 1, 2], [1, 2, 3],
            dense<[-0.5, -0.5, -0.5]> : tensor<3xf64>>,
        h_lin  = dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf64>
    } : !quantum.partition<4, 2>
    func.return
}
