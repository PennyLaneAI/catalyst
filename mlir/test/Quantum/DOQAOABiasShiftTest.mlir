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

// Full pipeline: landscape-overlap → bias-shift.  Check attributes written.
// RUN: quantum-opt %s \
// RUN:   --doqaoa-landscape-overlap \
// RUN:   --doqaoa-bias-shift \
// RUN:   --split-input-file 2>/dev/null | FileCheck %s --check-prefix=ATTR

// Tight basin-tol=0.001 → basin warning fires for the 4-cycle graph.
// RUN: quantum-opt %s \
// RUN:   --doqaoa-landscape-overlap \
// RUN:   '--doqaoa-bias-shift=basin-tol=0.001' \
// RUN:   --split-input-file 2>&1 | FileCheck %s --check-prefix=BASIN

// Guard: run bias-shift ALONE (no cluster attrs on op) → missing-attr warning.
// RUN: quantum-opt %s \
// RUN:   --doqaoa-bias-shift \
// RUN:   --split-input-file 2>&1 | FileCheck %s --check-prefix=GUARD

// ============================================================
// Test 1: 4-cycle m=1, zero bias.
// All 7 bias-shift attrs written in alphabetical order:
//   b_values, basin_beta, basin_gamma, bias_shifts, ..., init_beta, init_gamma,
//   ..., representatives.
// ============================================================

// ATTR-LABEL: func @cycle_bias_shift
// ATTR:       quantum.freeze_partition
// ATTR-SAME:  b_values
// ATTR-SAME:  basin_beta
// ATTR-SAME:  basin_gamma
// ATTR-SAME:  bias_shifts
// ATTR-SAME:  init_beta
// ATTR-SAME:  init_gamma
// ATTR-SAME:  representatives = array<i32: 0>

// BASIN: warning: doqaoa-bias-shift: basin centre
// BASIN-SAME: deviates from shortcut

// GUARD: warning: doqaoa-bias-shift: missing cluster attributes

func.func @cycle_bias_shift() {
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
// Test 2: Ising with strong bias.  Both attrs written; K=1 so
//         representatives = [0].
// ============================================================

// ATTR-LABEL: func @ising_strong_bias
// ATTR:       quantum.freeze_partition
// ATTR-SAME:  b_values
// ATTR-SAME:  basin_beta
// ATTR-SAME:  bias_shifts
// ATTR-SAME:  init_gamma = -0.52359877559829882
// ATTR-SAME:  representatives = array<i32: 0>

func.func @ising_strong_bias() {
    %p = quantum.freeze_partition {
        hotspot_count   = 1 : i32,
        hotspot_indices = array<i32: 0>,
        h_quad = #quantum.dense_graph<4, dense<[
            [0.0, -0.1, 0.0, 0.0],
            [-0.1, 0.0, -0.1, 0.0],
            [0.0, -0.1, 0.0, -0.1],
            [0.0, 0.0, -0.1, 0.0]]> : tensor<4x4xf64>>,
        h_lin  = dense<[2.0, 0.0, 0.0, -2.0]> : tensor<4xf64>
    } : !quantum.partition<4, 1>
    func.return
}

// -----

// ============================================================
// Test 3: Sparse path graph — both dense and sparse attrs work.
// ============================================================

// ATTR-LABEL: func @sparse_bias_shift
// ATTR:       quantum.freeze_partition
// ATTR-SAME:  b_values
// ATTR-SAME:  basin_beta
// ATTR-SAME:  bias_shifts
// ATTR-SAME:  init_beta = -0.39269908169872414
// ATTR-SAME:  representatives

func.func @sparse_bias_shift() {
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

// -----

// ============================================================
// Test 4: 4-cycle m=2.  K=2 → representatives has 2 entries.
// ============================================================

// ATTR-LABEL: func @cycle_m2_representatives
// ATTR:       quantum.freeze_partition
// ATTR-SAME:  b_values
// ATTR-SAME:  bias_shifts
// ATTR-SAME:  representatives = array<i32:

func.func @cycle_m2_representatives() {
    %p = quantum.freeze_partition {
        hotspot_count   = 2 : i32,
        hotspot_indices = array<i32: 0, 1>,
        h_quad = #quantum.dense_graph<4, dense<[
            [ 0.0, -0.5,  0.0, -0.5],
            [-0.5,  0.0, -0.5,  0.0],
            [ 0.0, -0.5,  0.0, -0.5],
            [-0.5,  0.0, -0.5,  0.0]]> : tensor<4x4xf64>>,
        h_lin  = dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf64>
    } : !quantum.partition<4, 2>
    func.return
}
