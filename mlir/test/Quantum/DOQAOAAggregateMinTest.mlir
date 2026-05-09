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

// Tests for --doqaoa-aggregate-min (Phase 3, Task 7).
//
// AGG prefix: full pipeline including warmstart-scheduler before aggregate-min.
// RUN: quantum-opt %s \
// RUN:   --doqaoa-landscape-overlap \
// RUN:   --doqaoa-bias-shift \
// RUN:   --doqaoa-representative-selection \
// RUN:   --doqaoa-training-schedule \
// RUN:   --doqaoa-shared-buffer \
// RUN:   --doqaoa-warmstart-scheduler \
// RUN:   --doqaoa-aggregate-min \
// RUN:   --split-input-file 2>/dev/null | FileCheck %s --check-prefix=AGG

// NOPRIOR prefix: run aggregate-min without warmstart-scheduler (all NaN energies).
//   Should default to best_k=0 gracefully.
// RUN: quantum-opt %s \
// RUN:   --doqaoa-aggregate-min \
// RUN:   --split-input-file 2>/dev/null | FileCheck %s --check-prefix=NOPRIOR

// FORCED prefix: use threshold=0.0 to force warm-starts → non-NaN energies.
// RUN: quantum-opt %s \
// RUN:   --doqaoa-landscape-overlap \
// RUN:   '--doqaoa-bias-shift=bias-threshold=0.0' \
// RUN:   '--doqaoa-representative-selection=bias-threshold=0.0' \
// RUN:   --doqaoa-training-schedule \
// RUN:   --doqaoa-shared-buffer \
// RUN:   '--doqaoa-warmstart-scheduler=warmstart-epochs=5' \
// RUN:   --doqaoa-aggregate-min \
// RUN:   --split-input-file 2>/dev/null | FileCheck %s --check-prefix=FORCED

// ============================================================
// Test 1: 4-cycle m=1, K=1 — mode 0 rep only, mode 1 direct copy.
//   All energies NaN → best_k=0, candidates_evaluated=0.
// ============================================================

// AGG-LABEL: func @cycle_agg
// AGG:       quantum.freeze_partition
// AGG-SAME:  agg_best_bitstring = array<i32: 0>
// AGG-SAME:  agg_best_k = 0 : i32
// AGG-SAME:  agg_candidates_evaluated = 0 : i32

// NOPRIOR-LABEL: func @cycle_agg
// NOPRIOR:   quantum.freeze_partition
// NOPRIOR-SAME: agg_best_bitstring = array<i32: 0>
// NOPRIOR-SAME: agg_best_k = 0 : i32

// FORCED-LABEL: func @cycle_agg
// FORCED:    quantum.freeze_partition
// FORCED-SAME: agg_best_bitstring
// FORCED-SAME: agg_best_k
// FORCED-SAME: agg_candidates_evaluated

func.func @cycle_agg() {
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
// Test 2: K4 m=1, biased — forces warm-start under FORCED prefix.
//   With finite warm-start energy, candidates_evaluated = 1.
//   Bitstring has 1 element (m=1).
// ============================================================

// AGG-LABEL: func @k4_agg
// AGG:       quantum.freeze_partition
// AGG-SAME:  agg_best_bitstring
// AGG-SAME:  agg_best_k

// NOPRIOR-LABEL: func @k4_agg
// NOPRIOR:   quantum.freeze_partition
// NOPRIOR-SAME: agg_best_k = 0 : i32

// FORCED-LABEL: func @k4_agg
// FORCED:    quantum.freeze_partition
// FORCED-SAME: agg_best_bitstring

func.func @k4_agg() {
    %p = quantum.freeze_partition {
        hotspot_count   = 1 : i32,
        hotspot_indices = array<i32: 0>,
        h_quad = #quantum.dense_graph<4, dense<[
            [ 0.0, -1.0, -1.0, -1.0],
            [-1.0,  0.0, -1.0, -1.0],
            [-1.0, -1.0,  0.0, -1.0],
            [-1.0, -1.0, -1.0,  0.0]]> : tensor<4x4xf64>>,
        h_lin  = dense<[0.8, -0.8, 0.8, -0.8]> : tensor<4xf64>
    } : !quantum.partition<4, 1>
    func.return
}

// -----

// ============================================================
// Test 3: Sparse m=2, 4 sub-problems.
//   All NaN energies by default → best_k=0, bitstring length=2.
// ============================================================

// AGG-LABEL: func @sparse_agg
// AGG:       quantum.freeze_partition
// AGG-SAME:  agg_best_bitstring = array<i32: 0, 0>
// AGG-SAME:  agg_best_k = 0 : i32
// AGG-SAME:  agg_candidates_evaluated = 0 : i32

// NOPRIOR-LABEL: func @sparse_agg
// NOPRIOR:   quantum.freeze_partition
// NOPRIOR-SAME: agg_best_bitstring = array<i32: 0, 0>
// NOPRIOR-SAME: agg_best_k = 0 : i32

func.func @sparse_agg() {
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
