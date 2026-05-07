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

// Tests for --doqaoa-depth-check (Phase 3, Task 9).
// Regression gate for Table IV of arXiv:2602.21689v1.
//
// DEPTH prefix: default (no bound).
// RUN: quantum-opt %s \
// RUN:   --doqaoa-depth-check \
// RUN:   --split-input-file 2>/dev/null | FileCheck %s --check-prefix=DEPTH

// BOUND prefix: expected-max-cnots=10 (passes all sections, including K4 with 6 CNOTs).
// RUN: quantum-opt %s \
// RUN:   '--doqaoa-depth-check=expected-max-cnots=10' \
// RUN:   --split-input-file 2>/dev/null | FileCheck %s --check-prefix=BOUND

// FAIL prefix: expected-max-cnots=2 (fails for any section with cnots>2).
//   quantum-opt exits non-zero; use `not` to invert so pipefail does not abort FileCheck.
//   The failed section's IR is not printed; check error message without LABEL anchor.
// RUN: not quantum-opt %s \
// RUN:   '--doqaoa-depth-check=expected-max-cnots=2' \
// RUN:   --split-input-file 2>&1 | FileCheck %s --check-prefix=FAIL

// REMARK prefix: check remark message content.
//   Remarks go to stderr; with 2>&1 they appear before the MLIR stdout in the pipe.
//   Use bare REMARK: (no LABEL) to find the remark anywhere in the merged output.
// RUN: quantum-opt %s \
// RUN:   '--doqaoa-depth-check=expected-max-cnots=100' \
// RUN:   --split-input-file 2>&1 | FileCheck %s --check-prefix=REMARK

// ============================================================
// Test 1: 4-cycle m=1, 3 free qubits.
//   Free edges: 1-2, 2-3, 1-3? No — cycle is 0-1-2-3-0.
//   With hotspot=0 frozen: free={1,2,3}.
//   Free-free edges: 1-2 (yes), 2-3 (yes), 3-0 (0 frozen, skip).
//   |E_free|=2, cnot_count=4.
// ============================================================

// DEPTH-LABEL: func @cycle_depth
// DEPTH:       quantum.freeze_partition
// DEPTH-SAME:  depth_cnot_counts = array<i32: 4, 4>
// DEPTH-SAME:  depth_max_cnots = 4 : i32
// DEPTH-SAME:  depth_regression_ok = 1 : i32

// BOUND-LABEL: func @cycle_depth
// BOUND:       quantum.freeze_partition
// BOUND-SAME:  depth_regression_ok = 1 : i32

// REMARK: remark: doqaoa-depth-check: max_cnots=4 free_edges=2

func.func @cycle_depth() {
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
// Test 2: K4 m=1, 6 free-free CNOTs.
//   Free={1,2,3}: edges 1-2, 1-3, 2-3 → |E_free|=3, cnots=6.
//   With bound=2: regression failure.
// ============================================================

// DEPTH-LABEL: func @k4_depth
// DEPTH:       quantum.freeze_partition
// DEPTH-SAME:  depth_max_cnots = 6 : i32
// DEPTH-SAME:  depth_regression_ok = 1 : i32

// BOUND-LABEL: func @k4_depth
// BOUND:       quantum.freeze_partition
// BOUND-SAME:  depth_regression_ok = 1 : i32

// FAIL: error: doqaoa-depth-check: max_cnots={{.*}} exceeds expected-max-cnots=2

func.func @k4_depth() {
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
// Test 3: Sparse path m=2 (Table IV: 10-node MaxCut m=2).
//   Hotspots {1,2} frozen, free={0,3}.
//   Path edges: 0-1(frozen), 1-2(frozen), 2-3(frozen) — no free-free edges.
//   |E_free|=0, cnot_count=0.
//   regression_ok=1 even with tight bound.
// ============================================================

// DEPTH-LABEL: func @sparse_depth
// DEPTH:       quantum.freeze_partition
// DEPTH-SAME:  depth_cnot_counts = array<i32: 0, 0, 0, 0>
// DEPTH-SAME:  depth_max_cnots = 0 : i32
// DEPTH-SAME:  depth_regression_ok = 1 : i32

// BOUND-LABEL: func @sparse_depth
// BOUND:       quantum.freeze_partition
// BOUND-SAME:  depth_regression_ok = 1 : i32

func.func @sparse_depth() {
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
