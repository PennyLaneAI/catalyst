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

// RUN: quantum-opt %s --doqaoa-landscape-overlap --split-input-file | FileCheck %s

// ============================================================
// Exact path: 4-qubit MaxCut, m=2 (N_free=2 ≤ 20).
// Statevector is 2^2 = 4 amplitudes.
// Attributes must be written.
// ============================================================

// CHECK-LABEL: func @exact_path_m2
// CHECK:       quantum.freeze_partition
// CHECK-SAME:  landscape_overlap_q
// CHECK-SAME:  recommended_k

func.func @exact_path_m2() {
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

// -----

// ============================================================
// Exact path: Ising with strong linear bias (h=5.0 >> J=0.1).
// Strong bias → near-identical landscapes → K=1.
// ============================================================

// CHECK-LABEL: func @exact_strong_bias_k1
// CHECK:       quantum.freeze_partition
// CHECK-SAME:  recommended_k = 1

func.func @exact_strong_bias_k1() {
    %p = quantum.freeze_partition {
        hotspot_count   = 1 : i32,
        hotspot_indices = array<i32: 0>,
        h_quad = #quantum.dense_graph<4, dense<[
            [0.0, -0.1, 0.0, 0.0],
            [-0.1, 0.0, -0.1, 0.0],
            [0.0, -0.1, 0.0, -0.1],
            [0.0, 0.0, -0.1, 0.0]]> : tensor<4x4xf64>>,
        h_lin  = dense<[5.0, 0.0, 0.0, -5.0]> : tensor<4xf64>
    } : !quantum.partition<4, 1>
    func.return
}

// -----

// ============================================================
// Exact path: sparse graph attribute (path graph 4 nodes).
// Backend must handle SparseGraphAttr correctly.
// ============================================================

// CHECK-LABEL: func @exact_sparse_path
// CHECK:       quantum.freeze_partition
// CHECK-SAME:  landscape_overlap_q
// CHECK-SAME:  recommended_k

func.func @exact_sparse_path() {
    %p = quantum.freeze_partition {
        hotspot_count   = 1 : i32,
        hotspot_indices = array<i32: 1>,
        h_quad = #quantum.sparse_graph<4, 3,
            [0, 1, 2], [1, 2, 3],
            dense<[-0.5, -0.5, -0.5]> : tensor<3xf64>>,
        h_lin  = dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf64>
    } : !quantum.partition<4, 1>
    func.return
}

// -----

// ============================================================
// Cache test: two identical freeze_partition ops in one function.
// Both must get identical landscape_overlap_q (cache hit on 2nd).
// ============================================================

// CHECK-LABEL: func @cache_hit_identical_graphs
// CHECK:       quantum.freeze_partition
// CHECK-SAME:  landscape_overlap_q
// CHECK:       quantum.freeze_partition
// CHECK-SAME:  landscape_overlap_q

func.func @cache_hit_identical_graphs() {
    %p1 = quantum.freeze_partition {
        hotspot_count   = 1 : i32,
        hotspot_indices = array<i32: 0>,
        h_quad = #quantum.dense_graph<3, dense<[
            [0.0, -0.5, -0.5],
            [-0.5, 0.0, -0.5],
            [-0.5, -0.5, 0.0]]> : tensor<3x3xf64>>,
        h_lin  = dense<[0.0, 0.0, 0.0]> : tensor<3xf64>
    } : !quantum.partition<3, 1>
    %p2 = quantum.freeze_partition {
        hotspot_count   = 1 : i32,
        hotspot_indices = array<i32: 0>,
        h_quad = #quantum.dense_graph<3, dense<[
            [0.0, -0.5, -0.5],
            [-0.5, 0.0, -0.5],
            [-0.5, -0.5, 0.0]]> : tensor<3x3xf64>>,
        h_lin  = dense<[0.0, 0.0, 0.0]> : tensor<3xf64>
    } : !quantum.partition<3, 1>
    func.return
}

// -----

// ============================================================
// No h_quad: pass skips gracefully, no attribute written.
// ============================================================

// CHECK-LABEL: func @no_hquad_skipped
// CHECK:       quantum.freeze_partition
// CHECK-NOT:   landscape_overlap_q

func.func @no_hquad_skipped() {
    %p = quantum.freeze_partition {
        hotspot_count   = 1 : i32,
        hotspot_indices = array<i32: 0>
    } : !quantum.partition<4, 1>
    func.return
}
