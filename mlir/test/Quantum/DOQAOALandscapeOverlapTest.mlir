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
// 4-node cycle MaxCut: pure ZZ, m=2.
// q < 0 (anti-correlated landscapes) → recommended_k = 4 (2^m).
// Key check: landscape_overlap_q attribute is written.
// ============================================================

// CHECK-LABEL: func @maxcut_4cycle_m2
// CHECK:       quantum.freeze_partition
// CHECK-SAME:  landscape_overlap_q
// CHECK-SAME:  recommended_k = 4

func.func @maxcut_4cycle_m2() {
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
// Single hotspot (m=1): 2 sub-problems.
// Freezing qubit 0 to ±1 flips effective bias on neighbours
// symmetrically → q ≈ 1.0 ≥ threshold → recommended_k = 1.
// ============================================================

// CHECK-LABEL: func @maxcut_4cycle_m1
// CHECK:       quantum.freeze_partition
// CHECK-SAME:  landscape_overlap_q
// CHECK-SAME:  recommended_k = 1

func.func @maxcut_4cycle_m1() {
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
// Ising with strong linear bias (h >> J).
// Sub-problems with opposite frozen spins produce nearly
// identical landscapes → q ≈ 0.99 ≥ threshold=0.9 → K=1.
// ============================================================

// CHECK-LABEL: func @ising_with_bias
// CHECK:       quantum.freeze_partition
// CHECK-SAME:  landscape_overlap_q
// CHECK-SAME:  recommended_k = 1

func.func @ising_with_bias() {
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
// No h_quad attribute: pass emits warning, skips, no attributes added.
// ============================================================

// CHECK-LABEL: func @no_h_quad
// CHECK:       quantum.freeze_partition
// CHECK-NOT:   landscape_overlap_q

func.func @no_h_quad() {
    %p = quantum.freeze_partition {
        hotspot_count   = 1 : i32,
        hotspot_indices = array<i32: 0>
    } : !quantum.partition<4, 1>
    func.return
}

// -----

// ============================================================
// Sparse graph attribute path: 4-node path graph.
// landscape_overlap_q and recommended_k are written.
// ============================================================

// CHECK-LABEL: func @sparse_path_graph
// CHECK:       quantum.freeze_partition
// CHECK-SAME:  landscape_overlap_q
// CHECK-SAME:  recommended_k

func.func @sparse_path_graph() {
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
