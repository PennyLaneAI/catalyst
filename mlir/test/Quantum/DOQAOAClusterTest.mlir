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

// Check that cluster_k and cluster_assignments attributes are written.
// RUN: quantum-opt %s --doqaoa-landscape-overlap --split-input-file 2>/dev/null | FileCheck %s --check-prefix=ATTR

// ============================================================
// Complete graph K4, m=1: q≈1.0 → K=1, all in cluster 0.
// ============================================================

// ATTR-LABEL: func @complete_k4_m1
// ATTR:       quantum.freeze_partition
// ATTR-SAME:  cluster_assignments = array<i32: 0, 0>
// ATTR-SAME:  cluster_k = 1

func.func @complete_k4_m1() {
    %p = quantum.freeze_partition {
        hotspot_count   = 1 : i32,
        hotspot_indices = array<i32: 0>,
        h_quad = #quantum.dense_graph<4, dense<[
            [0.0, -0.5, -0.5, -0.5],
            [-0.5, 0.0, -0.5, -0.5],
            [-0.5, -0.5, 0.0, -0.5],
            [-0.5, -0.5, -0.5, 0.0]]> : tensor<4x4xf64>>,
        h_lin  = dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf64>
    } : !quantum.partition<4, 1>
    func.return
}

// -----

// ============================================================
// 4-cycle MaxCut m=1: q≈1.0 → concentrated → K=1.
// Symmetric freeze → both sub-problems in cluster 0.
// ============================================================

// ATTR-LABEL: func @cycle_m1_concentrated
// ATTR:       quantum.freeze_partition
// ATTR-SAME:  cluster_assignments = array<i32: 0, 0>
// ATTR-SAME:  cluster_k = 1

func.func @cycle_m1_concentrated() {
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
// 4-cycle MaxCut m=2: q=0.586 < 0.9 → fragmented → K-means elbow.
// cluster_k and cluster_assignments are written (exact value
// depends on elbow criterion — just check presence).
// ============================================================

// ATTR-LABEL: func @cycle_m2_fragmented
// ATTR:       quantum.freeze_partition
// ATTR-SAME:  cluster_assignments = array<i32: 1, 0, 0, 1>
// ATTR-SAME:  cluster_k = 2

func.func @cycle_m2_fragmented() {
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
// Missing h_quad: pass skips gracefully (no cluster attrs).
// ============================================================

// ATTR-LABEL: func @no_h_quad_skip
// ATTR:       quantum.freeze_partition
// ATTR-NOT:   cluster_k

func.func @no_h_quad_skip() {
    %p = quantum.freeze_partition {
        hotspot_count   = 1 : i32,
        hotspot_indices = array<i32: 0>
    } : !quantum.partition<4, 1>
    func.return
}

// -----

// ============================================================
// Sparse path graph: both attributes present.
// ============================================================

// ATTR-LABEL: func @sparse_cluster
// ATTR:       quantum.freeze_partition
// ATTR-SAME:  cluster_assignments = array<i32: 0, 0, 0, 0>
// ATTR-SAME:  cluster_k = 1

func.func @sparse_cluster() {
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
