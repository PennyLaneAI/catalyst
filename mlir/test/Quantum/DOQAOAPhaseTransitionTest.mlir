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

// Check that s_eff attribute is written to every op (IR only, no stderr).
// RUN: quantum-opt %s --doqaoa-landscape-overlap --split-input-file 2>/dev/null | FileCheck %s --check-prefix=ATTR

// Check that fragmented-regime warnings fire for path/sparse graphs.
// RUN: quantum-opt %s --doqaoa-landscape-overlap --split-input-file 2>&1 | FileCheck %s --check-prefix=WARN

// ============================================================
// Complete graph K4: diameter=1 → s_eff=1.0 > sc=0.6.
// No warning.  s_eff attribute written.
// ============================================================

// ATTR-LABEL: func @complete_graph_k4
// ATTR:       quantum.freeze_partition
// ATTR-SAME:  s_eff

func.func @complete_graph_k4() {
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
// 4-cycle graph: diameter=2 → s_eff=0.667 > sc=0.6.
// Concentrated regime — no warning.
// ============================================================

// ATTR-LABEL: func @cycle_4_concentrated
// ATTR:       quantum.freeze_partition
// ATTR-SAME:  s_eff

func.func @cycle_4_concentrated() {
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
// Path graph P5: 0-1-2-3-4, diameter=4 → s_eff=0.4 < sc=0.6.
// Fragmented regime → warning must be emitted.
// ============================================================

// ATTR-LABEL: func @path_5_fragmented
// ATTR:       quantum.freeze_partition
// ATTR-SAME:  s_eff

// WARN: warning: doqaoa-landscape-overlap: fragmented landscape regime
// WARN-SAME: s_eff=0.400
// WARN-SAME: diameter=4

func.func @path_5_fragmented() {
    %p = quantum.freeze_partition {
        hotspot_count   = 1 : i32,
        hotspot_indices = array<i32: 2>,
        h_quad = #quantum.dense_graph<5, dense<[
            [0.0, -0.5, 0.0, 0.0, 0.0],
            [-0.5, 0.0, -0.5, 0.0, 0.0],
            [0.0, -0.5, 0.0, -0.5, 0.0],
            [0.0, 0.0, -0.5, 0.0, -0.5],
            [0.0, 0.0, 0.0, -0.5, 0.0]]> : tensor<5x5xf64>>,
        h_lin  = dense<[0.0, 0.0, 0.0, 0.0, 0.0]> : tensor<5xf64>
    } : !quantum.partition<5, 1>
    func.return
}

// -----

// ============================================================
// Path graph P4: 0-1-2-3, diameter=3 → s_eff=0.5 < sc=0.6.
// Fragmented regime → warning emitted.
// ============================================================

// ATTR-LABEL: func @path_4_fragmented
// ATTR:       quantum.freeze_partition
// ATTR-SAME:  s_eff

// WARN: warning: doqaoa-landscape-overlap: fragmented landscape regime
// WARN-SAME: diameter=3

func.func @path_4_fragmented() {
    %p = quantum.freeze_partition {
        hotspot_count   = 1 : i32,
        hotspot_indices = array<i32: 0>,
        h_quad = #quantum.dense_graph<4, dense<[
            [0.0, -0.5, 0.0, 0.0],
            [-0.5, 0.0, -0.5, 0.0],
            [0.0, -0.5, 0.0, -0.5],
            [0.0, 0.0, -0.5, 0.0]]> : tensor<4x4xf64>>,
        h_lin  = dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf64>
    } : !quantum.partition<4, 1>
    func.return
}

// -----

// ============================================================
// Sparse path graph P4 via SparseGraphAttr: same structure
// as above, diameter=3 → s_eff=0.5 < sc=0.6 → warning.
// ============================================================

// ATTR-LABEL: func @sparse_path_fragmented
// ATTR:       quantum.freeze_partition
// ATTR-SAME:  s_eff

// WARN: warning: doqaoa-landscape-overlap: fragmented landscape regime
// WARN-SAME: sc=0.600

func.func @sparse_path_fragmented() {
    %p = quantum.freeze_partition {
        hotspot_count   = 1 : i32,
        hotspot_indices = array<i32: 0>,
        h_quad = #quantum.sparse_graph<4, 3,
            [0, 1, 2], [1, 2, 3],
            dense<[-0.5, -0.5, -0.5]> : tensor<3xf64>>,
        h_lin  = dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf64>
    } : !quantum.partition<4, 1>
    func.return
}
