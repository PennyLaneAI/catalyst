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

// RUN: quantum-opt %s --split-input-file | FileCheck %s

// ============================================================
// Dense graph: 2-node graph, weight matrix [[0, -0.5],[-0.5, 0]]
// ============================================================

// CHECK-LABEL: func @dense_2node
// CHECK:       #quantum.dense_graph<2,
// CHECK-SAME:  dense<{{.*}}> : tensor<2x2xf64>
func.func @dense_2node() -> i1 {
    %c = arith.constant {
        graph = #quantum.dense_graph<2,
            dense<[[0.0, -0.5], [-0.5, 0.0]]> : tensor<2x2xf64>>
    } 0 : i1
    return %c : i1
}

// -----

// ============================================================
// Dense graph: 4-node cycle graph (MaxCut weights)
// ============================================================

// CHECK-LABEL: func @dense_4node_cycle
// CHECK:       #quantum.dense_graph<4,
func.func @dense_4node_cycle() -> i1 {
    %c = arith.constant {
        graph = #quantum.dense_graph<4,
            dense<[[0.0, -0.5, 0.0, -0.5],
                   [-0.5, 0.0, -0.5, 0.0],
                   [0.0, -0.5, 0.0, -0.5],
                   [-0.5, 0.0, -0.5, 0.0]]> : tensor<4x4xf64>>
    } 0 : i1
    return %c : i1
}

// -----

// ============================================================
// Sparse graph: 4-node cycle, upper-triangle COO
// Edges: (0,1), (1,2), (2,3), (0,3) each with weight -0.5
// ============================================================

// CHECK-LABEL: func @sparse_4node_cycle
// CHECK:       #quantum.sparse_graph<4, 4,
// CHECK-SAME:  [0, 0, 1, 2],
// CHECK-SAME:  [1, 3, 2, 3],
// CHECK-SAME:  dense<
func.func @sparse_4node_cycle() -> i1 {
    %c = arith.constant {
        graph = #quantum.sparse_graph<4, 4,
            [0, 0, 1, 2],
            [1, 3, 2, 3],
            dense<[-0.5, -0.5, -0.5, -0.5]> : tensor<4xf64>>
    } 0 : i1
    return %c : i1
}

// -----

// ============================================================
// Sparse graph: single edge
// ============================================================

// CHECK-LABEL: func @sparse_single_edge
// CHECK:       #quantum.sparse_graph<10, 1,
// CHECK-SAME:  [2],
// CHECK-SAME:  [7],
// CHECK-SAME:  dense<
func.func @sparse_single_edge() -> i1 {
    %c = arith.constant {
        graph = #quantum.sparse_graph<10, 1, [2], [7],
            dense<[-1.0]> : tensor<1xf64>>
    } 0 : i1
    return %c : i1
}
