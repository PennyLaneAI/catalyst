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

// RUN: quantum-opt %s --split-input-file --verify-diagnostics

// ============================================================
// DenseGraphAttr: weight matrix shape mismatch (3x3 for numNodes=4)
// ============================================================

func.func @dense_shape_mismatch() -> i1 {
    // expected-error@+1 {{dense_graph: weights shape [3x3] does not match numNodes=4}}
    %c = arith.constant {graph = #quantum.dense_graph<4, dense<[[0.0,-0.5,0.0],[-0.5,0.0,-0.5],[0.0,-0.5,0.0]]> : tensor<3x3xf64>>} 0 : i1
    return %c : i1
}

// -----

// ============================================================
// DenseGraphAttr: wrong element type (f32 instead of f64)
// ============================================================

func.func @dense_wrong_dtype() -> i1 {
    // expected-error@+1 {{dense_graph: weights element type must be f64}}
    %c = arith.constant {graph = #quantum.dense_graph<2, dense<[[0.0,-0.5],[-0.5,0.0]]> : tensor<2x2xf32>>} 0 : i1
    return %c : i1
}

// -----

// ============================================================
// DenseGraphAttr: rank-1 tensor instead of rank-2
// ============================================================

func.func @dense_wrong_rank() -> i1 {
    // expected-error@+1 {{dense_graph: weights must be a rank-2 tensor (NxN)}}
    %c = arith.constant {graph = #quantum.dense_graph<4, dense<[-0.5,-0.5,-0.5,-0.5]> : tensor<4xf64>>} 0 : i1
    return %c : i1
}

// -----

// ============================================================
// SparseGraphAttr: rowIndices length != numEdges
// ============================================================

func.func @sparse_row_length_mismatch() -> i1 {
    // expected-error@+1 {{sparse_graph: rowIndices length (1) != numEdges (3)}}
    %c = arith.constant {graph = #quantum.sparse_graph<5, 3, [0], [1, 2, 3], dense<[-0.5,-0.5,-0.5]> : tensor<3xf64>>} 0 : i1
    return %c : i1
}

// -----

// ============================================================
// SparseGraphAttr: col index out of range (col=10 >= numNodes=5)
// ============================================================

func.func @sparse_index_out_of_range() -> i1 {
    // expected-error@+1 {{sparse_graph: index out of range at edge 0}}
    %c = arith.constant {graph = #quantum.sparse_graph<5, 1, [0], [10], dense<[-0.5]> : tensor<1xf64>>} 0 : i1
    return %c : i1
}

// -----

// ============================================================
// SparseGraphAttr: lower-triangle entry (row > col) violates i < j
// ============================================================

func.func @sparse_lower_triangle() -> i1 {
    // expected-error@+1 {{sparse_graph: expected upper-triangle (i < j), got [3,1] at edge 0}}
    %c = arith.constant {graph = #quantum.sparse_graph<5, 1, [3], [1], dense<[-0.5]> : tensor<1xf64>>} 0 : i1
    return %c : i1
}

// -----

// ============================================================
// SparseGraphAttr: diagonal entry (i == j) violates i < j
// ============================================================

func.func @sparse_diagonal() -> i1 {
    // expected-error@+1 {{sparse_graph: expected upper-triangle (i < j), got [2,2] at edge 0}}
    %c = arith.constant {graph = #quantum.sparse_graph<5, 1, [2], [2], dense<[-0.5]> : tensor<1xf64>>} 0 : i1
    return %c : i1
}
