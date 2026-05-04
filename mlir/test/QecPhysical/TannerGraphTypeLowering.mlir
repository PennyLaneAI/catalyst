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

// RUN: quantum-opt %s \
// RUN:   --convert-arith-to-llvm \
// RUN:   --one-shot-bufferize \
// RUN:   --convert-qecp-to-llvm \
// RUN:   --reconcile-unrealized-casts \
// RUN:   --split-input-file -verify-diagnostics \
// RUN: | FileCheck %s

// CHECK-DAG: llvm.func @__catalyst__qecp__assemble_tanner_graph_int32(!llvm.ptr, !llvm.ptr, !llvm.ptr)

// CHECK-LABEL: unit_test_tanner_graph_lower
func.func @unit_test_tanner_graph_lower() {
    %row_idx = arith.constant dense<[0, 0, 1, 0, 1, 2, 0, 2, 1, 1, 2, 2]> : tensor<12xi32>
    %col_ptr = arith.constant dense<[0, 1, 3, 6, 8, 9, 11, 12]> : tensor<8xi32>
    // CHECK: llvm.call @__catalyst__qecp__assemble_tanner_graph_int32
    %0 = qecp.assemble_tanner %row_idx, %col_ptr : tensor<12xi32>, tensor<8xi32> -> !qecp.tanner_graph<12, 8, i32>
    func.return
}