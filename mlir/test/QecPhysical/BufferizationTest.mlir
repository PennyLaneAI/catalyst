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

// RUN: quantum-opt --one-shot-bufferize --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @assemble_tanner()
func.func @assemble_tanner() -> !qecp.tanner_graph<12, 8, i32> {
    // CHECK: [[row_idx:%.+]] = memref.get_global {{.*}} : memref<12xi32>
    // CHECK: [[col_ptr:%.+]] = memref.get_global {{.*}} : memref<8xi32>
    %row_idx = arith.constant dense<[0, 0, 1, 0, 1, 2, 0, 2, 1, 1, 2, 2]> : tensor<12xi32>
    %col_ptr = arith.constant dense<[0, 1, 3, 6, 8, 9, 11, 12]> : tensor<8xi32>

    //      CHECK: [[tanner:%.+]] = qecp.assemble_tanner [[row_idx]], [[col_ptr]] :
    // CHECK-SAME:   memref<12xi32>, memref<8xi32> -> !qecp.tanner_graph<12, 8, i32>
    %0 = qecp.assemble_tanner %row_idx, %col_ptr : tensor<12xi32>, tensor<8xi32> -> !qecp.tanner_graph<12, 8, i32>

    // CHECK: return [[tanner]] : !qecp.tanner_graph<12, 8, i32>
    func.return %0 : !qecp.tanner_graph<12, 8, i32>
}

// -----

// CHECK-LABEL: func.func @decode_esm_css(
//  CHECK-SAME:     [[esm:%.+]]: tensor<3xi1>
//  CHECK-SAME:     [[tanner:%.+]]: !qecp.tanner_graph<12, 8, i32>
func.func @decode_esm_css(
    %esm : tensor<3xi1>,
    %tanner : !qecp.tanner_graph<12, 8, i32>
) -> tensor<1xindex> {
    // CHECK-DAG: [[esm_buf:%.+]] = bufferization.to_buffer [[esm]] : tensor<3xi1> to memref<3xi1>
    // CHECK-DAG: [[idx_buf:%.+]] = memref.alloc() : memref<1xindex>

    //      CHECK: qecp.decode_esm_css([[tanner]] : !qecp.tanner_graph<12, 8, i32>) [[esm_buf]]
    // CHECK-SAME:   in([[idx_buf]] : memref<1xindex>) : memref<3xi1>
    %0 = qecp.decode_esm_css(%tanner : !qecp.tanner_graph<12, 8, i32>) %esm : tensor<3xi1> -> tensor<1xindex>

    // CHECK: [[idx_out:%.+]] = bufferization.to_tensor [[idx_buf]] : memref<1xindex> to tensor<1xindex>
    // CHECK: return [[idx_out]] : tensor<1xindex>
    func.return %0 : tensor<1xindex>
}

// -----

// CHECK-LABEL: func.func @decode_physical_meas(
//  CHECK-SAME:     [[pmeas:%.+]]: tensor<7xi1>
func.func @decode_physical_meas(%phys_meas : tensor<7xi1>) -> tensor<1xi1> {
    // CHECK-DAG: [[pmeas_buf:%.+]] = bufferization.to_buffer [[pmeas]] : tensor<7xi1> to memref<7xi1>
    // CHECK-DAG: [[lmeas_buf:%.+]] = memref.alloc() : memref<1xi1>

    // CHECK: qecp.decode_physical_meas [[pmeas_buf]] in([[lmeas_buf]] : memref<1xi1>) : memref<7xi1>
    %0 = qecp.decode_physical_meas %phys_meas : tensor<7xi1> -> tensor<1xi1>

    // CHECK: [[lmeas_out:%.+]] = bufferization.to_tensor [[lmeas_buf]] : memref<1xi1> to tensor<1xi1>
    // CHECK: return [[lmeas_out]] : tensor<1xi1>
    func.return %0 : tensor<1xi1>
}
