// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt %s --lower-gradients=split --split-input-file | FileCheck %s

// CHECK-LABEL: func.func private @straight_line.qsplit(%arg0: tensor<?xf64>, %arg1: tensor<?xindex>, %arg2: tensor<?xi64>) -> tensor<f64>
func.func private @straight_line(%arg0: f64) -> tensor<f64> attributes {qnode, diff_method = "parameter-shift"} {
    // CHECK-NEXT: [[idx1:%.+]] = index.constant 1
    // CHECK-NEXT: [[idx0:%.+]] = index.constant 0
    // CHECK-NEXT: [[paramCounter:%.+]] = memref.alloca() : memref<index>
    // CHECK: memref.store [[idx0]], [[paramCounter]]

    %0 = quantum.alloc(1) : !quantum.reg
    %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
    // CHECK: [[pidx:%.+]] = memref.load [[paramCounter]]
    // CHECK-NEXT: [[param:%.+]] = tensor.extract %arg0[[[pidx]]]
    // CHECK-NEXT: [[pidxNext:%.+]] = index.add [[pidx]], [[idx1]]
    // CHECK-NEXT: quantum.custom "RZ"([[param]])
    // CHECK-NEXT: memref.store [[pidxNext]], [[paramCounter]]
    %2 = quantum.custom "RZ"(%arg0) %1 : !quantum.bit
    %3 = quantum.insert %0[1], %2 : !quantum.reg, !quantum.bit
    %4 = quantum.namedobs %2[PauliZ] : !quantum.obs
    %5 = quantum.expval %4 : f64
    %6 = tensor.from_elements %5 : tensor<f64>
    quantum.dealloc %0 : !quantum.reg
    return %6 : tensor<f64>
}

func.func @dstraight_line(%arg0: f64) {
    gradient.grad "ps" @straight_line(%arg0) : (f64) -> tensor<f64>
    return
}

// -----

// CHECK-LABEL: func.func private @for_loop.qsplit(%arg0: tensor<?xf64>, %arg1: tensor<?xindex>, %arg2: tensor<?xi64>) -> tensor<f64>
func.func private @for_loop(%start: index, %stop: index, %step: index, %arg0: tensor<f64>) -> tensor<f64> attributes {qnode, diff_method = "parameter-shift"} {
    %0 = quantum.alloc(4) : !quantum.reg
    %cst = arith.constant dense<2.000000e+00> : tensor<f64>
    %cst_0 = arith.constant dense<3.1415926535897931> : tensor<f64>
    %cst_1 = arith.constant dense<3.400000e+00> : tensor<f64>
    %1:2 = scf.for %arg1 = %start to %stop step %step iter_args(%arg2 = %arg0, %arg3 = %0) -> (tensor<f64>, !quantum.reg) {
        %5 = arith.index_cast %arg1 : index to i64
        %12 = quantum.extract %arg3[%5] : !quantum.reg -> !quantum.bit
        %extracted = tensor.extract %arg2[] : tensor<f64>
        %13 = quantum.custom "RX"(%extracted) %12 : !quantum.bit
        %14 = quantum.insert %arg3[0], %13 : !quantum.reg, !quantum.bit
        %15 = arith.addf %arg2, %cst_0 : tensor<f64>
        scf.yield %15, %14 : tensor<f64>, !quantum.reg
    }

    %2 = quantum.extract %1#1[0] : !quantum.reg -> !quantum.bit
    %3 = quantum.namedobs %2[PauliZ] : !quantum.obs
    %4 = quantum.expval %3 : f64
    quantum.dealloc %0 : !quantum.reg
    %5 = tensor.from_elements %4 : tensor<f64>
    return %5 : tensor<f64>
}

func.func @dfor_loop() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant dense<0.2> : tensor<f64>
    gradient.grad "ps" @for_loop(%c0, %c4, %c1, %cst) {diffArgIndices = dense<3> : tensor<i64>} : (index, index, index, tensor<f64>) -> tensor<f64>
    return
}
