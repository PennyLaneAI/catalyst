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

// RUN: quantum-opt --gradient-bufferize --split-input-file %s | FileCheck %s

//////////////////////
// Native Gradients //
//////////////////////

func.func private @circuit(%arg0: f64)

func.func @adjoint(%arg0: f64, %arg1: index) {

    // CHECK:   [[alloc:%.+]] = memref.alloc(%arg1) : memref<?xf64>
    // CHECK:   gradient.adjoint @circuit(%arg0) size(%arg1) in([[alloc]] : memref<?xf64>) : (f64) -> ()
    %grad = gradient.adjoint @circuit(%arg0) size(%arg1) : (f64) -> tensor<?xf64>
    return
}

// -----

func.func private @circuit2(%arg0: tensor<f64>)

// CHECK-LABEL: @backprop
func.func @backprop(%arg0: tensor<f64>, %arg1: memref<?xf64>) {

    // CHECK:   [[alloc:%.+]] = memref.alloc() : memref<f64>
    // CHECK:   gradient.backprop @circuit2({{.*}}) qjacobian(%arg1 : memref<?xf64>) in([[alloc]] : memref<f64>) {diffArgIndices = dense<0> : tensor<1xindex>} : (memref<f64>) -> ()
    %grad = gradient.backprop @circuit2(%arg0) qjacobian(%arg1: memref<?xf64>) {diffArgIndices = dense<0> : tensor<1xindex>}: (tensor<f64>) -> tensor<f64>
    return
}

// -----

func.func private @circuit3(%arg0: tensor<3x2xf64>)

// CHECK-LABEL: @backprop2
func.func @backprop2(%arg0: tensor<3x2xf64>, %arg1: memref<?xf64>) {

    // CHECK:   [[alloc:%.+]] = memref.alloc() : memref<3x2xf64>
    // CHECK:   gradient.backprop @circuit3({{.*}}) qjacobian(%arg1 : memref<?xf64>) in([[alloc]] : memref<3x2xf64>) {diffArgIndices = dense<0> : tensor<1xindex>} : (memref<3x2xf64>) -> ()
    %grad = gradient.backprop @circuit3(%arg0) qjacobian(%arg1: memref<?xf64>) {diffArgIndices = dense<0> : tensor<1xindex>}: (tensor<3x2xf64>) -> tensor<3x2xf64>
    return
}

// -----

func.func private @circuit4(%arg0: tensor<10xf64>, %arg1: tensor<2xf64>)

// CHECK-LABEL: @backprop3
func.func @backprop3(%arg0: tensor<10xf64>, %arg1: tensor<2xf64>, %arg2: memref<?xf64>) {

    // CHECK:   [[alloc:%.+]] = memref.alloc() : memref<10xf64>
    // CHECK:   [[alloc_2:%.+]] = memref.alloc() : memref<2xf64>
    // CHECK:   gradient.backprop @circuit4({{.*}}, {{.*}}) qjacobian(%arg2 : memref<?xf64>) in([[alloc]], [[alloc_2]] : memref<10xf64>, memref<2xf64>) {diffArgIndices = dense<0> : tensor<1xindex>} : (memref<10xf64>, memref<2xf64>) -> ()
    %grad0, %grad1 = gradient.backprop @circuit4(%arg0, %arg1) qjacobian(%arg2: memref<?xf64>) {diffArgIndices = dense<0> : tensor<1xindex>}: (tensor<10xf64>, tensor<2xf64>) -> (tensor<10xf64>, tensor<2xf64>)
    return
}
