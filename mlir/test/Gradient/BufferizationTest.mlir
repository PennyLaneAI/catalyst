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

func.func private @circuit2(%arg0: f64)

// CHECK-LABEL: @backprop
func.func @backprop(%arg0: f64, %arg1: tensor<?xf64>) {

    // CHECK:   [[dim:%.+]] = memref.dim
    // CHECK:   [[calleeRes:%.+]] = memref.alloc([[dim]]) : memref<?xf64>
    // CHECK:   gradient.backprop @circuit2({{%.+}}) callee_out([[calleeRes]] : memref<?xf64>) cotangents({{%.+}} : memref<?xf64>) {diffArgIndices = dense<0> : tensor<1xindex>} : (f64) -> f64
    %grad = gradient.backprop @circuit2(%arg0) cotangents(%arg1: tensor<?xf64>) {diffArgIndices = dense<0> : tensor<1xindex>}: (f64) -> f64
    return
}

// -----

func.func private @circuit3(%arg0: tensor<?x2xf64>)

// CHECK-LABEL: @backprop2
func.func @backprop2(%arg0: tensor<?x2xf64>, %arg1: tensor<?xf64>) {

    // CHECK:   [[argShadow:%.+]] = memref.alloc(%dim) : memref<?x2xf64>
    // CHECK:   [[dim:%.+]] = memref.dim
    // CHECK:   [[calleeRes:%.+]] = memref.alloc([[dim]]) : memref<?xf64>
    // CHECK:   gradient.backprop @circuit3({{%.+}}) grad_out([[argShadow]] : memref<?x2xf64>) callee_out([[calleeRes]] : memref<?xf64>) cotangents({{%.+}} : memref<?xf64>) {diffArgIndices = dense<0> : tensor<1xindex>} : (memref<?x2xf64>) -> ()
    %grad = gradient.backprop @circuit3(%arg0) cotangents(%arg1: tensor<?xf64>) {diffArgIndices = dense<0> : tensor<1xindex>}: (tensor<?x2xf64>) -> tensor<?x2xf64>
    return
}

// -----

func.func private @circuit4(%arg0: tensor<10xf64>, %arg1: tensor<2xf64>)

// CHECK-LABEL: @backprop3
func.func @backprop3(%arg0: tensor<10xf64>, %arg1: tensor<2xf64>, %arg2: tensor<?xf64>) {

    // CHECK:   [[argShadow1:%.+]] = memref.alloc() : memref<10xf64>
    // CHECK:   [[argShadow2:%.+]] = memref.alloc() : memref<2xf64>
    // CHECK:   [[dim:%.+]] = memref.dim
    // CHECK:   [[calleeRes:%.+]] = memref.alloc([[dim]]) : memref<?xf64>
    // CHECK:   gradient.backprop @circuit4({{%.+}}, {{%.+}}) grad_out([[argShadow1]], [[argShadow2]] : memref<10xf64>, memref<2xf64>) callee_out([[calleeRes]] : memref<?xf64>) cotangents({{%.+}} : memref<?xf64>) {diffArgIndices = dense<[0, 1]> : tensor<2xindex>} : (memref<10xf64>, memref<2xf64>) -> ()
    %grad:2 = gradient.backprop @circuit4(%arg0, %arg1) cotangents(%arg2: tensor<?xf64>) {diffArgIndices = dense<[0, 1]> : tensor<2xindex>}: (tensor<10xf64>, tensor<2xf64>) -> (tensor<10xf64>, tensor<2xf64>)
    return
}

// -----

// CHECK-LABEL: @test0
module @test0 {

  func.func private @fwd(memref<f64>) -> (memref<f64>, memref<f64>)

  // CHECK-LABEL: gradient.forward @fwd.fwd(
  // CHECK-SAME:[[inp0:%.+]]: memref<f64>, [[inpshd0:%.+]]: memref<f64>, [[out0:%.+]]: memref<f64>, [[outshd0:%.+]]: memref<f64>) -> memref<f64>
  gradient.forward @fwd.fwd(tensor<f64>) -> (tensor<f64>, tensor<f64>) attributes {implementation = @fwd, argc = 1: i64, resc = 1 : i64, tape = 1: i64}
  // CHECK: [[call0:%.+]]:2 = func.call @fwd([[inp0]])
  // CHECK: memref.copy [[call0]]#0, [[out0]]
  // CHECK: gradient.return [[call0]]#1

}

