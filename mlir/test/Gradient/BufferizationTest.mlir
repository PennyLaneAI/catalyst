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
    // CHECK:   gradient.backprop @circuit2({{%.+}}) callee_out([[calleeRes]] : memref<?xf64>) cotangents({{%.+}} : memref<?xf64>) {diffArgIndices = dense<0> : tensor<1xindex>, resultSegmentSizes = array<i32: 0, 1>} : (f64) -> f64
    %grad = gradient.backprop @circuit2(%arg0) cotangents(%arg1: tensor<?xf64>) {diffArgIndices = dense<0> : tensor<1xindex>, resultSegmentSizes = array<i32: 0, 1>}: (f64) -> f64
    return
}

// -----

func.func private @circuit3(%arg0: tensor<?x2xf64>)

// CHECK-LABEL: @backprop2
func.func @backprop2(%arg0: tensor<?x2xf64>, %arg1: tensor<?xf64>) {

    // CHECK:   [[argShadow:%.+]] = memref.alloc(%dim) : memref<?x2xf64>
    // CHECK:   [[dim:%.+]] = memref.dim
    // CHECK:   [[calleeRes:%.+]] = memref.alloc([[dim]]) : memref<?xf64>
    // CHECK:   gradient.backprop @circuit3({{%.+}}) grad_out([[argShadow]] : memref<?x2xf64>) callee_out([[calleeRes]] : memref<?xf64>) cotangents({{%.+}} : memref<?xf64>) {diffArgIndices = dense<0> : tensor<1xindex>, resultSegmentSizes = array<i32: 0, 0>} : (memref<?x2xf64>) -> ()
    %grad = gradient.backprop @circuit3(%arg0) cotangents(%arg1: tensor<?xf64>) {diffArgIndices = dense<0> : tensor<1xindex>, resultSegmentSizes = array<i32: 0, 1>}: (tensor<?x2xf64>) -> tensor<?x2xf64>
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
    // CHECK:   gradient.backprop @circuit4({{%.+}}, {{%.+}}) grad_out([[argShadow1]], [[argShadow2]] : memref<10xf64>, memref<2xf64>) callee_out([[calleeRes]] : memref<?xf64>) cotangents({{%.+}} : memref<?xf64>) {diffArgIndices = dense<[0, 1]> : tensor<2xindex>, resultSegmentSizes = array<i32: 0, 0>} : (memref<10xf64>, memref<2xf64>) -> ()
    %grad:2 = gradient.backprop @circuit4(%arg0, %arg1) cotangents(%arg2: tensor<?xf64>) {diffArgIndices = dense<[0, 1]> : tensor<2xindex>, resultSegmentSizes = array<i32: 0, 2>}: (tensor<10xf64>, tensor<2xf64>) -> (tensor<10xf64>, tensor<2xf64>)
    return
}

// -----

// CHECK-LABEL: @test0
module @test0 {

  func.func private @fwd(tensor<f64>) -> (tensor<f64>, tensor<f64>)

  // CHECK-LABEL: gradient.forward @fwd.fwd(
  // CHECK:[[in0:%.+]]: memref<f64>, [[diff0:%.+]]: memref<f64>, [[out0:%.+]]: memref<f64>, [[cotang0:%.+]]: memref<f64>) -> memref<f64> 
  gradient.forward @fwd.fwd(tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>) attributes {implementation = @fwd, argc = 1: i64, resc = 1 : i64, tape = 1: i64}
  // CHECK: [[tensor0:%.+]] = bufferization.to_tensor [[in0]]
  // CHECK: [[outAndTape:%.+]]:2 = func.call @fwd([[tensor0]])
  // CHECK: [[outMemref:%.+]] = bufferization.to_memref [[outAndTape]]#0
  // CHECK: memref.copy [[outMemref]], [[out0]]
  // CHECK: [[tapeMemref:%.+]] = bufferization.to_memref [[outAndTape]]#1
  // CHECK: gradient.return {empty = false} [[tapeMemref]]

}

// -----

// CHECK-LABEL: @fwd_test_no_tape
module @fwd_test_no_tape {

  func.func private @fwd(%arg0: tensor<f64>) -> tensor<f64>

  // CHECK-LABEL: gradient.forward @fwd.fwd
  // CHECK-SAME:[[in0:%.+]]: memref<f64>, [[diff0:%.+]]: memref<f64>, [[out0:%.+]]: memref<f64>, [[cotang0:%.+]]: memref<f64>) attributes
  gradient.forward @fwd.fwd(tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) attributes {argc = 1 : i64, implementation = @fwd, resc = 1 : i64, tape = 0 : i64}
  // CHECK: [[tensor0:%.+]] = bufferization.to_tensor [[in0]]
  // CHECK: [[outTensor0:%.+]] = func.call @fwd([[tensor0]])
  // CHECK: [[outMemref0:%.+]] = bufferization.to_memref [[outTensor0]]
  // CHECK: memref.copy [[outMemref0]], [[out0]]
  // CHECK: gradient.return {empty = false}

}

// -----

// CHECK-LABEL: @test1
module @test1 {
  func.func private @rev(tensor<f64>, tensor<f64>) -> tensor<f64>

  // CHECK-LABEL: gradient.reverse @rev.rev
  // CHECK:[[in0:%.+]]: memref<f64>, [[diff0:%.+]]: memref<f64>, [[out0:%.+]]: memref<f64>, [[cotan0:%.+]]: memref<f64>, [[tape0:%.+]]: memref<f64>)
  gradient.reverse @rev.rev(tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) -> () attributes {implementation = @rev, argc = 1: i64, resc = 1 : i64, tape = 1: i64}
  // CHECK: [[tensorTape0:%.+]] = bufferization.to_tensor [[tape0]]
  // CHECK: [[tensorCotangent0:%.+]] = bufferization.to_tensor [[cotan0]]
  // CHECK: [[tensorDiff0:%.+]] = func.call @rev([[tensorTape0]], [[tensorCotangent0]])
  // CHECK: [[memrefDiff0:%.+]] = bufferization.to_memref [[tensorDiff0]]
  // CHECK: memref.copy [[memrefDiff0]], [[diff0]]
  // CHECK: gradient.return {empty = true}
}

// -----

module @rev_test_no_tape {

  func.func private @bwd(%arg0: tensor<f64>) -> tensor<f64>

  // CHECK-LABEL: gradient.reverse @bwd.rev
  // CHECK-SAME:[[in0:%.+]]: memref<f64>, [[diff0:%.+]]: memref<f64>, [[out0:%.+]]: memref<f64>, [[cotan0:%.+]]: memref<f64>)
  gradient.reverse @bwd.rev(tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) attributes {argc = 1 : i64, implementation = @bwd, llvm.linkage = #llvm.linkage<internal>, resc = 1 : i64, tape = 0 : i64}
  // CHECK: [[tensorCotangent0:%.+]] = bufferization.to_tensor [[cotan0]]
  // CHECK: [[tensorDiff0:%.+]] = func.call @bwd([[tensorCotangent0]])
  // CHECK: [[memrefDiff0:%.+]] = bufferization.to_memref [[tensorDiff0]]
  // CHECK: gradient.return {empty = true}
}
