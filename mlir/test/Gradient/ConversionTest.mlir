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

// RUN: quantum-opt --convert-gradient-to-llvm --convert-memref-to-llvm --split-input-file %s | FileCheck %s

//////////////////////
// Native Gradients //
//////////////////////

func.func private @circuit.nodealloc(%arg0: f32) -> (!quantum.reg)

// CHECK-DAG:   llvm.func @__quantum__rt__toggle_recorder(i1)
// CHECK-DAG:   llvm.func @__quantum__qis__Gradient(i64, ...)

// CHECK-LABEL: func.func @adjoint(%arg0: f32, %arg1: index) {{.+}} {
func.func @adjoint(%arg0: f32, %arg1 : index) -> (memref<?xf64>, memref<?xf64>) {
    // CHECK-DAG:   [[T:%.+]] = llvm.mlir.constant(true) : i1
    // CHECK-DAG:   [[F:%.+]] = llvm.mlir.constant(false) : i1

    // CHECK:       llvm.call @__quantum__rt__toggle_recorder([[T]]) : (i1) -> ()
    // CHECK:       [[QREG:%.+]] = call @circuit.nodealloc(%arg0)
    // CHECK:       llvm.call @__quantum__rt__toggle_recorder([[F]])

    // CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK-DAG:   [[C2:%.+]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:       [[GRAD1:%.+]] = llvm.alloca [[C1]] {{.+}} -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    // CHECK:       [[GRAD2:%.+]] = llvm.alloca [[C1]] {{.+}} -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>

    // CHECK:       llvm.call @__quantum__qis__Gradient([[C2]], [[GRAD1]], [[GRAD2]])
    // CHECK:       quantum.dealloc [[QREG]]
    %alloc0 = memref.alloc(%arg1) : memref<?xf64>
    %alloc1 = memref.alloc(%arg1) : memref<?xf64>
    gradient.adjoint @circuit.nodealloc(%arg0) size(%arg1) in(%alloc0, %alloc1 : memref<?xf64>, memref<?xf64>) : (f32) -> ()

    return %alloc0, %alloc1 : memref<?xf64>, memref<?xf64>
}

// -----

func.func private @argmap(%arg0: memref<f64>) -> (memref<?xf64>)

// CHECK-DAG:  llvm.func @memset(!llvm.ptr, i32, i64) -> !llvm.ptr
// CHECK-DAG:  llvm.func @__enzyme_autodiff(...)
// CHECK-DAG:  llvm.func @_mlir_memref_to_llvm_alloc(i64) -> !llvm.ptr
// CHECK-DAG:  func.func private @argmap(memref<f64>) -> memref<?xf64>
// CHECK-DAG:  func.func private @argmap.enzyme_wrapper(%arg0: !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>, %arg1: !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>) {

// CHECK-LABEL: func.func @backpropArgmap(%arg0: memref<f64>, %arg1: index, %arg2: memref<?xf64>) -> memref<?xf64> {
func.func @backpropArgmap(%arg0: memref<f64>, %arg1 : index, %arg2: memref<?xf64>) -> memref<?xf64> {

    // CHECK-DAG:   [[c0:%.+]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:   [[c1:%.+]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:   {{.*}} = constant @argmap.enzyme_wrapper : (!llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>, !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>) -> ()
    // CHECK:   {{.*}} = llvm.call @_mlir_memref_to_llvm_alloc({{.*}}) : (i64) -> !llvm.ptr
    // CHECK:   {{.*}} = llvm.call @memset({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr, i32, i64) -> !llvm.ptr
    // CHECK:   {{.*}} = call @argmap(%arg0) : (memref<f64>) -> memref<?xf64>
    // CHECK:   {{.*}} = builtin.unrealized_conversion_cast {{.*}} : memref<?xf64> to !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK:   llvm.call @__enzyme_autodiff({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<func<void (ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>)>>, !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>, !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>, !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>, !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>) -> ()
    %alloc0 = memref.alloc(%arg1) : memref<?xf64>
    gradient.backprop @argmap(%arg0) size(%arg1) qjacobians(%arg2: memref<?xf64>) in(%alloc0 : memref<?xf64>) {diffArgIndices=dense<0> : tensor<1xindex>} : (memref<f64>) -> ()

    return %alloc0: memref<?xf64>
}

// -----

// CHECK-DAG:  llvm.func @memset(!llvm.ptr, i32, i64) -> !llvm.ptr
// CHECK-DAG:  llvm.func @__enzyme_autodiff(...)
// CHECK-DAG:  llvm.func @_mlir_memref_to_llvm_alloc(i64) -> !llvm.ptr
// CHECK-DAG:  func.func private @argmap(memref<1xf64>) -> memref<?xf64>
// CHECK-DAG:  func.func private @argmap.enzyme_wrapper(%arg0: !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>, %arg1: !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>) {

func.func private @argmap(%arg0: memref<1xf64>) -> (memref<?xf64>)

// CHECK-LABEL: func.func @backpropArgmap2(%arg0: memref<1xf64>, %arg1: index, %arg2: memref<?xf64>) -> memref<?xf64> {
func.func @backpropArgmap2(%arg0: memref<1xf64>, %arg1 : index, %arg2: memref<?xf64>) -> memref<?xf64> {
    
    // CHECK-DAG:   [[c0:%.+]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:   [[c1:%.+]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:   {{.*}} = constant @argmap.enzyme_wrapper : (!llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>, !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>) -> ()
    // CHECK:   {{.*}} = llvm.call @_mlir_memref_to_llvm_alloc({{.*}}) : (i64) -> !llvm.ptr
    // CHECK:   {{.*}} = llvm.call @memset({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr, i32, i64) -> !llvm.ptr
    // CHECK:   {{.*}} = call @argmap(%arg0) : (memref<1xf64>) -> memref<?xf64>
    // CHECK:   {{.*}} = builtin.unrealized_conversion_cast {{.*}} : memref<?xf64> to !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK:   llvm.call @__enzyme_autodiff({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<func<void (ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>)>>, !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>, !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>, !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>, !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>) -> ()
    %alloc0 = memref.alloc(%arg1) : memref<?xf64>
    gradient.backprop @argmap(%arg0) size(%arg1) qjacobians(%arg2: memref<?xf64>) in(%alloc0 : memref<?xf64>) {diffArgIndices=dense<0> : tensor<1xindex>} : (memref<1xf64>) -> ()

    return %alloc0: memref<?xf64>
}