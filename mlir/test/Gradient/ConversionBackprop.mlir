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

// RUN: quantum-opt --convert-gradient-to-llvm --split-input-file %s | FileCheck %s

//////////////////////
// Native Gradients //
//////////////////////

func.func private @argmap(%arg0: memref<f64>) -> (memref<?xf64>)

// CHECK-DAG:  llvm.func @memset(!llvm.ptr, i32, i64) -> !llvm.ptr
// CHECK-DAG:  llvm.func @__enzyme_autodiff(...)
// CHECK-DAG:  llvm.func @_mlir_memref_to_llvm_alloc(i64) -> !llvm.ptr
// CHECK-DAG:  func.func private @argmap(memref<f64>) -> memref<?xf64>
// CHECK-DAG:  func.func private @argmap.enzyme_wrapper(%arg0: !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64)>>, %arg1: !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>) {
// CHECK-DAG:  [[LOADARG0:%.+]] = llvm.load %arg0 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64)>>
// CHECK-DAG:  [[CASTARG0:%.+]] = builtin.unrealized_conversion_cast [[LOADARG0]] : !llvm.struct<(ptr<f64>, ptr<f64>, i64)> to memref<f64>
// CHECK-DAG:  [[ARGMAPRES:%.+]] = call @argmap([[CASTARG0]]) : (memref<f64>) -> memref<?xf64>
// CHECK-DAG:  [[ARGMAPRESCAST:%.+]] = builtin.unrealized_conversion_cast [[ARGMAPRES]] : memref<?xf64> to !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:  llvm.store [[ARGMAPRESCAST]], %arg1 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>

// CHECK-LABEL: func.func @backpropArgmap(%arg0: memref<f64>, %arg1: memref<?xf64>) -> memref<f64> {
func.func @backpropArgmap(%arg0: memref<f64>, %arg1: memref<?xf64>) -> memref<f64> {
    // CHECK:   [[ARG0CAST:%.+]] = builtin.unrealized_conversion_cast %arg0 : memref<f64> to !llvm.struct<(ptr<f64>, ptr<f64>, i64)>
    // CHECK:   [[QJACS:%.+]] = builtin.unrealized_conversion_cast %arg1 : memref<?xf64> to !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK:   [[ENZYMEWRAPPER:%.+]] = constant @argmap.enzyme_wrapper : (!llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64)>>, !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>) -> ()
    // CHECK:   [[ENZYMEWRAPPERCAST:%.+]] = builtin.unrealized_conversion_cast [[ENZYMEWRAPPER]] : (!llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64)>>, !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>) -> () to !llvm.ptr<func<void (ptr<struct<(ptr<f64>, ptr<f64>, i64)>>, ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>)>>
    // CHECK:   [[ARGPTR:%.+]] = llvm.alloca %3 x !llvm.struct<(ptr<f64>, ptr<f64>, i64)> : (i32) -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64)>>
    // CHECK:   llvm.store [[ARG0CAST]], [[ARGPTR]] : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64)>>
    // CHECK:   [[MEMREFSIZE:%.+]] = llvm.mul %7, %5  : i64
    // CHECK:   [[ALLOC:%.+]] = llvm.call @_mlir_memref_to_llvm_alloc([[MEMREFSIZE]]) : (i64) -> !llvm.ptr
    // CHECK:   [[MEMSET:%.+]] = llvm.call @memset([[ALLOC]], {{.*}}, [[MEMREFSIZE]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr
    // CHECK:   [[BITCAST:%.+]] = llvm.bitcast [[ALLOC]] : !llvm.ptr to !llvm.ptr<f64>
    // CHECK:   [[INSERT0:%.+]] = llvm.insertvalue [[BITCAST]], %2[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64)> 
    // CHECK:   [[INSERT1:%.+]] = llvm.insertvalue [[BITCAST]], [[INSERT0]][1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64)> 
    // CHECK:   [[ARGSHADOW:%.+]] = llvm.alloca %3 x !llvm.struct<(ptr<f64>, ptr<f64>, i64)> : (i32) -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64)>>
    // CHECK:   llvm.store [[INSERT1]], [[ARGSHADOW]] : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64)>>
    // CHECK:   [[CALLARGMAP:%.+]] = call @argmap(%arg0) : (memref<f64>) -> memref<?xf64>
    // CHECK:   [[CALLARGMAPSCAST:%.+]] = builtin.unrealized_conversion_cast [[CALLARGMAP]] : memref<?xf64> to !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK:   [[RES:%.+]] = llvm.alloca %3 x !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> : (i32) -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    // CHECK:   llvm.store [[CALLARGMAPSCAST]], [[RES]] : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    // CHECK:   [[RESSHADOW:%.+]] = llvm.alloca %3 x !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> : (i32) -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    // CHECK:   llvm.store [[QJACS]], [[RESSHADOW]] : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    // CHECK:   llvm.call @__enzyme_autodiff([[ENZYMEWRAPPERCAST]], [[ARGPTR]], [[ARGSHADOW]], [[RES]], [[RESSHADOW]]) : (!llvm.ptr<func<void (ptr<struct<(ptr<f64>, ptr<f64>, i64)>>, ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>)>>, !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64)>>, !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64)>>, !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>, !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>) -> ()

    %alloc0 = memref.alloc() : memref<f64>
    gradient.backprop @argmap(%arg0) qjacobian(%arg1: memref<?xf64>) in(%alloc0 : memref<f64>) {diffArgIndices=dense<0> : tensor<1xindex>} : (memref<f64>) -> ()

    return %alloc0: memref<f64>
}

// -----

func.func private @argmap(%arg0: memref<1xf64>) -> (memref<?xf64>)

// CHECK-DAG:  llvm.func @memset(!llvm.ptr, i32, i64) -> !llvm.ptr
// CHECK-DAG:  llvm.func @__enzyme_autodiff(...)
// CHECK-DAG:  llvm.func @_mlir_memref_to_llvm_alloc(i64) -> !llvm.ptr
// CHECK-DAG:  func.func private @argmap(memref<1xf64>) -> memref<?xf64>
// CHECK-DAG:  func.func private @argmap.enzyme_wrapper(%arg0: !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>, %arg1: !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>) {
// CHECK-DAG:  [[LOADARG0:%.+]] = llvm.load %arg0 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
// CHECK-DAG:  [[CASTARG0:%.+]] = builtin.unrealized_conversion_cast [[LOADARG0]] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> to memref<1xf64>
// CHECK-DAG:  [[ARGMAPRES:%.+]] = call @argmap([[CASTARG0]]) : (memref<1xf64>) -> memref<?xf64>
// CHECK-DAG:  [[ARGMAPRESCAST:%.+]] = builtin.unrealized_conversion_cast [[ARGMAPRES]] : memref<?xf64> to !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:  llvm.store [[ARGMAPRESCAST]], %arg1 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>

// CHECK-LABEL: func.func @backpropArgmap2(%arg0: memref<1xf64>, %arg1: memref<?xf64>) -> memref<1xf64> {
func.func @backpropArgmap2(%arg0: memref<1xf64>, %arg1: memref<?xf64>) -> memref<1xf64> {
    // CHECK:   [[ARG0CAST:%.+]] = builtin.unrealized_conversion_cast %arg0 : memref<1xf64> to !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK:   [[QJACS:%.+]] = builtin.unrealized_conversion_cast %arg1 : memref<?xf64> to !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK:   [[ENZYMEWRAPPER:%.+]] = constant @argmap.enzyme_wrapper : (!llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>, !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>) -> ()
    // CHECK:   [[ENZYMEWRAPPERCAST:%.+]] = builtin.unrealized_conversion_cast [[ENZYMEWRAPPER]] : (!llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>, !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>) -> () to !llvm.ptr<func<void (ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>)>>
    // CHECK:   [[ARGPTR:%.+]] = llvm.alloca %3 x !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> : (i32) -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    // CHECK:   llvm.store [[ARG0CAST]], [[ARGPTR]] : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    // CHECK:   [[MEMREFSIZE:%.+]] = llvm.mul %11, %5  : i64
    // CHECK:   [[ALLOC:%.+]] = llvm.call @_mlir_memref_to_llvm_alloc([[MEMREFSIZE]]) : (i64) -> !llvm.ptr
    // CHECK:   [[MEMSET:%.+]] = llvm.call @memset([[ALLOC]], {{.*}}, [[MEMREFSIZE]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr
    // CHECK:   [[BITCAST:%.+]] = llvm.bitcast [[ALLOC]] : !llvm.ptr to !llvm.ptr<f64>
    // CHECK:   [[INSERT0:%.+]] = llvm.insertvalue [[BITCAST]], %2[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK:   [[INSERT1:%.+]] = llvm.insertvalue [[BITCAST]], [[INSERT0]][1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK:   [[ARGSHADOW:%.+]] = llvm.alloca %3 x !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> : (i32) -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    // CHECK:   llvm.store [[INSERT1]], [[ARGSHADOW]] : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    // CHECK:   [[CALLARGMAP:%.+]] = call @argmap(%arg0) : (memref<1xf64>) -> memref<?xf64>
    // CHECK:   [[CALLARGMAPSCAST:%.+]] = builtin.unrealized_conversion_cast [[CALLARGMAP]] : memref<?xf64> to !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK:   [[RES:%.+]] = llvm.alloca %3 x !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> : (i32) -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    // CHECK:   llvm.store [[CALLARGMAPSCAST]], [[RES]] : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    // CHECK:   [[RESSHADOW:%.+]] = llvm.alloca %3 x !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> : (i32) -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    // CHECK:   llvm.store [[QJACS]], [[RESSHADOW]] : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
    // CHECK:   llvm.call @__enzyme_autodiff([[ENZYMEWRAPPERCAST]], [[ARGPTR]], [[ARGSHADOW]], [[RES]], [[RESSHADOW]]) : (!llvm.ptr<func<void (ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>)>>, !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>, !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>, !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>, !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>) -> ()
    %alloc0 = memref.alloc() : memref<1xf64>
    gradient.backprop @argmap(%arg0) qjacobian(%arg1: memref<?xf64>) in(%alloc0 : memref<1xf64>) {diffArgIndices=dense<0> : tensor<1xindex>} : (memref<1xf64>) -> ()

    return %alloc0: memref<1xf64>
}