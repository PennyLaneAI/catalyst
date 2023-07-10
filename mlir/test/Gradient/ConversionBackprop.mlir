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

// RUN: quantum-opt --convert-gradient-to-llvm --canonicalize -cse --split-input-file %s | FileCheck %s

//////////////////////
// Native Gradients //
//////////////////////

func.func private @argmap(%arg0: memref<f64>, %arg1: memref<?xf64>)

// CHECK-DAG:  llvm.mlir.global linkonce constant @enzyme_dupnoneed
// CHECK-DAG:  llvm.mlir.global linkonce constant @enzyme_const
// CHECK-DAG:  llvm.func @__enzyme_autodiff(...)
// CHECK-DAG:  llvm.func @_mlir_memref_to_llvm_alloc(i64) -> !llvm.ptr
// CHECK-DAG:  func.func private @argmap(memref<f64>, memref<?xf64>)

// CHECK-LABEL: func.func @backpropArgmap(%arg0: memref<f64>, %arg1: memref<?xf64>) -> memref<f64> {
func.func @backpropArgmap(%arg0: memref<f64>, %arg1: memref<?xf64>) -> memref<f64> {
    // Constants and quantum gradient casting
    // CHECK-DAG:   [[memsetVal:%.+]] = llvm.mlir.constant(0 : i8) : i8
    // CHECK-DAG:   [[c8:%.+]] = llvm.mlir.constant(8 : index) : i64
    // CHECK-DAG:   [[argmapPtr:%.+]] = constant @argmap
    // CHECK-DAG:   [[c0:%.+]] = llvm.mlir.constant(0 : index) : i64
    // CHECK-DAG:   [[c1:%.+]] = llvm.mlir.constant(1 : index) : i64
    // CHECK-DAG:   [[qJacobianCasted:%.+]] = builtin.unrealized_conversion_cast %arg1 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>

    // Allocate space for the shadow
    // CHECK:   [[nullPtr:%.+]] = llvm.mlir.null
    // CHECK:   [[sizeOfF64:%.+]] = llvm.getelementptr [[nullPtr]][1] : (!llvm.ptr) -> !llvm.ptr, f64
    // CHECK:   [[sizeOfF64Int:%.+]] = llvm.ptrtoint [[sizeOfF64]] : !llvm.ptr to i64
    // CHECK:   [[shadowAlloc:%.+]] = llvm.call @_mlir_memref_to_llvm_alloc([[sizeOfF64Int]]) : (i64) -> !llvm.ptr
    // CHECK:   [[shadowMem0:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    // CHECK:   [[shadowMem1:%.+]] = llvm.insertvalue [[shadowAlloc]], [[shadowMem0]][0]
    // CHECK:   [[shadowMem2:%.+]] = llvm.insertvalue [[shadowAlloc]], [[shadowMem1]][1]
    // CHECK:   [[shadowMem3:%.+]] = llvm.insertvalue [[c0]], [[shadowMem2]][2]
    // CHECK:   [[shadowMemRef:%.+]] = builtin.unrealized_conversion_cast [[shadowMem3]] : !llvm.struct<(ptr, ptr, i64)> to memref<f64>

    // More casting
    // CHECK-DAG: [[argmapCasted:%.+]] = builtin.unrealized_conversion_cast [[argmapPtr]] : (memref<f64>, memref<?xf64>) -> () to !llvm.ptr
    // CHECK-DAG: [[enzymeConst:%.+]] = llvm.mlir.addressof @enzyme_const

    // Unpack the MemRef and zero out the shadow
    // CHECK: [[arg0Casted:%.+]] = builtin.unrealized_conversion_cast %arg0 : memref<f64> to !llvm.struct<(ptr, ptr, i64)>
    // CHECK: [[arg0Allocated:%.+]] = llvm.extractvalue [[arg0Casted]][0]
    // CHECK: [[arg0Aligned:%.+]] = llvm.extractvalue [[arg0Casted]][1]
    // CHECK: [[memsetSize:%.+]] = llvm.mul [[c8]], [[c1]]
    // CHECK-DAG: "llvm.intr.memset"([[shadowAlloc]], [[memsetVal]], [[memsetSize]]) <{isVolatile = false}>
    // CHECK-DAG: [[arg0Offset:%.+]] = llvm.extractvalue [[arg0Casted]][2]
    // CHECK-DAG: [[qJacobianSize:%.+]] = llvm.extractvalue [[qJacobianCasted]][3, 0]

    // Allocate space for the primal result
    // CHECK: [[sizeOfResult:%.+]] = llvm.getelementptr [[nullPtr]][[[qJacobianSize]]] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    // CHECK: [[sizeOfResultInt:%.+]] = llvm.ptrtoint [[sizeOfResult]] : !llvm.ptr to i64
    // CHECK-DAG: [[outputAlloc:%.+]] = llvm.call @_mlir_memref_to_llvm_alloc([[sizeOfResultInt]]) : (i64) -> !llvm.ptr
    // CHECK-DAG: [[enzymeDupNoNeed:%.+]] = llvm.mlir.addressof @enzyme_dupnoneed

    // Call Enzyme
    // CHECK-DAG: [[qJacobianAligned:%.+]] = llvm.extractvalue [[qJacobianCasted]][1]
    // CHECK: llvm.call @__enzyme_autodiff([[argmapCasted]], [[enzymeConst]], [[arg0Allocated]], [[arg0Aligned]], [[shadowAlloc]], [[arg0Offset]], [[enzymeConst]], [[outputAlloc]], [[enzymeDupNoNeed]], [[outputAlloc]], [[qJacobianAligned]], [[c0]], [[qJacobianSize]], [[c1]])

    %alloc0 = memref.alloc() : memref<f64>
    gradient.backprop @argmap(%arg0) qjacobian(%arg1: memref<?xf64>) in(%alloc0 : memref<f64>) {diffArgIndices=dense<0> : tensor<1xindex>} : (memref<f64>) -> ()

    return %alloc0: memref<f64>
}
