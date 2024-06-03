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

// RUN: quantum-opt --convert-gradient-to-llvm --finalize-memref-to-llvm --split-input-file %s | FileCheck %s

//////////////////////
// Native Gradients //
//////////////////////

func.func private @circuit.nodealloc(%arg0: f32) -> (!quantum.reg)

// CHECK-DAG:   llvm.func @__catalyst__rt__toggle_recorder(i1)
// CHECK-DAG:   llvm.func @__catalyst__qis__Gradient(i64, ...)

// CHECK-LABEL: func.func @adjoint(%arg0: f32, %arg1: index) {{.+}} {
func.func @adjoint(%arg0: f32, %arg1 : index) -> (memref<?xf64>, memref<?xf64>) {
    // CHECK-DAG:   [[T:%.+]] = llvm.mlir.constant(true) : i1
    // CHECK-DAG:   [[F:%.+]] = llvm.mlir.constant(false) : i1

    // CHECK:       llvm.call @__catalyst__rt__toggle_recorder([[T]]) : (i1) -> ()
    // CHECK:       [[QREG:%.+]] = call @circuit.nodealloc(%arg0)
    // CHECK:       llvm.call @__catalyst__rt__toggle_recorder([[F]])

    // CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK-DAG:   [[C2:%.+]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:       [[GRAD1:%.+]] = llvm.alloca [[C1]] x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK:       [[GRAD2:%.+]] = llvm.alloca [[C1]] x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>

    // CHECK:       llvm.call @__catalyst__qis__Gradient([[C2]], [[GRAD1]], [[GRAD2]])
    // CHECK:       quantum.dealloc [[QREG]]
    %alloc0 = memref.alloc(%arg1) : memref<?xf64>
    %alloc1 = memref.alloc(%arg1) : memref<?xf64>
    gradient.adjoint @circuit.nodealloc(%arg0) size(%arg1) in(%alloc0, %alloc1 : memref<?xf64>, memref<?xf64>) : (f32) -> ()

    return %alloc0, %alloc1 : memref<?xf64>, memref<?xf64>
}

// -----

// CHECK-LABEL: @test0
module @test0 {
  memref.global "private" constant @__constant_xf64 : memref<f64> = dense<1.000000e+00>

  func.func private @fwd(%arg0: memref<f64>) -> (memref<f64>, memref<f64>) {
    %0 = memref.get_global @__constant_xf64 : memref<f64>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f64>
    catalyst.callback_call @callback_140505513630752(%arg0, %alloc) : (memref<f64>, memref<f64>) -> ()
    return %alloc, %0 : memref<f64>, memref<f64>
  }

  // CHECK-LABEL: func.func private @fwd.fwd(
  // CHECK: [[in0ptr:%.+]]: !llvm.ptr, [[diff0ptr:%.+]]: !llvm.ptr, [[out0ptr:%.+]]: !llvm.ptr, [[cotan0ptr:%.+]]: !llvm.ptr) -> !llvm.struct<(struct<(struct<(ptr, ptr, i64)>)>)>
  gradient.forward @fwd.fwd(%arg0: memref<f64>, %arg1: memref<f64>, %arg2: memref<f64>, %arg3: memref<f64>) -> (memref<f64>) attributes {argc = 1 : i64, implementation = @fwd, resc = 1 : i64, tape = 1 : i64} {
    // CHECK: [[in0struct:%.+]] = llvm.load [[in0ptr]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    // CHECK: [[in0memref:%.+]] = builtin.unrealized_conversion_cast [[in0struct]] : !llvm.struct<(ptr, ptr, i64)> to memref<f64>
    // CHECK: [[diff0struct:%.+]] = llvm.load [[diff0ptr]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    // CHECK: [[diff0memref:%.+]] = builtin.unrealized_conversion_cast [[diff0struct]] : !llvm.struct<(ptr, ptr, i64)> to memref<f64>
    // CHECK: [[out0struct:%.+]] = llvm.load [[out0ptr]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    // CHECK: [[out0memref:%.+]] = builtin.unrealized_conversion_cast [[out0struct]] : !llvm.struct<(ptr, ptr, i64)> to memref<f64>
    // CHECK: [[cotan0struct:%.+]] = llvm.load [[cotan0ptr]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    // CHECK: [[cotan0memref:%.+]] = builtin.unrealized_conversion_cast [[cotan0struct]] : !llvm.struct<(ptr, ptr, i64)> to memref<f64>
    // CHECK: [[results:%.+]]:2 = call @fwd([[in0memref]])

    %1:2 = func.call @fwd(%arg0) : (memref<f64>) -> (memref<f64>, memref<f64>)
    memref.copy %1#0, %arg2 : memref<f64> to memref<f64>
    gradient.return {empty = false} %1#1 : memref<f64>

    // CHECK: [[resultOut0struct:%.+]] = builtin.unrealized_conversion_cast [[results]]#0 : memref<f64> to !llvm.struct<(ptr, ptr, i64)>
    // CHECK: [[resultTape0struct:%.+]] = builtin.unrealized_conversion_cast [[results]]#1 : memref<f64> to !llvm.struct<(ptr, ptr, i64)>
    // [[undef:%.+]] = llvm.mlir.undef : !llvm.struct<(struct<(struct<(ptr, ptr, i64)>)>)>
    // [[returnTape:%.+]] = llvm.insertvalue [[resultTape0struct]], [[undef]][0, 0]
    // llvm.return [[returnTape]]
  }

}

