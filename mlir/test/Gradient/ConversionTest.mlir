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
  func.func private @callback_140505513630752(memref<f64>, memref<f64>)
  func.func private @fwd(%arg0: memref<f64>) -> (memref<f64>, memref<f64>) {
    %0 = memref.get_global @__constant_xf64 : memref<f64>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f64>
    catalyst.callback_call @callback_140505513630752(%arg0, %alloc) : (memref<f64>, memref<f64>) -> ()
    return %alloc, %0 : memref<f64>, memref<f64>
  }
  // CHECK-LABEL: func.func private @fwd.fwd(
  // CHECK-SAME: [[arg0:%.+]]: !llvm.ptr, [[ash0:%.+]]: !llvm.ptr, [[out0:%.+]]: !llvm.ptr, [[osh0:%.+]]: !llvm.ptr)
  // CHECK-NOT: gradient.return
  gradient.forward @fwd.fwd(%arg0: memref<f64>, %arg1: memref<f64>, %arg2: memref<f64>, %arg3: memref<f64>) -> (memref<f64>) attributes {argc = 1 : i64, implementation = @fwd, resc = 1 : i64, tape = 1 : i64} {
    %0:2 = func.call @fwd(%arg0) : (memref<f64>) -> (memref<f64>, memref<f64>)
    memref.copy %0#0, %arg2 : memref<f64> to memref<f64>
    gradient.return { empty = false } %0#1 : memref<f64>
  }

}

// -----

// CHECK-LABEL: @test1
module @test1 {

  func.func private @bwd(%arg0: memref<f64>, %arg1: memref<f64>) -> memref<f64> attributes {llvm.linkage = #llvm.linkage<internal>} {
    return %arg1 : memref<f64>
  }

  // CHECK-LABEL: func.func private @bwd.rev
  // CHECK-SAME: [[arg0:%.+]]: !llvm.ptr, [[ash0:%.+]]: !llvm.ptr, [[out0:%.+]]: !llvm.ptr, [[osh0:%.+]]: !llvm.ptr, [[tap0:%.+]]: !llvm.struct<(struct<(ptr, ptr, i64)>)>)
  // CHECK-NOT: gradient-return
  gradient.reverse @bwd.rev(%arg0: memref<f64>, %arg1: memref<f64>, %arg2: memref<f64>, %arg3: memref<f64>, %arg4: memref<f64>) attributes {argc = 1 : i64, implementation = @bwd, llvm.linkage = #llvm.linkage<internal>, resc = 1 : i64, tape = 1 : i64} {
    %0 = func.call @bwd(%arg4, %arg3) : (memref<f64>, memref<f64>) -> memref<f64>
    memref.copy %0, %arg1 : memref<f64> to memref<f64>
    gradient.return { empty = true }
  }

}
