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

func.func private @circuit.nodealloc(%arg0: f32) -> (!quantum.reg, f64)

// CHECK-DAG:   llvm.func @__catalyst__rt__toggle_recorder(i1)
// CHECK-DAG:   llvm.func @__catalyst__qis__Gradient(i64, ...)

// CHECK-LABEL: func.func @adjoint(%arg0: f32, %arg1: index) {{.+}} {
func.func @adjoint(%arg0: f32, %arg1 : index) -> (memref<?xf64>, memref<?xf64>) {
    // CHECK-DAG:   [[T:%.+]] = llvm.mlir.constant(true) : i1
    // CHECK-DAG:   [[F:%.+]] = llvm.mlir.constant(false) : i1

    // CHECK:       llvm.call @__catalyst__rt__toggle_recorder([[T]]) : (i1) -> ()
    // CHECK:       [[QREG_and_expval:%.+]]:2 = call @circuit.nodealloc(%arg0)
    // CHECK:       llvm.call @__catalyst__rt__toggle_recorder([[F]])

    // CHECK-DAG:   [[C1:%.+]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK-DAG:   [[C2:%.+]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:       [[GRAD1:%.+]] = llvm.alloca [[C1]] x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK:       [[GRAD2:%.+]] = llvm.alloca [[C1]] x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>

    // CHECK:       llvm.call @__catalyst__qis__Gradient([[C2]], [[GRAD1]], [[GRAD2]])
    // CHECK:       quantum.dealloc [[QREG_and_expval]]#0
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
    return %arg0, %arg0 : memref<f64>, memref<f64>
  }

  // CHECK-LABEL: func.func private @fwd.fwd(
  // CHECK: [[in0ptr:%.+]]: !llvm.ptr, [[diff0ptr:%.+]]: !llvm.ptr, [[out0ptr:%.+]]: !llvm.ptr, [[cotan0ptr:%.+]]: !llvm.ptr) -> !llvm.struct<(struct<(struct<(ptr, ptr, i64)>)>)>
  gradient.forward @fwd.fwd(%arg0: memref<f64>, %arg1: memref<f64>, %arg2: memref<f64>, %arg3: memref<f64>) -> (memref<f64>) attributes {argc = 1 : i64, implementation = @fwd, resc = 1 : i64, tape = 1 : i64} {
    // CHECK: [[in0struct:%.+]] = llvm.load [[in0ptr]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    // CHECK: [[in0memref:%.+]] = builtin.unrealized_conversion_cast [[in0struct]] : !llvm.struct<(ptr, ptr, i64)> to memref<f64>
    // CHECK: [[diff0struct:%.+]] = llvm.load [[diff0ptr]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    // CHECK: [[out0struct:%.+]] = llvm.load [[out0ptr]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    // CHECK: [[cotan0struct:%.+]] = llvm.load [[cotan0ptr]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
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

// -----

// CHECK-LABEL: @test_fwd_lowering_no_tape
module @test_fwd_lowering_no_tape {
  func.func private @fwd(%arg0: memref<f64>) -> memref<f64> attributes {llvm.linkage = #llvm.linkage<internal>} {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f64>
    catalyst.callback_call @callback_139840982544656(%arg0, %alloc) : (memref<f64>, memref<f64>) -> ()
    return %alloc : memref<f64>
  }
  // CHECK-LABEL: @fwd.fwd
  gradient.forward @fwd.fwd(%arg0: memref<f64>, %arg1: memref<f64>, %arg2: memref<f64>, %arg3: memref<f64>) attributes {argc = 1 : i64, implementation = @fwd, llvm.linkage = #llvm.linkage<internal>, resc = 1 : i64, tape = 0 : i64} {
    %0 = func.call @fwd(%arg0) : (memref<f64>) -> memref<f64>
    memref.copy %0, %arg2 : memref<f64> to memref<f64>
    gradient.return {empty = false}
    // CHECK: "llvm.intr.memcpy"
    // CHECK: [[null:%.+]] = llvm.mlir.zero : !llvm.ptr
    // CHECK-NEXT: llvm.return [[null]] : !llvm.ptr
  }
  catalyst.callback @callback_139840982544656(memref<f64>, memref<f64>) attributes {argc = 1 : i64, id = 139840982544656 : i64, llvm.linkage = #llvm.linkage<internal>, resc = 1 : i64}
}

// -----

// CHECK-LABEL: @test1
module @test1 {

  func.func private @rev(%arg0: memref<f64>, %arg1: memref<f64>) -> memref<f64> attributes {llvm.linkage = #llvm.linkage<internal>} {
    return %arg1 : memref<f64>
  }

  // CHECK-LABEL: func.func private @rev.rev
  // CHECK: [[in0ptr:%.+]]: !llvm.ptr, [[diff0ptr:%.+]]: !llvm.ptr, [[out0ptr:%.+]]: !llvm.ptr, [[cotan0ptr:%.+]]: !llvm.ptr, [[tape:%.+]]: !llvm.struct<(struct<(struct<(ptr, ptr, i64)>)>)>) attributes {passthrough = ["noinline"]}
  gradient.reverse @rev.rev(%arg0: memref<f64>, %arg1: memref<f64>, %arg2: memref<f64>, %arg3: memref<f64>, %arg4: memref<f64>) attributes {argc = 1 : i64, implementation = @rev, resc = 1 : i64, tape = 1 : i64} {

    // CHECK: [[in0struct:%.+]] = llvm.load [[in0ptr]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    // CHECK: [[diff0struct:%.+]] = llvm.load [[diff0ptr]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    // CHECK: [[out0struct:%.+]] = llvm.load [[out0ptr]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    // CHECK: [[out0memref:%.+]] = builtin.unrealized_conversion_cast [[out0struct]] : !llvm.struct<(ptr, ptr, i64)> to memref<f64>
    // CHECK: [[cotan0struct:%.+]] = llvm.load [[cotan0ptr]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    // CHECK: [[cotan0memref:%.+]] = builtin.unrealized_conversion_cast [[cotan0struct]] : !llvm.struct<(ptr, ptr, i64)> to memref<f64>
    // CHECK: [[tape0struct:%.+]] = llvm.extractvalue [[tape]][0, 0] : !llvm.struct<(struct<(struct<(ptr, ptr, i64)>)>)>
    // CHECK: [[tape0memref:%.+]] = builtin.unrealized_conversion_cast [[tape0struct]] : !llvm.struct<(ptr, ptr, i64)> to memref<f64>
    // CHECK: [[result0memref:%.+]] = call @rev([[tape0memref]], [[cotan0memref]])
    %0 = func.call @rev(%arg4, %arg3) : (memref<f64>, memref<f64>) -> memref<f64>

    // CHECK: [[result0struct:%.+]] = builtin.unrealized_conversion_cast [[result0memref]] : memref<f64> to !llvm.struct<(ptr, ptr, i64)>
    // CHECK: [[one:%.+]] = llvm.mlir.constant(1
    // CHECK: [[null:%.+]] = llvm.mlir.zero
    // CHECK: [[pointerToNullPlusOne:%.+]] = llvm.getelementptr [[null]][1]
    // CHECK: [[sizeOfF64InBytes:%.+]] = llvm.ptrtoint [[pointerToNullPlusOne]]
    // CHECK: [[sizeOfF64:%.+]] = llvm.mul [[one]], [[sizeOfF64InBytes]]
    // CHECK: [[alignedPtrSource:%.+]] = llvm.extractvalue [[result0struct]][1]
    // CHECK: [[offsetSource:%.+]] = llvm.extractvalue [[result0struct]][2]
    // CHECK: [[dataPtrSource:%.+]] = llvm.getelementptr [[alignedPtrSource]][[[offsetSource]]]
    // CHECK: [[alignedPtrDest:%.+]] = llvm.extractvalue [[diff0struct]][1]
    // CHECK: [[offsetDest:%.+]] = llvm.extractvalue [[diff0struct]][2]
    // CHECK: [[dataPtrDest:%.+]] = llvm.getelementptr [[alignedPtrDest]][[[offsetDest]]]

    memref.copy %0, %arg1 : memref<f64> to memref<f64>
    // CHECK: "llvm.intr.memcpy"([[dataPtrDest]], [[dataPtrSource]], [[sizeOfF64]])


    gradient.return {empty = true}
    // CHECK: llvm.return
  }

}

// -----

module @test_rev_no_tape {
  func.func private @bwd(%arg0: memref<f64>) -> memref<f64> attributes {llvm.linkage = #llvm.linkage<internal>} {
    return %arg0 : memref<f64>
  }
  // CHECK-LABEL: func.func private @bwd.rev(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr)
  gradient.reverse @bwd.rev(%arg0: memref<f64>, %arg1: memref<f64>, %arg2: memref<f64>, %arg3: memref<f64>) attributes {argc = 1 : i64, implementation = @bwd, llvm.linkage = #llvm.linkage<internal>, resc = 1 : i64, tape = 0 : i64} {
    %0 = func.call @bwd(%arg3) : (memref<f64>) -> memref<f64>
    memref.copy %0, %arg1 : memref<f64> to memref<f64>
    gradient.return {empty = true}
  }
}
