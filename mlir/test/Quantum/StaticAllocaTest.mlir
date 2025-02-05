// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --convert-quantum-to-llvm --split-input-file %s | FileCheck %s

// CHECK-LABEL: @static_alloca_qubit_unitary
module @static_alloca_qubit_unitary {
  // CHECK-LABEL: @test
  func.func @test(%arg0: memref<2x2xcomplex<f64>>, %arg1 : !quantum.bit) -> () {
    // CHECK-NOT: ^bb1:
    // CHECK: [[val:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK-NEXT: llvm.alloca [[val]] x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK: ^bb1:
    cf.br ^bb1
  ^bb1:
    %q1 = quantum.unitary(%arg0 : memref<2x2xcomplex<f64>>) %arg1 : !quantum.bit
    return
  }
}

// -----

// CHECK-LABEL: @static_alloca_qubit_unitary_ctrl
module @static_alloca_qubit_unitary_ctrl {
  // CHECK-LABEL: @test
  func.func @test(%arg0: memref<2x2xcomplex<f64>>, %arg1 : !quantum.bit, %arg2 : !quantum.bit, %arg3 : i1) -> () {
    // CHECK-NOT: ^bb1:
    // CHECK:      [[one0:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK-NEXT: [[one1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK-NEXT: [[one2:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK-NEXT: llvm.alloca [[one2]] x !llvm.struct<(i1, i64, ptr, ptr)>
    // CHECK-NEXT: llvm.alloca [[one1]] x !llvm.ptr
    // CHECK-NEXT: llvm.alloca [[one1]] x i1
    // CHECK-NEXT: llvm.alloca [[one0]] x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    cf.br ^bb1
  ^bb1:
    %q1, %q2 = quantum.unitary(%arg0 : memref<2x2xcomplex<f64>>) %arg1 ctrls(%arg2) ctrlvals(%arg3) : !quantum.bit ctrls !quantum.bit
    return
  }
}

// -----

// CHECK-LABEL: @static_alloca_hermitian
module @static_alloca_hermitian {
  // CHECK-LABEL: @test
  func.func @test(%arg0: memref<2x2xcomplex<f64>>, %arg1: !quantum.bit) -> () {
    // CHECK-NOT: ^bb1:
    // CHECK:      [[one:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK-NEXT: llvm.alloca [[one]] x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK: ^bb1:
    cf.br ^bb1
  ^bb1:
    %0 = quantum.hermitian(%arg0: memref<2x2xcomplex<f64>>) %arg1: !quantum.obs
    return
  }
}
