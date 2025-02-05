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

// RUN: quantum-opt --convert-catalyst-to-llvm --split-input-file %s | FileCheck %s

// CHECK-LABEL: @static_alloca_qubit_unitary
module @static_alloca_qubit_unitary {
  // CHECK-LABEL: @test
  func.func @test(%arg0: memref<i64>) -> () {
    // CHECK-NOT: ^bb1:
    // CHECK: [[one_1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: [[one_2:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: llvm.alloca [[one_2]] x !llvm.struct<(ptr, ptr, i64)>
    // CHECK: llvm.alloca [[one_1]] x !llvm.struct<(i64, ptr, i8)>
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.alloca %2 x !llvm.struct<(ptr, ptr, i64)> : (i64) -> !llvm.ptr
    %4 = llvm.alloca %1 x !llvm.struct<(i64, ptr, i8)> : (i64) -> !llvm.ptr
    cf.br ^bb1
  ^bb1:
    "catalyst.print"(%arg0) <{print_descriptor}> : (memref<i64>) -> ()
    return
  }
}

