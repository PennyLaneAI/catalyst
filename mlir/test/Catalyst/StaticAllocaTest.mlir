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

// CHECK-LABEL: @static_alloca_print
module @static_alloca_print {
  // CHECK-LABEL: @test
  func.func @test(%arg0: memref<i64>) -> () {
    // CHECK-NOT: ^bb1:
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: llvm.alloca [[c1]] x !llvm.struct<(i64, ptr, i8)>
    // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: llvm.alloca [[c1]] x !llvm.struct<(ptr, ptr, i64)>
    cf.br ^bb1
  ^bb1:
    "catalyst.print"(%arg0) <{print_descriptor}> : (memref<i64>) -> ()
    return
  }
}

// -----


// CHECK-LABEL: @static_alloca_custom_call
module @static_alloca_custom_call {
  // CHECK-LABEL: @custom_call
  func.func @custom_call(%arg0: memref<3x3xf64>, %arg1: memref<3x3xf64>) -> () {
    // CHECK-NOT: ^bb1:
    // CHECK: [[one:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: llvm.alloca [[one]] x !llvm.array<1 x ptr>
    // CHECK: [[one:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: llvm.alloca [[one]] x !llvm.struct<(i64, ptr, i8)>
    // CHECK: [[one:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: llvm.alloca [[one]] x !llvm.array<1 x ptr>
    // CHECK: [[one:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: llvm.alloca [[one]] x !llvm.struct<(i64, ptr, i8)>
    // CHECK: ^bb1:
    cf.br ^bb1
  ^bb1:
    %0 = catalyst.custom_call fn("lapack_dgesdd") (%arg0, %arg1) {number_original_arg = array<i32: 1>} : (memref<3x3xf64>, memref<3x3xf64>) -> memref<3x3xf64>
    return
  }
}

// -----

// CHECK-LABEL: @static_alloca_callback
module @static_alloca_callback {
  // CHECK-LABEL: @callback
  // This is good enough, if there are no jumps, it means this is the entry block
  // This always happens for callbacks since the callback function is just a wrapper
  // function without any logic.
  // CHECK-NOT: ^bb1
  catalyst.callback @callback(memref<f64>, memref<f64>) attributes {argc = 2 : i64, id = 4 : i64, resc = 3 : i64}

  // CHECK-LABEL: @test
  func.func @test(%arg0 : memref<f64>) -> () {
    // CHECK-NOT: ^bb1:
    // CHECK: [[one:%.+]] = llvm.mlir.constant(1 : i64)
    // CHECK: llvm.alloca [[one]] x !llvm.struct<(ptr, ptr, i64)>
    // CHECK: ^bb1:
    cf.br ^bb1
  ^bb1:
    catalyst.callback_call @callback(%arg0, %arg0) : (memref<f64>, memref<f64>) -> ()
    return
  }
}
