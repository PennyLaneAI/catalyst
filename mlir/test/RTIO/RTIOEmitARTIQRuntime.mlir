// Copyright 2025 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt %s --emit-artiq-runtime --split-input-file | FileCheck %s

// Test basic ARTIQ runtime wrapper generation
// CHECK-LABEL: module @test_basic
module @test_basic {
  // CHECK: llvm.func @__artiq_personality(...) -> i32
  // CHECK: llvm.func @__modinit__(%arg0: !llvm.ptr) attributes {personality = @__artiq_personality}
  // CHECK-SAME: {
  // CHECK:   llvm.call tail @__kernel__() : () -> ()
  // CHECK:   llvm.return
  // CHECK: }
  llvm.func @__kernel__() {
    llvm.return
  }
}

// -----

// Test with kernel function that has arguments
// CHECK-LABEL: module @test_kernel_with_args
module @test_kernel_with_args {
  // CHECK: llvm.func @__artiq_personality(...) -> i32
  // CHECK: llvm.func @__modinit__(%arg0: !llvm.ptr) attributes {personality = @__artiq_personality}
  // CHECK-SAME: {
  // CHECK:   %[[ZERO_PTR0:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK:   %[[ZERO_PTR1:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK:   %[[ZERO_I64:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK:   llvm.call tail @__kernel__(%[[ZERO_PTR0]], %[[ZERO_PTR1]], %[[ZERO_I64]])
  // CHECK:   llvm.return
  // CHECK: }
  llvm.func @__kernel__(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64) {
    llvm.return
  }
}

