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

// RUN: quantum-opt %s --register-inactive-callback --split-input-file --verify-diagnostics | FileCheck %s

// This test just makes sure that we can
// run the compiler with the option
//
//   --register-inactive-callback
//
// and that if there are no callbacks present
// it doesn't change anything

// CHECK-LABEL: @test0
module @test0 {
  // CHECK-NOT: llvm.mlir.global external @__enzyme_inactivefn
  // CHECK-LABEL: @foo
  func.func @foo() {
    return
  }
  // CHECK-NOT: llvm.mlir.global external @__enzyme_inactivefn
}

// -----

// This test checks the invariant that after the transformation
// the attribute has been removed.

// CHECK-LABEL: @test1
module @test1 {

  // CHECK: llvm.mlir.global external @__enzyme_inactivefn
  // CHECK: [[undef:%.+]] = llvm.mlir.undef
  // CHECK: [[ptr:%.+]] = llvm.mlir.addressof @__catalyst_inactive_callback
  // CHECK: [[retval:%.+]] = llvm.insertvalue [[ptr]], [[undef]][0]
  // CHECK: llvm.return [[retval]]

  llvm.func @__catalyst_inactive_callback(i64, i64, i64, ...)
  llvm.func @wrapper() {
    %0 = llvm.mlir.constant(139935726668624 : i64) : i64
    %1 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @__catalyst_inactive_callback(%0, %1, %1) vararg(!llvm.func<void (i64, i64, i64, ...)>) : (i64, i64, i64) -> ()
    llvm.return
  }
}
