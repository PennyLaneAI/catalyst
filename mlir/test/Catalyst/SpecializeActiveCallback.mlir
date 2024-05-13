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

// RUN: quantum-opt %s --specialize-active-callback-pass --split-input-file --verify-diagnostics | FileCheck %s

// This test just makes sure that we can
// run the compiler with the option
//
//   --specialize-active-callback-pass

// CHECK-LABEL: @foo
func.func @foo() {
  return
}

// -----

// This test checks that if an active callback
// then we also add the specialization.
// The specialization is denoted with `as` followed by the name
// of the specialized version.
// Suggestions for syntax are welcomed.

// CHECK-LABEL: @test1
module @test1 {

  llvm.func @cir() {
     // CHECK: catalyst.activeCallbackCall() as @active_callback_0
     catalyst.activeCallbackCall() { identifier = 0 } : () -> ()
     llvm.return
  }
}

// -----

// This test checks to see that the active callback is correctly defined
// CHECK-LABEL: @test2
module @test2 {


  // CHECK: llvm.func @inactive_callback(i64, i64, i64, ...)
  // CHECK-LABEL: @active_callback_0()
  // CHECK-NEXT: [[zero:%.+]] = llvm.mlir.constant(0 : i64)
  // CHEKC-NEXT: llvm.call @inactive_callback([[zero]], [[zero]], [[zero]])
  llvm.func @cir() {
     catalyst.activeCallbackCall() { identifier = 0 } : () -> ()
     llvm.return
  }

}

