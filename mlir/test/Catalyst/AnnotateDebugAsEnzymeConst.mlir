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

// RUN: quantum-opt %s --annotate-debug-callback-as-enzyme-const --split-input-file --verify-diagnostics | FileCheck %s

// This test just makes sure that we can
// run the compiler with the option
//
//   --annotate-debug-callback-as-enzyme-const

// CHECK-LABEL: @foo
func.func @foo() {
  return
}

// -----

// This test checks the invariant that after the transformation
// the attribute has been removed.

// CHECK-LABEL: @test1
module @test1 {
  // CHECK: llvm.call @pyregistry
  // CHECK-NOT: catalyst.debugCallback
  llvm.func @pyregistry(i64, i64, i64, ...)  attributes { catalyst.debugCallback }
  llvm.func @wrapper() {
    %0 = llvm.mlir.constant(139935726668624 : i64) : i64
    %1 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @pyregistry(%0, %1, %1) vararg(!llvm.func<void (i64, i64, i64, ...)>) : (i64, i64, i64) -> ()
    llvm.return
  }
}
