// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt %s --outline-quantum-module --lower-quantum-module --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: test
func.func @test() {
  func.return
}

// -----

module {
  func.func @cir(%arg0 : i64) -> i64 attributes { qnode } {
    %0 = arith.constant 1 : i64
    %1 = arith.addi %arg0, %0 : i64
    func.return %1 : i64
  }
  func.func @jit_foo(%arg0 : i64) -> i64 attributes {llvm.emit_c_interface} {
    %0 = func.call @cir(%arg0) : (i64) -> i64
    return %0 : i64
  }
}
