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

// RUN: quantum-opt %s --lower-quantum-module --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: test
func.func @test() {
  func.return
}

// -----

module @foo {
  func.func public @jit_foo() -> i1 attributes {llvm.emit_c_interface} {
    builtin.module @payload.cir {
      func.func private @cir() -> i1 attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
        %1 = quantum.alloc( 1) : !quantum.reg
        %2 = quantum.extract %1[ 0] : !quantum.reg -> !quantum.bit
        %mres, %out_qubit = quantum.measure %2 : i1, !quantum.bit
        %3 = quantum.insert %1[ 0], %out_qubit : !quantum.reg, !quantum.bit
        quantum.dealloc %3 : !quantum.reg
        quantum.device_release
        return %mres: i1
      }
    }
    %0 = catalyst.exec() {module = @payload.cir} : () -> i1
    return %0 : i1
  }
}
