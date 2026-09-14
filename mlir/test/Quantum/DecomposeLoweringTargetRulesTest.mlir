// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --pass-pipeline='builtin.module(decompose-lowering{target-rules=my_X_decomp,my_Z_decomp})' --split-input-file -verify-diagnostics %s | FileCheck %s

// Test that decompose-lowering only applies the rules requested by the `target-rules` option when present

module @test_module {
    // CHECK: func.func private @my_X_decomp
    func.func private @my_X_decomp(%q: !quantum.bit) -> !quantum.bit attributes {target_gate="X"} {
        %angle = arith.constant 1.57 : f64
        %out = quantum.custom "RX"(%angle) %q : !quantum.bit
        return %out : !quantum.bit
    }

    // CHECK: func.func private @my_Y_decomp
    func.func private @my_Y_decomp(%q: !quantum.bit) -> !quantum.bit attributes {target_gate="Y"} {
        %angle = arith.constant 1.57 : f64
        %out = quantum.custom "RY"(%angle) %q : !quantum.bit
        return %out : !quantum.bit
    }

    // CHECK: func.func private @my_Z_decomp
    func.func private @my_Z_decomp(%q: !quantum.bit) -> !quantum.bit attributes {target_gate="Z"} {
        %angle = arith.constant 1.57 : f64
        %out = quantum.custom "RZ"(%angle) %q : !quantum.bit
        return %out : !quantum.bit
    }

    func.func @circuit() attributes {quantum.node} {
      // CHECK: [[q:%.+]] = qref.alloc_qb
      // CHECK: qref.custom "RX"(%{{.+}}) [[q]]
      // CHECK: qref.custom "Y"() [[q]]
      // CHECK: qref.custom "RZ"(%{{.+}}) [[q]]
      // CHECK: qref.dealloc_qb [[q]]
      %0 = quantum.alloc_qb : !quantum.bit
      %1 = quantum.custom "X"() %0 : !quantum.bit
      %2 = quantum.custom "Y"() %1 : !quantum.bit
      %3 = quantum.custom "Z"() %2 : !quantum.bit
      quantum.dealloc_qb %3 : !quantum.bit
      return
    }
}

// -----

// The pass normalizes to reference semantics even when no decomposition rules are present.
// CHECK-LABEL: module @no_rules
// CHECK: func.func @circuit() attributes {quantum.node}
// CHECK: [[Q:%.+]] = qref.alloc_qb : !qref.bit
// CHECK: qref.custom "Hadamard"() [[Q]] : !qref.bit
// CHECK: qref.dealloc_qb [[Q]] : !qref.bit
module @no_rules {
  func.func @circuit() attributes {quantum.node} {
    %q = quantum.alloc_qb : !quantum.bit
    %out = quantum.custom "Hadamard"() %q : !quantum.bit
    quantum.dealloc_qb %out : !quantum.bit
    return
  }
}
