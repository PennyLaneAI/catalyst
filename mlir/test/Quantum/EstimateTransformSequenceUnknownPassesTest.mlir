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

// RUN: quantum-opt -estimate-transform-sequence=allow-unknown-passes=true -verify-diagnostics %s -o /dev/null | FileCheck %s

// When unknown passes are allowed, a pass without a resource effect is assumed not to change the
// resources. Every level of the sequence is written.

// CHECK:      "level": 0
// CHECK-NEXT: "pass": null
// CHECK:            "Hadamard": 1
// CHECK:      "level": 1
// CHECK-NEXT: "pass": "cse"
// CHECK:            "Hadamard": 1
// CHECK:      "level": 2
// CHECK-NEXT: "pass": "canonicalize"
// CHECK:        "circuit": {
// CHECK:            "Hadamard": 1
module {
  func.func public @circuit() attributes {quantum.node} {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.custom "Hadamard"() %1 : !quantum.bit
    %3 = quantum.insert %0[ 0], %2 : !quantum.reg, !quantum.bit
    quantum.dealloc %3 : !quantum.reg
    return
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
      // expected-warning@+1 {{no resource effect is declared for the pass 'cse'; it is assumed not to change the resources}}
      %0 = transform.apply_registered_pass "cse" to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
      // expected-warning@+1 {{no resource effect is declared for the pass 'canonicalize'; it is assumed not to change the resources}}
      %1 = transform.apply_registered_pass "canonicalize" to %0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
      transform.yield
    }
  }
}
