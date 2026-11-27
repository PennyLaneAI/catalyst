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

// RUN: catalyst --tool=opt --catalyst-pipeline="pipe(estimate-transform-stage{allow-unknown-passes=true})" %s -o %t.mlir 2> /dev/null | FileCheck %s
// RUN: FileCheck %s --check-prefix=PROGRAM --input-file=%t.mlir

// The estimate stage runs the user transform stage, but estimates the transform sequence of each
// qnode module instead of applying it.

// CHECK:      "level": 0
// CHECK:        "circuit": {
// CHECK:            "Hadamard": 2
// CHECK:      "level": 1
// CHECK-NEXT: "pass": "cancel-inverses"
// CHECK:        "circuit": {
// CHECK:            "Hadamard": 2

// The passes of the sequence are not applied.
// PROGRAM: quantum.custom "Hadamard"
// PROGRAM: quantum.custom "Hadamard"
// PROGRAM-NOT: transform.apply_registered_pass
module @program {
  module @circuit {
    func.func public @circuit() attributes {quantum.node} {
      %0 = quantum.alloc( 1) : !quantum.reg
      %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
      %2 = quantum.custom "Hadamard"() %1 : !quantum.bit
      %3 = quantum.custom "Hadamard"() %2 : !quantum.bit
      %4 = quantum.insert %0[ 0], %3 : !quantum.reg, !quantum.bit
      quantum.dealloc %4 : !quantum.reg
      return
    }

    module attributes {transform.with_named_sequence} {
      transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
        %0 = transform.apply_registered_pass "cancel-inverses" to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
        transform.yield
      }
    }
  }
}
