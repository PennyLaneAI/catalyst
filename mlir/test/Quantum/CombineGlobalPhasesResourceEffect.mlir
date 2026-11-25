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

// RUN: quantum-opt --combine-global-phases --resource-analysis=output-json %s -o /dev/null | FileCheck %s
// RUN: quantum-opt --estimate-transform-sequence %s -o /dev/null | FileCheck %s --check-prefixes=ESTIMATE,CHECK

// The resource effect of combine-global-phases leaves one uncontrolled global phase in each function,
// and keeps the controlled ones.

// ESTIMATE:    "pass": "combine-global-phases"
// CHECK:       "circuit": {
// CHECK:         "quantum_operations": {
// CHECK-NEXT:      "0": {
// CHECK-NEXT:        "GlobalPhase": 1
// CHECK-NEXT:      },
// CHECK-NEXT:      "1": {
// CHECK-NEXT:        "1C(GlobalPhase)": 1
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK:       "for_loop_1": {
// CHECK:         "quantum_operations": {
// CHECK-NEXT:      "0": {
// CHECK-NEXT:        "GlobalPhase": 1
// CHECK-NEXT:      }
// CHECK-NEXT:    }
module {
  func.func public @circuit(%a: f64, %b: f64, %q: !quantum.bit) -> !quantum.bit attributes {quantum.node} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %true = arith.constant true
    quantum.gphase(%a)
    quantum.gphase(%b) adj
    %out = quantum.gphase(%a) ctrls(%q) ctrlvals(%true) : ctrls !quantum.bit
    quantum.gphase(%b)
    scf.for %i = %c0 to %c4 step %c1 {
      quantum.gphase(%a)
      quantum.gphase(%b)
    }
    return %out : !quantum.bit
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
      %0 = transform.apply_registered_pass "combine-global-phases" to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
      transform.yield
    }
  }
}
