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

// RUN: quantum-opt --to-ppr --resource-analysis=output-json %s -o /dev/null | FileCheck %s
// RUN: quantum-opt --estimate-transform-sequence %s -o /dev/null | FileCheck %s --check-prefixes=ESTIMATE,CHECK

// The resource effect of to-ppr gives the resources of the program after the pass. Rotations by an
// angle that is not known become arbitrary-angle PPRs.

// ESTIMATE:    "pass": "to-ppr"
// CHECK:       "circuit": {
// CHECK:         "num_qubits": {
// CHECK-NEXT:      "alloc": 2,
// CHECK:           "peak": 2,
// CHECK:         "quantum_operations": {
// CHECK-NEXT:      "0": {
// CHECK-NEXT:        "GlobalPhase": 5
// CHECK-NEXT:      },
// CHECK-NEXT:      "1": {
// CHECK-NEXT:        "PPM": 1,
// CHECK-NEXT:        "PPR-Phi": 1,
// CHECK-NEXT:        "PPR-pi/2": 1,
// CHECK-NEXT:        "PPR-pi/4": 6,
// CHECK-NEXT:        "PPR-pi/8": 1
// CHECK-NEXT:      },
// CHECK-NEXT:      "2": {
// CHECK-NEXT:        "PPM": 1,
// CHECK-NEXT:        "PPR-Phi": 1,
// CHECK-NEXT:        "PPR-pi/4": 1,
// CHECK-NEXT:        "PPR-pi/8": 1
// CHECK-NEXT:      }
// CHECK-NEXT:    }
module {
  func.func public @circuit(%angle: f64) attributes {quantum.node} {
    %r = quantum.alloc( 2) : !quantum.reg
    %q0 = quantum.extract %r[ 0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %r[ 1] : !quantum.reg -> !quantum.bit
    %0 = quantum.custom "Hadamard"() %q0 : !quantum.bit
    %1 = quantum.custom "S"() %0 adj : !quantum.bit
    %2 = quantum.custom "T"() %1 : !quantum.bit
    %3 = quantum.custom "PauliX"() %2 : !quantum.bit
    %4 = quantum.custom "Identity"() %3 : !quantum.bit
    %5:2 = quantum.custom "CNOT"() %4, %q1 : !quantum.bit, !quantum.bit
    %6 = quantum.custom "RZ"(%angle) %5#0 : !quantum.bit
    %7:2 = quantum.multirz(%angle) %6, %5#1 : !quantum.bit, !quantum.bit
    %8:2 = quantum.operator "PPR_8"() qubits(%7#0, %7#1)
        static_data = {sign = -1 : i64, pauli_word = "XZ"}
    %m0, %9:2 = quantum.measurement "PauliMeasure"() qubits(%8#0, %8#1)
        static_data = {pauli_word = "XZ", postselect = none}
        : (i1, !quantum.bit, !quantum.bit)
    %m1, %10 = quantum.measure %9#0 : i1, !quantum.bit
    %11 = quantum.insert %r[ 0], %10 : !quantum.reg, !quantum.bit
    %12 = quantum.insert %11[ 1], %9#1 : !quantum.reg, !quantum.bit
    quantum.dealloc %12 : !quantum.reg
    return
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
      %0 = transform.apply_registered_pass "to-ppr" to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
      transform.yield
    }
  }
}
