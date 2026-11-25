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

// RUN: quantum-opt --ppr-to-ppm --resource-analysis=output-json --split-input-file %s -o /dev/null | FileCheck %s
// RUN: quantum-opt --estimate-transform-sequence --split-input-file -verify-diagnostics %s -o /dev/null | FileCheck %s --check-prefixes=ESTIMATE,CHECK

// The resource effect of ppr-to-ppm gives the resources of the program after the pass, in each
// function. A pi/8 PPR uses a magic state and a pi/4 PPR an ancilla, released after it.

// ESTIMATE:    "pass": "ppr-to-ppm"
// CHECK:       "circuit": {
// CHECK:         "function_calls": {
// CHECK:           "for_loop_1": 3
// CHECK:         "num_qubits": {
// CHECK-NEXT:      "alloc": 6,
// CHECK:           "peak": 3,
// CHECK:         "quantum_operations": {
// CHECK-NEXT:      "0": {
// CHECK-NEXT:        "pbc.fabricate": 2
// CHECK-NEXT:      },
// CHECK-NEXT:      "1": {
// CHECK-NEXT:        "PPM": 5,
// CHECK-NEXT:        "PPR-Phi": 1,
// CHECK-NEXT:        "PPR-pi/2": 2
// CHECK-NEXT:      },
// CHECK-NEXT:      "2": {
// CHECK-NEXT:        "PPM": 2,
// CHECK-NEXT:        "PPR-pi/2": 3
// CHECK-NEXT:      },
// CHECK-NEXT:      "3": {
// CHECK-NEXT:        "PPM": 2
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK:       "for_loop_1": {
// CHECK:         "num_qubits": {
// CHECK-NEXT:      "alloc": 1,
// CHECK:           "peak": 1,
// CHECK:         "quantum_operations": {
// CHECK-NEXT:      "0": {
// CHECK-NEXT:        "pbc.fabricate": 1
// CHECK-NEXT:      },
// CHECK-NEXT:      "1": {
// CHECK-NEXT:        "PPM": 1,
// CHECK-NEXT:        "PPR-pi/2": 1
// CHECK-NEXT:      },
// CHECK-NEXT:      "2": {
// CHECK-NEXT:        "PPM": 1
// CHECK-NEXT:      }
// CHECK-NEXT:    }
module {
  func.func public @circuit(%angle: f64) attributes {quantum.node} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %r = quantum.alloc( 2) : !quantum.reg
    %q0 = quantum.extract %r[ 0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %r[ 1] : !quantum.reg -> !quantum.bit
    %0:2 = pbc.ppr ["X", "Z"](8) %q0, %q1 : !quantum.bit, !quantum.bit
    %1 = pbc.ppr ["Z"](-8) %0#0 : !quantum.bit
    %2 = pbc.ppr ["Y"](4) %1 : !quantum.bit
    %3:2 = pbc.ppr ["X", "Y"](-4) %2, %0#1 : !quantum.bit, !quantum.bit
    %4:2 = pbc.ppr ["X", "Y"](2) %3#0, %3#1 : !quantum.bit, !quantum.bit
    %5 = pbc.ppr.arbitrary ["Z"](%angle) %4#0 : !quantum.bit
    %6 = scf.for %i = %c0 to %c3 step %c1 iter_args(%q = %5) -> (!quantum.bit) {
      %t = pbc.ppr ["Z"](8) %q : !quantum.bit
      scf.yield %t : !quantum.bit
    }
    %m, %7 = pbc.ppm ["Z"] %6 : i1, !quantum.bit
    %8 = quantum.insert %r[ 0], %7 : !quantum.reg, !quantum.bit
    %9 = quantum.insert %8[ 1], %4#1 : !quantum.reg, !quantum.bit
    quantum.dealloc %9 : !quantum.reg
    return
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
      %0 = transform.apply_registered_pass "ppr-to-ppm" to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
      transform.yield
    }
  }
}

// -----

module {
  func.func public @avoid_y_measure() attributes {quantum.node} {
    return
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
      // expected-error@+1 {{resource estimation only supports ppr-to-ppm with decompose-method=pauli-corrected and without avoid-y-measure}}
      %0 = transform.apply_registered_pass "ppr-to-ppm" with options = {"avoid-y-measure" = true} to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
      transform.yield
    }
  }
}

// -----

module {
  func.func public @auto_corrected() attributes {quantum.node} {
    return
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
      // expected-error@+1 {{resource estimation only supports ppr-to-ppm with decompose-method=pauli-corrected and without avoid-y-measure}}
      %0 = transform.apply_registered_pass "ppr-to-ppm" with options = {"decompose-method" = "auto-corrected"} to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
      transform.yield
    }
  }
}
