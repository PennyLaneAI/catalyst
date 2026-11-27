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

// RUN: quantum-opt --pass-pipeline="builtin.module(gridsynth{epsilon=0.001},resource-analysis{output-json})" --split-input-file %s -o /dev/null | FileCheck %s --check-prefix=PASS
// RUN: quantum-opt --estimate-transform-sequence --split-input-file %s -o /dev/null | FileCheck %s --check-prefix=ESTIMATE

// The pass calls a decomposition function for each rotation, whose loop the resource analysis
// weighs by its hint. The resource effect of gridsynth gives the same counts in the function of the
// rotations, e.g. 3 rotations x 33.524 iterations x 0.94299 T per iteration = 94.839 T.

// PASS:       "__catalyst_decompose_RZ": {
// PASS:           "for_loop_1": 33.524{{[0-9]+}}
// PASS:         "quantum_operations": {
// PASS-NEXT:      "0": {
// PASS-NEXT:        "GlobalPhase": 1
// PASS:       "circuit": {
// PASS:           "__catalyst_decompose_RZ": 3
// PASS:       "for_loop_1": {
// PASS:             "T": 0.94299{{[0-9]+}}

// ESTIMATE:    "pass": "gridsynth"
// ESTIMATE:      "circuit": {
// ESTIMATE:        "num_qubits": {
// ESTIMATE-NEXT:     "alloc": 1,
// ESTIMATE:          "peak": 1,
// ESTIMATE:        "quantum_operations": {
// ESTIMATE-NEXT:     "0": {
// ESTIMATE-NEXT:       "GlobalPhase": 4
// ESTIMATE-NEXT:     },
// ESTIMATE-NEXT:     "1": {
// ESTIMATE-NEXT:       "Adjoint(S)": 0.9436{{[0-9]+}},
// ESTIMATE-NEXT:       "Hadamard": 95.278{{[0-9]+}},
// ESTIMATE-NEXT:       "Identity": 0.2339{{[0-9]+}},
// ESTIMATE-NEXT:       "PauliX": 0.3737{{[0-9]+}},
// ESTIMATE-NEXT:       "PauliY": 0.3978{{[0-9]+}},
// ESTIMATE-NEXT:       "PauliZ": 0.9482{{[0-9]+}},
// ESTIMATE-NEXT:       "S": 47.881{{[0-9]+}},
// ESTIMATE-NEXT:       "T": 94.839{{[0-9]+}}
// ESTIMATE-NEXT:     }
// ESTIMATE-NEXT:   }
module {
  func.func public @circuit(%a: f64) attributes {quantum.node} {
    %r = quantum.alloc( 1) : !quantum.reg
    %q = quantum.extract %r[ 0] : !quantum.reg -> !quantum.bit
    %0 = quantum.custom "RZ"(%a) %q : !quantum.bit
    %1 = quantum.custom "PhaseShift"(%a) %0 adj : !quantum.bit
    %2 = pbc.ppr.arbitrary ["Z"](%a) %1 : !quantum.bit
    %3 = quantum.insert %r[ 0], %2 : !quantum.reg, !quantum.bit
    quantum.dealloc %3 : !quantum.reg
    return
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
      %0 = transform.apply_registered_pass "gridsynth" with options = {"epsilon" = 1.0e-3 : f64} to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
      transform.yield
    }
  }
}

// -----

// The mixed fallback method uses an ancilla, rotated between two CY gates and measured, and a
// recovery sequence on failure.

// ESTIMATE:    "pass": "gridsynth"
// ESTIMATE:      "circuit": {
// ESTIMATE:        "num_qubits": {
// ESTIMATE-NEXT:     "alloc": 4,
// ESTIMATE:          "peak": 2,
// ESTIMATE:        "quantum_operations": {
// ESTIMATE-NEXT:     "0": {
// ESTIMATE-NEXT:       "GlobalPhase": 4
// ESTIMATE-NEXT:     },
// ESTIMATE-NEXT:     "1": {
// ESTIMATE-NEXT:       "PPM": 3,
// ESTIMATE-NEXT:       "PPR-pi/2": 0.00088{{[0-9]+}},
// ESTIMATE-NEXT:       "PPR-pi/4": 12.008{{[0-9]+}},
// ESTIMATE-NEXT:       "PPR-pi/8": 28.340{{[0-9]+}}
// ESTIMATE-NEXT:     },
// ESTIMATE-NEXT:     "2": {
// ESTIMATE-NEXT:       "PPR-pi/4": 6
// ESTIMATE-NEXT:     }
// ESTIMATE-NEXT:   }
module {
  func.func public @circuit(%a: f64) attributes {quantum.node} {
    %r = quantum.alloc( 1) : !quantum.reg
    %q = quantum.extract %r[ 0] : !quantum.reg -> !quantum.bit
    %0 = quantum.custom "RZ"(%a) %q : !quantum.bit
    %1 = quantum.custom "PhaseShift"(%a) %0 adj : !quantum.bit
    %2 = pbc.ppr.arbitrary ["Z"](%a) %1 : !quantum.bit
    %3 = quantum.insert %r[ 0], %2 : !quantum.reg, !quantum.bit
    quantum.dealloc %3 : !quantum.reg
    return
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
      %0 = transform.apply_registered_pass "gridsynth" with options = {"epsilon" = 1.0e-3 : f64, "ppr-basis" = true, "method" = "mixed-fallback"} to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
      transform.yield
    }
  }
}
