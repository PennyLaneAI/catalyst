// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=RZ=1.0,GlobalPhase=1.0})' %s | FileCheck %s
// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(resource-estimate{output-json gate-set=RZ=1.0,GlobalPhase=1.0})' %s | FileCheck %s --check-prefix=ESTIMATE

// S needs on-demand rules. Its PhaseShift dependency already has a loaded alternative, but
// discovery must retain the cheaper builtin rule with a different identity for that same target.
// This rule performs ten small rotations instead of one full rotation.
func.func private @expensive_phase(%theta: f64, %q: !quantum.bit) -> !quantum.bit attributes {
  target_gate = "PhaseShift{0:[f64]}{wires:1}{}",
  frontend_name = "expensive_phase",
  resources = {operations = {
    "GlobalPhase{phi:[f64]}{}{}" = 1 : i64,
    "RZ{0:[f64]}{wires:1}{}" = 10 : i64
  }}
} {
  %tenth = arith.constant 0.1 : f64
  %angle = arith.mulf %theta, %tenth : f64
  %q0 = quantum.custom "RZ"(%angle) %q : !quantum.bit
  %q1 = quantum.custom "RZ"(%angle) %q0 : !quantum.bit
  %q2 = quantum.custom "RZ"(%angle) %q1 : !quantum.bit
  %q3 = quantum.custom "RZ"(%angle) %q2 : !quantum.bit
  %q4 = quantum.custom "RZ"(%angle) %q3 : !quantum.bit
  %q5 = quantum.custom "RZ"(%angle) %q4 : !quantum.bit
  %q6 = quantum.custom "RZ"(%angle) %q5 : !quantum.bit
  %q7 = quantum.custom "RZ"(%angle) %q6 : !quantum.bit
  %q8 = quantum.custom "RZ"(%angle) %q7 : !quantum.bit
  %q9 = quantum.custom "RZ"(%angle) %q8 : !quantum.bit
  %negative_half = arith.constant -0.5 : f64
  %phase = arith.mulf %theta, %negative_half : f64
  quantum.gphase(%phase)
  return %q9 : !quantum.bit
}

// A loaded rule with incomplete identity metadata must not produce a malformed Python entry.
func.func private @incomplete_rz(%theta: f64, %q: !quantum.bit) -> !quantum.bit attributes {
  target_gate = "RZ{0:[f64]}{wires:1}{}"
} {
  %out = quantum.custom "RZ"(%theta) %q : !quantum.bit
  return %out : !quantum.bit
}

// CHECK-LABEL: func.func @circuit
// CHECK-NOT: quantum.custom "S"
// CHECK: call @__builtin_
// CHECK: quantum.dealloc
func.func @circuit() attributes {quantum.node} {
  %r = quantum.alloc(1) : !quantum.reg
  %q = quantum.extract %r[0] : !quantum.reg -> !quantum.bit
  %out = quantum.custom "S"() %q : !quantum.bit
  %r0 = quantum.insert %r[0], %out : !quantum.reg, !quantum.bit
  quantum.dealloc %r0 : !quantum.reg
  return
}

// CHECK: frontend_name = "to_rz_gp"
// ESTIMATE: "quantum_operations"
// ESTIMATE-DAG: "GlobalPhase": 1
// ESTIMATE-DAG: "RZ": 1
