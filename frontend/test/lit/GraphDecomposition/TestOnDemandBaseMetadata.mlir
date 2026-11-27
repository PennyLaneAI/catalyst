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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{bytecode-rules= gate-set=PhaseShift=1.0,GlobalPhase=1.0,CNOT=1.0,PauliX=1.0})' %s | FileCheck %s
// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(resource-estimate{output-json bytecode-rules= gate-set=PhaseShift=1.0,GlobalPhase=1.0,CNOT=1.0,PauliX=1.0})' %s -o %t.estimated.mlir | FileCheck %s --check-prefix=ESTIMATE
// RUN: catalyst --tool=opt --empty %s -o %t.original.mlir
// RUN: diff %t.original.mlir %t.estimated.mlir

// The first representative of PCPhase already has lowered control work wires. It is unreachable,
// so only the plain variant below needs rules. Cached base metadata must discard the reserved
// modifier arguments even though the requested identity itself has no controls.
// ESTIMATE: "quantum_operations"
// ESTIMATE: "PhaseShift"
func.func private @unused(%theta: f64, %q0: !quantum.bit, %q1: !quantum.bit,
                          %work: !quantum.bit, %ctrl: !quantum.bit)
    -> (!quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit) {
  %true = arith.constant true
  %out:3, %oc = quantum.operator "PCPhase"(%theta: f64) qubits(%q0, %q1, %work)
    ctrls(%ctrl) ctrl_vals(%true)
    static_data = {ctrl_work_wire_type = "zeroed", dim = 2 : i64}
    param_map = {phi = [0]} qubit_map = {ctrl_work_wires = [2], wires = [0, 1]}
  return %out#0, %out#1, %out#2, %oc : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
}

// CHECK-LABEL: func.func @circuit
// CHECK-NOT: quantum.pcphase
// CHECK: call @__builtin_controlled_phase_shifts
// CHECK: quantum.dealloc
func.func @circuit(%theta: f64) attributes {quantum.node} {
  %r = quantum.alloc(2) : !quantum.reg
  %q0 = quantum.extract %r[0] : !quantum.reg -> !quantum.bit
  %q1 = quantum.extract %r[1] : !quantum.reg -> !quantum.bit
  %out:2 = quantum.pcphase(%theta, dim : 2) %q0, %q1 : !quantum.bit, !quantum.bit
  %r0 = quantum.insert %r[0], %out#0 : !quantum.reg, !quantum.bit
  %r1 = quantum.insert %r0[1], %out#1 : !quantum.reg, !quantum.bit
  quantum.dealloc %r1 : !quantum.reg
  return
}

// CHECK: target_gate = "PCPhase{phi:[f64]}{wires:2}{dim = 2 : i64}"
