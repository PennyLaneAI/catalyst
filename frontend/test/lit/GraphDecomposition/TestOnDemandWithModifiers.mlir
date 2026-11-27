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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=PhaseShift=1.0,GlobalPhase=1.0})' %s | FileCheck %s --check-prefixes=ALL,CALL

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{inline-rule-body gate-set=PhaseShift=1.0,GlobalPhase=1.0})' %s | FileCheck %s --check-prefixes=ALL,INLINE

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(resource-estimate{output-json gate-set=PhaseShift=1.0,GlobalPhase=1.0})' %s -o %t.estimated.mlir | FileCheck %s --check-prefix=ESTIMATE
// RUN: catalyst --tool=opt --empty %s -o %t.original.mlir
// RUN: diff %t.original.mlir %t.estimated.mlir

// The same helper is reached with and without an adjoint. Resource analysis supplies both
// identities from one unmodified operation; on-demand loading must query each context.
// Resource estimation must load those rules without changing the input IR.
// ESTIMATE: "quantum_operations"
// ESTIMATE: "PhaseShift"

// ALL-LABEL: func.func private @phase.adjoint(
// ALL-NOT: quantum.pcphase
// CALL: call @__builtin_
// INLINE: quantum.custom "PhaseShift"
// ALL: return
// ALL-LABEL: func.func @circuit
// ALL-NOT: quantum.adjoint
// ALL: call @phase.adjoint
// ALL: call @phase(
// ALL: quantum.dealloc
func.func @circuit(%theta: f64) attributes {quantum.node} {
    %r = quantum.alloc(2) : !quantum.reg
    %out = quantum.adjoint(%r) : !quantum.reg {
    ^bb0(%arg0: !quantum.reg):
      %0 = func.call @phase(%theta, %arg0) : (f64, !quantum.reg) -> !quantum.reg
      quantum.yield %0 : !quantum.reg
    }
    %out2 = func.call @phase(%theta, %out) : (f64, !quantum.reg) -> !quantum.reg
    quantum.dealloc %out2 : !quantum.reg
    return
}

// ALL-LABEL: func.func private @phase(
// ALL-NOT: quantum.pcphase
// CALL: call @__builtin_
// INLINE: quantum.custom "PhaseShift"
// ALL: return
func.func private @phase(%theta: f64, %r: !quantum.reg) -> !quantum.reg {
    %q0 = quantum.extract %r[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %r[1] : !quantum.reg -> !quantum.bit
    %out:2 = quantum.pcphase(%theta, dim : 2) %q0, %q1 : !quantum.bit, !quantum.bit
    %r0 = quantum.insert %r[0], %out#0 : !quantum.reg, !quantum.bit
    %r1 = quantum.insert %r0[1], %out#1 : !quantum.reg, !quantum.bit
    return %r1 : !quantum.reg
}

// ALL-DAG: target_gate = "Adjoint(PCPhase){phi:[f64]}{wires:2}{dim = 2 : i64}"
// ALL-DAG: target_gate = "PCPhase{phi:[f64]}{wires:2}{dim = 2 : i64}"
