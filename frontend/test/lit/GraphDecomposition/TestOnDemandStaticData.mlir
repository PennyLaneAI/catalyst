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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=PhaseShift=1.0,GlobalPhase=1.0,CNOT=1.0,PauliX=1.0})' %s | FileCheck %s --check-prefixes=ALL,CALL

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{inline-rule-body gate-set=PhaseShift=1.0,GlobalPhase=1.0,CNOT=1.0,PauliX=1.0})' %s | FileCheck %s --check-prefixes=ALL,INLINE

// PCPhase keeps its `dim` in its static data as an integer attribute, so asking the frontend for its
// rules on demand only works if that attribute reaches Python as an int (see
// getPyvalFromMlirAttribute in PythonFunction.cpp). If it arrives as anything else, the rule fails
// to build, the solver finds nothing for the op, and the pass reports it as undecomposable.

// ALL-LABEL: func.func @circuit
// ALL-NOT: quantum.pcphase
// CALL: call @__builtin__decompose_pcphase{{(_[0-9]+)*}}
// CALL: call @__builtin__decompose_pcphase{{(_[0-9]+)*}}
// INLINE: quantum.custom "PhaseShift"
// INLINE-NOT: quantum.pcphase
// INLINE: quantum.dealloc
func.func @circuit(%theta: f64) attributes {quantum.node} {
    %r = quantum.alloc(2) : !quantum.reg
    %q0 = quantum.extract %r[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %r[1] : !quantum.reg -> !quantum.bit
    %out:2 = quantum.pcphase(%theta, dim : 2) %q0, %q1 : !quantum.bit, !quantum.bit
    %out2:2 = quantum.pcphase(%theta, dim : 1) %out#0, %out#1 : !quantum.bit, !quantum.bit
    %r0 = quantum.insert %r[0], %out2#0 : !quantum.reg, !quantum.bit
    %r1 = quantum.insert %r0[1], %out2#1 : !quantum.reg, !quantum.bit
    quantum.dealloc %r1 : !quantum.reg
    return
}

// Both roots have the same frontend rule name, but distinct static identities. Their closures
// share resource rules, and must coexist with this existing symbol without dropping alternatives.
func.func private @__builtin__decompose_pcphase() {
    return
}

// The rules come back keyed on the ids the compiler prints for the ops, `dim` included.
// ALL-DAG: target_gate = "PCPhase{phi:[f64]}{wires:2}{dim = 2 : i64}"
// ALL-DAG: target_gate = "PCPhase{phi:[f64]}{wires:2}{dim = 1 : i64}"
