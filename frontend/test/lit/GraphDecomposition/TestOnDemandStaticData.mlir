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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=PhaseShift=1.0,GlobalPhase=1.0})' %s | FileCheck %s

// PCPhase keeps its `dim` in its static data as an integer attribute, so asking the frontend for its
// rules on demand only works if that attribute reaches Python as an int (see
// getPyvalFromMlirAttribute in PythonFunction.cpp). If it arrives as anything else, the rule fails
// to build, the solver finds nothing for the op, and the pass reports it as undecomposable.

// CHECK-LABEL: func.func @circuit
// CHECK-NOT: quantum.pcphase
// CHECK: quantum.operator "PhaseShift"
// CHECK: quantum.gphase
func.func @circuit(%theta: f64) -> !quantum.reg {
    %r = quantum.alloc(2) : !quantum.reg
    %q0 = quantum.extract %r[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %r[1] : !quantum.reg -> !quantum.bit
    %out:2 = quantum.pcphase(%theta, dim : 2) %q0, %q1 : !quantum.bit, !quantum.bit
    %r0 = quantum.insert %r[0], %out#0 : !quantum.reg, !quantum.bit
    %r1 = quantum.insert %r0[1], %out#1 : !quantum.reg, !quantum.bit
    return %r1 : !quantum.reg
}

// The rule came back keyed on the id the compiler prints for the op, `dim` included.
// CHECK: target_gate = "PCPhase{phi:[f64]}{wires:2}{dim = 2 : i64}"
