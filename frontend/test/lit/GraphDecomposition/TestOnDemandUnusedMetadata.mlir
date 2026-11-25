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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(resource-estimate{output-json gate-set=PhaseShift=1.0,GlobalPhase=1.0})' %s | FileCheck %s

// Resolve metadata only for requested roots. The unreachable gate's symbol attribute cannot
// be converted to Python, but must not prevent loading PCPhase's rules.
// CHECK: "quantum_operations"
// CHECK: "PhaseShift"
func.func @circuit(%theta: f64, %reg: !quantum.reg) -> !quantum.reg attributes {quantum.node} {
    %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[1] : !quantum.reg -> !quantum.bit
    %out:2 = quantum.pcphase(%theta, dim : 2) %q0, %q1 : !quantum.bit, !quantum.bit
    %r0 = quantum.insert %reg[0], %out#0 : !quantum.reg, !quantum.bit
    %r1 = quantum.insert %r0[1], %out#1 : !quantum.reg, !quantum.bit
    return %r1 : !quantum.reg
}

func.func private @unused(%qubit: !quantum.bit) -> !quantum.bit {
    %out = quantum.operator "S"() qubits(%qubit) static_data = {unsupported = @unused}
        param_map = {} qubit_map = {wires = [0]}
    return %out : !quantum.bit
}
