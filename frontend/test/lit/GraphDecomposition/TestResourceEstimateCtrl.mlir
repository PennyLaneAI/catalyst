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

// A subroutine called inside a quantum.ctrl region is counted as a controlled op (2C_helper).

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(resource-estimate{output-json gate-set=PauliX=1.0,Hadamard=1.0,CNOT=1.0,T=1.0,Adjoint(T)=1.0 bytecode-rules="%PRECOMPILED_BYTECODE_PATH"})' %s -o /dev/null | FileCheck %s

// CHECK-LABEL: "2C_helper": {
// CHECK:       "quantum_operations"
// CHECK-NEXT:    "-1"
// CHECK-NEXT:      "Adjoint(T)": 3
// CHECK-NEXT:      "CNOT": 6
// CHECK-NEXT:      "Hadamard": 2
// CHECK-NEXT:      "PauliX": 2
// CHECK-NEXT:      "T": 4
func.func public @ctrl_circuit() attributes {quantum.node} {
    %true = arith.constant true
    %0 = quantum.alloc( 3) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit
    %ctrls:2, %4 = quantum.ctrl(%2, %3) ctrlvals(%true, %true) (%1) : !quantum.bit, !quantum.bit -> !quantum.bit {
    ^bb0(%arg0: !quantum.bit):
        %5 = func.call @helper(%arg0) : (!quantum.bit) -> !quantum.bit
        quantum.yield %5 : !quantum.bit
    }
    %6 = quantum.insert %0[ 0], %4 : !quantum.reg, !quantum.bit
    %7 = quantum.insert %6[ 1], %ctrls#0 : !quantum.reg, !quantum.bit
    %8 = quantum.insert %7[ 2], %ctrls#1 : !quantum.reg, !quantum.bit
    quantum.dealloc %8 : !quantum.reg
    return
}

func.func private @helper(%q: !quantum.bit) -> !quantum.bit {
    %0 = quantum.custom "PauliX"() %q : !quantum.bit
    return %0 : !quantum.bit
}

// resource-estimate only reads the target gate and resources of a rule.
func.func private @ctrl_x_to_toffoli(%q: !quantum.bit) -> !quantum.bit attributes {
    target_gate = "2C(PauliX){}{wires:1}{}",
    resources = {operations = {"PauliX{}{wires:1}{}" = 2 : i64, "Toffoli{}{wires:3}{}" = 1 : i64}}} {
    return %q : !quantum.bit
}

func.func private @toffoli_to_clifford_t(%q: !quantum.bit) -> !quantum.bit attributes {
    target_gate = "Toffoli{}{wires:3}{}",
    resources = {operations = {"Adjoint(T){}{wires:1}{}" = 3 : i64, "CNOT{}{wires:2}{}" = 6 : i64, "Hadamard{}{wires:1}{}" = 2 : i64, "T{}{wires:1}{}" = 4 : i64}}} {
    return %q : !quantum.bit
}
