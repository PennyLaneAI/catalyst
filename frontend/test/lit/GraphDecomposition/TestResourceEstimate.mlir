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

// Test that resource-estimate counts each op by the basis gates of its chosen decomposition

// RUN: quantum-opt --split-input-file --pass-pipeline='builtin.module(resource-estimate{output-json gate-set=testRX=1.0,testRZ=1.0,S=1.0,T=1.0,Adjoint(S)=1.0,Adjoint(T)=1.0,PauliX=1.0,Hadamard=1.0,CNOT=1.0 bytecode-rules="%BYTECODE_PATH"})' %s -o /dev/null | FileCheck %s

// CHECK-LABEL: "circuit"
// CHECK: "num_qubits"
// CHECK:   "alloc": 1
// CHECK:   "arg": 0
// CHECK:   "total": 1
// CHECK: "quantum_operations"
// CHECK:   "-1"
// CHECK:       "testRX": 3
// CHECK:       "testRZ": 4
// CHECK-NOT:   "testHadamard"
func.func public @circuit() attributes {quantum.node} {
    %angle = arith.constant 0.5 : f64
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.custom "testHadamard"() %1 : !quantum.bit
    %3 = quantum.custom "testHadamard"() %2 : !quantum.bit
    %4 = quantum.custom "testRX"(%angle) %3 : !quantum.bit
    %5 = quantum.insert %0[ 0], %4 : !quantum.reg, !quantum.bit
    quantum.dealloc %5 : !quantum.reg
    return
}

// -----

// An adjoint applied to a subroutine also applies to the subroutines it calls.

// CHECK-LABEL: "A_inner": {
// CHECK:         "Adjoint(S)": 1
// CHECK-LABEL: "A_outer": {
// CHECK:         "A_inner": 1
// CHECK:         "Adjoint(T)": 1
// CHECK-LABEL: "inner": {
// CHECK:         "S": 1
// CHECK-LABEL: "outer": {
// CHECK:         "inner": 1
// CHECK:         "T": 1
func.func public @adjoint_circuit() attributes {quantum.node} {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.adjoint(%1) : !quantum.bit {
    ^bb0(%arg0: !quantum.bit):
        %3 = func.call @outer(%arg0) : (!quantum.bit) -> !quantum.bit
        quantum.yield %3 : !quantum.bit
    }
    %4 = func.call @outer(%2) : (!quantum.bit) -> !quantum.bit
    %5 = quantum.insert %0[ 0], %4 : !quantum.reg, !quantum.bit
    quantum.dealloc %5 : !quantum.reg
    return
}

func.func private @outer(%q: !quantum.bit) -> !quantum.bit {
    %0 = func.call @inner(%q) : (!quantum.bit) -> !quantum.bit
    %1 = quantum.custom "T"() %0 : !quantum.bit
    return %1 : !quantum.bit
}

func.func private @inner(%q: !quantum.bit) -> !quantum.bit {
    %0 = quantum.custom "S"() %q : !quantum.bit
    return %0 : !quantum.bit
}

// -----

// A subroutine called inside a quantum.ctrl region is counted as a controlled op (2C_helper).

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
