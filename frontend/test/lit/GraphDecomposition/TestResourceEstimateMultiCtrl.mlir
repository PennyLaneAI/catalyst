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

// Test that resource-estimate counts gates with different numbers of control qubits separately,
// with the number of control qubits in their names, as resource-analysis does. This holds both for
// gates in the gate set and for the gates of a decomposition.

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(resource-estimate{output-json gate-set=C(RX)=1.0,C(PauliX)=1.0,C(Adjoint(S))=1.0 bytecode-rules="%PRECOMPILED_BYTECODE_PATH"})' %s -o /dev/null | FileCheck %s

// CHECK-LABEL: "circuit"
// CHECK:       "quantum_operations"
// CHECK-NEXT:    "-1"
// CHECK-NEXT:      "1C(RX)": 1
// CHECK-NEXT:      "2C(Adjoint(S))": 1
// CHECK-NEXT:      "2C(PauliX)": 2
// CHECK-NEXT:      "2C(RX)": 1
// CHECK-NEXT:      "3C(Adjoint(S))": 1
// CHECK-NEXT:      "3C(RX)": 1
func.func public @circuit() attributes {quantum.node} {
    %angle = arith.constant 0.5 : f64
    %true = arith.constant true
    %0 = quantum.alloc( 4) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit
    %4 = quantum.extract %0[ 3] : !quantum.reg -> !quantum.bit
    %5:3 = quantum.custom "RX"(%angle) %4 ctrls(%1, %2) ctrlvals(%true, %true) : !quantum.bit ctrls !quantum.bit, !quantum.bit
    %6:4 = quantum.custom "RX"(%angle) %5#0 ctrls(%5#1, %5#2, %3) ctrlvals(%true, %true, %true) : !quantum.bit ctrls !quantum.bit, !quantum.bit, !quantum.bit
    %7:3 = quantum.custom "S"() %6#0 adj ctrls(%6#1, %6#2) ctrlvals(%true, %true) : !quantum.bit ctrls !quantum.bit, !quantum.bit
    %8:3 = quantum.custom "MyOp"() %7#0 ctrls(%7#1, %7#2) ctrlvals(%true, %true) : !quantum.bit ctrls !quantum.bit, !quantum.bit
    %9:2 = quantum.custom "RX"(%angle) %8#0 ctrls(%6#3) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %10 = quantum.insert %0[ 0], %8#1 : !quantum.reg, !quantum.bit
    %11 = quantum.insert %10[ 1], %8#2 : !quantum.reg, !quantum.bit
    %12 = quantum.insert %11[ 2], %9#1 : !quantum.reg, !quantum.bit
    %13 = quantum.insert %12[ 3], %9#0 : !quantum.reg, !quantum.bit
    quantum.dealloc %13 : !quantum.reg
    return
}

// resource-estimate only reads the target gate and resources of a rule.
func.func private @ctrl_my_op(%q: !quantum.bit) -> !quantum.bit attributes {
    target_gate = "2C(MyOp){}{wires:1}{}",
    resources = {operations = {"2C(PauliX){}{wires:1}{}" = 2 : i64, "3C(Adjoint(S)){}{wires:1}{}" = 1 : i64}}} {
    return %q : !quantum.bit
}
