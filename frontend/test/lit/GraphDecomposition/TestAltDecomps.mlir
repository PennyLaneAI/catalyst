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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=RY=1.0,PauliX=3.0,PauliZ=3.0,GlobalPhase=1.0 alt-decomps=PauliY=[y_to_ry,y_to_x_z] bytecode-rules="%BYTECODE_PATH"})' %s | FileCheck %s --check-prefixes RY

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=RY=3.0,PauliX=1.0,PauliZ=1.0,GlobalPhase=1.0 alt-decomps=PauliY=[y_to_ry,y_to_x_z] bytecode-rules="%BYTECODE_PATH"})' %s | FileCheck %s --check-prefixes XZ

func.func @circuit() -> !quantum.bit {
    %0 = quantum.alloc(2) : !quantum.reg
    %q = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
    // RY-NOT: PauliY
    // RY: RY
    // RY: gphase

    // XZ-NOT: PauliY
    // XZ: PauliX
    // XZ: PauliZ
    %qout = quantum.custom "PauliY"() %q : !quantum.bit

    // needed to ensure we don't match in the following decomposition rules
    // CHECK: return
    return %qout : !quantum.bit
}

// CHECK-LABEL: y_to_ry
func.func @y_to_ry(%q0 : !quantum.bit) -> !quantum.bit attributes {target_gate="PauliY[][1]{}", resources = { operations = {"RY[f64][1]{}"=1, "GlobalPhase[][]{}"=1}}} {
    %pi = arith.constant 3.14 : f64
    %negpiby2 = arith.constant -1.57 : f64
    %q1 = quantum.custom "RY"(%pi) %q0 : !quantum.bit
    quantum.gphase(%negpiby2)
    return %q1 : !quantum.bit
}

// CHECK-LABEL: y_to_x_z
func.func @y_to_x_z(%q0 : !quantum.bit) -> !quantum.bit attributes {target_gate="PauliY[][1]{}", resources = { operations = {"PauliX[][1]{}"=1, "PauliZ[][1]{}"=1}}} {
    %q1 = quantum.custom "PauliX"() %q0 : !quantum.bit
    %q2 = quantum.custom "PauliZ"() %q1 : !quantum.bit
    return %q2 : !quantum.bit
}
