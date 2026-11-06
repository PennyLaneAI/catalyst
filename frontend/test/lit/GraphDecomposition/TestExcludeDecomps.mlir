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

// Excluding the cheaper rule forces the solver to choose the more expensive one.
// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{inline-rule-body gate-set=testRY=3.0,testX=1.0,testZ=1.0 exclude-decomps=testY=[y_to_x_z] bytecode-rules="%BYTECODE_PATH"})' %s | FileCheck %s --check-prefixes RY,KEEP

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{inline-rule-body gate-set=testRY=1.0,testX=3.0,testZ=3.0 exclude-decomps=testY=y_to_ry bytecode-rules="%BYTECODE_PATH"})' %s | FileCheck %s --check-prefixes XZ,KEEP

// An empty exclusion list allows all rules.
// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{inline-rule-body gate-set=testRY=1.0,testX=3.0,testZ=3.0 exclude-decomps=testY=[]})' %s | FileCheck %s --check-prefixes RY,KEEP

// Excluding a rule for another operator does not exclude it for testY.
// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{inline-rule-body gate-set=testRY=1.0,testX=3.0,testZ=3.0 exclude-decomps=testOther=y_to_ry})' %s | FileCheck %s --check-prefixes RY,KEEP

// Excluding all rules leaves no solution, including when a fixed rule is also excluded.
// RUN: not --crash catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=testRY=1.0,testX=1.0,testZ=1.0 exclude-decomps=testY=[y_to_ry,y_to_x_z]})' %s 2>&1 | FileCheck %s --check-prefix=FAIL
// RUN: not --crash catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=testRY=1.0 fixed-decomps=testY=y_to_ry exclude-decomps=testY=y_to_ry})' %s 2>&1 | FileCheck %s --check-prefix=FAIL
// FAIL: GraphSolverFailedError
// FAIL: Decomposition rule not found for operator 'testY{}{wires:1}{}'

func.func @circuit() {
    %0 = quantum.alloc(2) : !quantum.reg
    %q = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
    // RY-NOT: testY
    // RY: testRY

    // XZ-NOT: testY
    // XZ: testX
    // XZ: testZ
    %qout = quantum.custom "testY"() %q : !quantum.bit

    // needed to ensure we don't match in the following decomposition rules
    // RY: return
    // XZ: return
    %1 = quantum.insert %0[ 0], %qout : !quantum.reg, !quantum.bit
    quantum.dealloc %1 : !quantum.reg
    return
}

// Exclusions match frontend names even when the symbol has been renamed.
// KEEP-LABEL: func.func private @renamed_y_to_ry
// KEEP-SAME: frontend_name = "y_to_ry"
func.func private @renamed_y_to_ry(%q0 : !quantum.bit) -> !quantum.bit attributes {target_gate="testY{}{wires:1}{}", resources = { operations = {"testRY{0:[f64]}{wires:1}{}"=1}}, frontend_name = "y_to_ry"} {
    %pi = arith.constant 3.14 : f64
    %negpiby2 = arith.constant -1.57 : f64
    %q1 = quantum.custom "testRY"(%pi) %q0 : !quantum.bit
    return %q1 : !quantum.bit
}

// KEEP-LABEL: func.func private @y_to_x_z
func.func private @y_to_x_z(%q0 : !quantum.bit) -> !quantum.bit attributes {target_gate="testY{}{wires:1}{}", resources = { operations = {"testX{}{wires:1}{}"=1, "testZ{}{wires:1}{}"=1}}, frontend_name = "y_to_x_z"} {
    %q1 = quantum.custom "testX"() %q0 : !quantum.bit
    %q2 = quantum.custom "testZ"() %q1 : !quantum.bit
    return %q2 : !quantum.bit
}
