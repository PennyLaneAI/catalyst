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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=testRX=2.0,testRY=1.0,testRZ=1.0 fixed-decomps=testHadamard=fixed_decomp})' %s | FileCheck %s

func.func @circuit() {
    %q = quantum.alloc_qb : !quantum.bit
    // CHECK-NOT: testHadamard"
    // CHECK: testRX
    // CHECK: testRZ
    // CHECK: testRX
    %qout = quantum.custom "testHadamard"() %q : !quantum.bit
    quantum.dealloc_qb %qout : !quantum.bit
    return 
}

// CHECK: @fixed_decomp
func.func @fixed_decomp(%q0 : !quantum.bit) -> !quantum.bit attributes {target_gate = "testHadamard{}{wires:1}{}", frontend_name = "fixed_decomp", resources = { operations = { "testRX{0:[f64]}{wires:1}{}"=2, "testRZ{0:[f64]}{wires:1}{}"=1}}} {
    %cst = arith.constant 1.5707963267948966 : f64
    %q1 = quantum.custom "testRX"(%cst) %q0 : !quantum.bit
    %q2 = quantum.custom "testRZ"(%cst) %q1 : !quantum.bit
    %q3 = quantum.custom "testRX"(%cst) %q2 : !quantum.bit
    return %q3 : !quantum.bit
}

// CHECK: @cheaper_decomp
func.func @cheaper_decomp(%q0 : !quantum.bit) -> !quantum.bit attributes {target_gate = "testHadamard{}{wires:1}{}", frontend_name = "cheaper_decomp", resources = { operations = { "testRX{0:[f64]}{wires:1}{}"}} } {
    %cst = arith.constant 1.5707963267948966 : f64
    %q1 = quantum.custom "testRX"(%cst) %q0 : !quantum.bit
    return %q1 : !quantum.bit
}
