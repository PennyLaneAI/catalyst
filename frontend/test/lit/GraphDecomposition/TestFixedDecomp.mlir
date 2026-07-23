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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=testRX=2.0,testRY=1.0,testRZ=1.0 fixed-decomps=testHadamard=custom_decomp bytecode-rules="%BYTECODE_PATH"})' %s | FileCheck %s

func.func @circuit() -> !quantum.bit {
    %0 = quantum.alloc(1) : !quantum.reg
    %q = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
    // CHECK-NOT: testHadamard"
    // CHECK: testRX
    // CHECK: testRZ
    // CHECK: testRX
    %qout = quantum.custom "testHadamard"() %q : !quantum.bit
    return %qout : !quantum.bit
}

func.func @custom_decomp(%q0 : !quantum.bit) -> !quantum.bit {
    %cst = arith.constant 1.5707963267948966 : f64
    %q1 = quantum.custom "testRX"(%cst) %q0 : !quantum.bit
    %q2 = quantum.custom "testRZ"(%cst) %q1 : !quantum.bit
    %q3 = quantum.custom "testRX"(%cst) %q2 : !quantum.bit
    return %q3 : !quantum.bit
}
