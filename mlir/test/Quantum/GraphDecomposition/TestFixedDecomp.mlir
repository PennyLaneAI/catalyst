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

// RUN: quantum-opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=RX=2.0,RY=1.0,RZ=1.0,GlobalPhase=0.0 fixed-decomps=Hadamard=custom_decomp bytecode-rules="%BYTECODE_PATH"})' %s | FileCheck %s

func.func @circuit() -> !quantum.bit {
    %0 = quantum.alloc(1) : !quantum.reg
    %q = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
    // CHECK-NOT: Hadamard"
    // CHECK: RX
    // CHECK: RZ
    // CHECK: RX
    %qout = quantum.custom "Hadamard"() %q : !quantum.bit
    return %qout : !quantum.bit
}

func.func @custom_decomp(%q0 : !quantum.bit) -> !quantum.bit {
    %cst = arith.constant 1.5707963267948966 : f64
    %q1 = quantum.custom "RX"(%cst) %q0 : !quantum.bit
    %q2 = quantum.custom "RZ"(%cst) %q1 : !quantum.bit
    %q3 = quantum.custom "RX"(%cst) %q2 : !quantum.bit
    return %q3 : !quantum.bit
}
