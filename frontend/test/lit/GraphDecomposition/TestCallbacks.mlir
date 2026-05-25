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

// Test that decomposition chooses cheapest decomposition path

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=RX=1.0,Hadamard=1.0,MultiRZ=1.0,GlobalPhase=1.0 bytecode-rules="%BYTECODE_PATH"})' %s | FileCheck %s


func.func @circuit() -> (!quantum.bit, !quantum.bit, !quantum.bit) {
    %0 = quantum.alloc(3) : !quantum.reg
    %q0 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %0[1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %0[2] : !quantum.reg -> !quantum.bit

    %pi = arith.constant 3.1 : f64
    %qout:3 = quantum.paulirot ["Y", "X", "Z"](%pi) %q0, %q1, %q2 : !quantum.bit, !quantum.bit, !quantum.bit

    // CHECK-NOT: quantum.paulirot

    // CHECK-DAG: RX
    // CHECK-DAG: Hadamard
    // CHECK: multirz
    // CHECK-DAG: RX
    // CHECK-DAG: Hadamard

    return %qout#0, %qout#1, %qout#2 : !quantum.bit, !quantum.bit, !quantum.bit
}
