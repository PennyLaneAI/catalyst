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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(resource-estimate{output-json gate-set=testRX=1.0,testRZ=1.0 bytecode-rules="%BYTECODE_PATH"})' %s -o /dev/null | FileCheck %s

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
