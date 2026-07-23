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

// RUN: catalyst --tool=opt --split-input-file --pass-pipeline='builtin.module( graph-decomposition{gate-set=testHadamard=1.0 fixed-decomps=testPauliX=x_to_h bytecode-rules="%BYTECODE_PATH"}, graph-decomposition{gate-set=testPauliX=1.0 fixed-decomps=testHadamard=h_to_x bytecode-rules="%BYTECODE_PATH"}, graph-decomposition{gate-set=testHadamard=1.0 fixed-decomps=testPauliX=x_to_h bytecode-rules="%BYTECODE_PATH"})' %s | FileCheck %s

func.func @circuit() -> !quantum.bit {
    %0 = quantum.alloc(2) : !quantum.reg
    %q = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
    // CHECK-NOT testPauliX
    // CHECK: testHadamard
    %qout = quantum.custom "testPauliX"() %q : !quantum.bit
    return %qout : !quantum.bit
}

// CHECK-LABEL: h_to_x
func.func @h_to_x(%q : !quantum.bit) -> !quantum.bit attributes {target_gate="testHadamard[][1]{}"} {
    %q1 = quantum.custom "testPauliX"() %q : !quantum.bit
    return %q1 : !quantum.bit
}

func.func @x_to_h(%q : !quantum.bit) -> !quantum.bit attributes {target_gate="testPauliX[][1]{}"} {
    %q1 = quantum.custom "testHadamard"() %q : !quantum.bit
    return %q1 : !quantum.bit
}
