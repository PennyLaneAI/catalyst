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

// Test that decomposition handles circuits that are already decomposed to the given gateset.

// RUN: quantum-opt --split-input-file --pass-pipeline='builtin.module(graph-decomposition{gate-set=Hadamard=1.0,CNOT=1.0 alt-decomps=Hadamard=false_decomp bytecode-rules="%BYTECODE_PATH"})' %s | FileCheck %s

func.func @circuit() -> !quantum.bit {
    %0 = quantum.alloc(2) : !quantum.reg
    %q0 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %0[1] : !quantum.reg -> !quantum.bit
    // CHECK: Hadamard
    // CHECK: Hadamard
    // CHECK: CNOT
    %q0out = quantum.custom "Hadamard"() %q0 : !quantum.bit
    %q1out = quantum.custom "Hadamard"() %q1 : !quantum.bit
    %q:2 = quantum.custom "CNOT"() %q0out, %q1out : !quantum.bit, !quantum.bit
    return %q1 : !quantum.bit
}

// -----

module @test_module {
    func.func public @circuit() -> !quantum.bit {
        %0 = quantum.alloc(2) : !quantum.reg
        %q0 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
        %q1 = quantum.extract %0[1] : !quantum.reg -> !quantum.bit
        // CHECK: Hadamard
        // CHECK: Hadamard
        // CHECK: CNOT
        %q0out = quantum.custom "Hadamard"() %q0 : !quantum.bit
        %q1out = quantum.custom "Hadamard"() %q1 : !quantum.bit
        %q:2 = quantum.custom "CNOT"() %q0out, %q1out : !quantum.bit, !quantum.bit
        return %q1 : !quantum.bit
    }

    func.func private @false_decomp(%q : !quantum.bit) -> !quantum.bit attributes {target_gate="Hadamard"} {
        %qout = quantum.custom "PauliX"() %q : !quantum.bit
        return %qout : !quantum.bit
    }
}
