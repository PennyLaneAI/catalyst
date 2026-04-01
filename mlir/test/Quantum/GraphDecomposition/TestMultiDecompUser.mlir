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

// RUN: quantum-opt --split-input-file --pass-pipeline='builtin.module( graph-decomposition{gate-set=Hadamard=1.0 fixed-decomps=PauliX=x_to_h bytecode-rules="%BYTECODE_PATH"}, graph-decomposition{gate-set=PauliX=1.0 fixed-decomps=Hadamard=h_to_x bytecode-rules="%BYTECODE_PATH"}, graph-decomposition{gate-set=Hadamard=1.0 fixed-decomps=PauliX=x_to_h bytecode-rules="%BYTECODE_PATH"})' %s | FileCheck %s --check-prefixes TRIPLE

func.func @circuit() -> !quantum.bit {
    %0 = quantum.alloc(2) : !quantum.reg
    %q = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
    // TRIPLE-NOT PauliX
    // TRIPLE: Hadamard
    %qout = quantum.custom "PauliX"() %q : !quantum.bit
    return %qout : !quantum.bit
}

func.func @h_to_x(%q : !quantum.bit) -> !quantum.bit attributes {target_gate="Hadamard"} {
    %q1 = quantum.custom "PauliX"() %q : !quantum.bit
    return %q1 : !quantum.bit
}

func.func @x_to_h(%q : !quantum.bit) -> !quantum.bit attributes {target_gate="PauliX"} {
    %q1 = quantum.custom "Hadamard"() %q : !quantum.bit
    return %q1 : !quantum.bit
}
