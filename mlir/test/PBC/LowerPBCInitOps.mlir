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

// RUN: quantum-opt --lower-pbc-init-ops --split-input-file --verify-diagnostics %s | FileCheck %s

// Test lowering pbc.prepare zero (no gates needed)
func.func @test_prepare_zero(%q : !quantum.bit) -> !quantum.bit {
    %0 = pbc.prepare zero %q : !quantum.bit
    return %0 : !quantum.bit

    // CHECK-LABEL: func.func @test_prepare_zero
    // CHECK-SAME: (%[[Q:.*]]: !quantum.bit)
    // CHECK: return %[[Q]]
}

// -----

// Test lowering pbc.prepare one to PauliX gate
func.func @test_prepare_one(%q : !quantum.bit) -> !quantum.bit {
    %0 = pbc.prepare one %q : !quantum.bit
    return %0 : !quantum.bit

    // CHECK-LABEL: func.func @test_prepare_one
    // CHECK: [[OUT:%.+]] = quantum.custom "PauliX"() {{.*}} : !quantum.bit
    // CHECK: return [[OUT]]
}

// -----

// Test lowering pbc.prepare plus to Hadamard gate
func.func @test_prepare_plus(%q : !quantum.bit) -> !quantum.bit {
    %0 = pbc.prepare plus %q : !quantum.bit
    return %0 : !quantum.bit

    // CHECK-LABEL: func.func @test_prepare_plus
    // CHECK: [[OUT:%.+]] = quantum.custom "Hadamard"() {{.*}} : !quantum.bit
    // CHECK: return [[OUT]]
}

// -----

// Test lowering pbc.prepare minus to Hadamard + PauliZ gates
func.func @test_prepare_minus(%q : !quantum.bit) -> !quantum.bit {
    %0 = pbc.prepare minus %q : !quantum.bit
    return %0 : !quantum.bit

    // CHECK-LABEL: func.func @test_prepare_minus
    // CHECK: [[H:%.+]] = quantum.custom "Hadamard"() {{.*}} : !quantum.bit
    // CHECK: [[OUT:%.+]] = quantum.custom "PauliZ"() [[H]] : !quantum.bit
    // CHECK: return [[OUT]]
}

// -----

// Test lowering pbc.prepare plus_i to Hadamard + S gates
func.func @test_prepare_plus_i(%q : !quantum.bit) -> !quantum.bit {
    %0 = pbc.prepare plus_i %q : !quantum.bit
    return %0 : !quantum.bit

    // CHECK-LABEL: func.func @test_prepare_plus_i
    // CHECK: [[H:%.+]] = quantum.custom "Hadamard"() {{.*}} : !quantum.bit
    // CHECK: [[OUT:%.+]] = quantum.custom "S"() [[H]] : !quantum.bit
    // CHECK: return [[OUT]]
}

// -----

// Test lowering pbc.prepare minus_i to Hadamard + S† gates
func.func @test_prepare_minus_i(%q : !quantum.bit) -> !quantum.bit {
    %0 = pbc.prepare minus_i %q : !quantum.bit
    return %0 : !quantum.bit

    // CHECK-LABEL: func.func @test_prepare_minus_i
    // CHECK: [[H:%.+]] = quantum.custom "Hadamard"() {{.*}} : !quantum.bit
    // CHECK: [[OUT:%.+]] = quantum.custom "S"() [[H]] adj : !quantum.bit
    // CHECK: return [[OUT]]
}

// -----

// Test lowering pbc.fabricate magic to quantum.alloc_qb + Hadamard + T
func.func @test_fabricate_magic() -> !quantum.bit {
    %0 = pbc.fabricate magic : !quantum.bit
    return %0 : !quantum.bit

    // CHECK-LABEL: func.func @test_fabricate_magic
    // CHECK: [[Q:%.+]] = quantum.alloc_qb : !quantum.bit
    // CHECK: [[H:%.+]] = quantum.custom "Hadamard"() [[Q]] : !quantum.bit
    // CHECK: [[OUT:%.+]] = quantum.custom "T"() [[H]] : !quantum.bit
    // CHECK: return [[OUT]]
}

// -----

// Test lowering pbc.fabricate magic_conj to quantum.alloc_qb + Hadamard + T†
func.func @test_fabricate_magic_conj() -> !quantum.bit {
    %0 = pbc.fabricate magic_conj : !quantum.bit
    return %0 : !quantum.bit

    // CHECK-LABEL: func.func @test_fabricate_magic_conj
    // CHECK: [[Q:%.+]] = quantum.alloc_qb : !quantum.bit
    // CHECK: [[H:%.+]] = quantum.custom "Hadamard"() [[Q]] : !quantum.bit
    // CHECK: [[OUT:%.+]] = quantum.custom "T"() [[H]] adj : !quantum.bit
    // CHECK: return [[OUT]]
}

// -----

// Test lowering pbc.fabricate plus_i to quantum.alloc_qb + Hadamard + S
func.func @test_fabricate_plus_i() -> !quantum.bit {
    %0 = pbc.fabricate plus_i : !quantum.bit
    return %0 : !quantum.bit

    // CHECK-LABEL: func.func @test_fabricate_plus_i
    // CHECK: [[Q:%.+]] = quantum.alloc_qb : !quantum.bit
    // CHECK: [[H:%.+]] = quantum.custom "Hadamard"() [[Q]] : !quantum.bit
    // CHECK: [[OUT:%.+]] = quantum.custom "S"() [[H]] : !quantum.bit
    // CHECK: return [[OUT]]
}

// -----

// Test lowering pbc.prepare with multiple qubits
func.func @test_prepare_multiple_qubits(%q1 : !quantum.bit, %q2 : !quantum.bit) -> (!quantum.bit, !quantum.bit) {
    %0, %1 = pbc.prepare plus %q1, %q2 : !quantum.bit, !quantum.bit
    return %0, %1 : !quantum.bit, !quantum.bit

    // CHECK-LABEL: func.func @test_prepare_multiple_qubits
    // CHECK-SAME: (%[[Q1:.*]]: !quantum.bit, %[[Q2:.*]]: !quantum.bit)
    // CHECK: [[OUT1:%.+]] = quantum.custom "Hadamard"() %[[Q1]] : !quantum.bit
    // CHECK: [[OUT2:%.+]] = quantum.custom "Hadamard"() %[[Q2]] : !quantum.bit
    // CHECK: return [[OUT1]], [[OUT2]]
}

// -----

// Test lowering pbc.fabricate with multiple qubits
func.func @test_fabricate_multiple_qubits() -> (!quantum.bit, !quantum.bit) {
    %0, %1 = pbc.fabricate magic : !quantum.bit, !quantum.bit
    return %0, %1 : !quantum.bit, !quantum.bit

    // CHECK-LABEL: func.func @test_fabricate_multiple_qubits
    // CHECK: [[Q1:%.+]] = quantum.alloc_qb : !quantum.bit
    // CHECK: [[H1:%.+]] = quantum.custom "Hadamard"() [[Q1]] : !quantum.bit
    // CHECK: [[OUT1:%.+]] = quantum.custom "T"() [[H1]] : !quantum.bit
    // CHECK: [[Q2:%.+]] = quantum.alloc_qb : !quantum.bit
    // CHECK: [[H2:%.+]] = quantum.custom "Hadamard"() [[Q2]] : !quantum.bit
    // CHECK: [[OUT2:%.+]] = quantum.custom "T"() [[H2]] : !quantum.bit
    // CHECK: return [[OUT1]], [[OUT2]]
}
