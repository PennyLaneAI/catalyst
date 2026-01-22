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

// RUN: quantum-opt --lower-qec-init-ops --split-input-file --verify-diagnostics %s | FileCheck %s

// Test lowering qec.prepare zero to quantum.set_state
func.func @test_prepare_zero(%q : !quantum.bit) -> !quantum.bit {
    %0 = qec.prepare zero %q : !quantum.bit
    return %0 : !quantum.bit

    // CHECK-LABEL: func.func @test_prepare_zero
    // CHECK: [[STATE:%.+]] = arith.constant dense<[(1.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)]> : tensor<2xcomplex<f64>>
    // CHECK: [[OUT:%.+]] = quantum.set_state([[STATE]]) {{.*}} : (tensor<2xcomplex<f64>>, !quantum.bit) -> !quantum.bit
    // CHECK: return [[OUT]]
}

// -----

// Test lowering qec.prepare one to quantum.set_state
func.func @test_prepare_one(%q : !quantum.bit) -> !quantum.bit {
    %0 = qec.prepare one %q : !quantum.bit
    return %0 : !quantum.bit

    // CHECK-LABEL: func.func @test_prepare_one
    // CHECK: [[STATE:%.+]] = arith.constant dense<[(0.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00)]> : tensor<2xcomplex<f64>>
    // CHECK: [[OUT:%.+]] = quantum.set_state([[STATE]]) {{.*}} : (tensor<2xcomplex<f64>>, !quantum.bit) -> !quantum.bit
    // CHECK: return [[OUT]]
}

// -----

// Test lowering qec.prepare plus to quantum.set_state
func.func @test_prepare_plus(%q : !quantum.bit) -> !quantum.bit {
    %0 = qec.prepare plus %q : !quantum.bit
    return %0 : !quantum.bit

    // CHECK-LABEL: func.func @test_prepare_plus
    // CHECK: [[STATE:%.+]] = arith.constant dense<(0.707{{.*}},0.000000e+00)> : tensor<2xcomplex<f64>>
    // CHECK: [[OUT:%.+]] = quantum.set_state([[STATE]]) {{.*}} : (tensor<2xcomplex<f64>>, !quantum.bit) -> !quantum.bit
    // CHECK: return [[OUT]]
}

// -----

// Test lowering qec.prepare minus to quantum.set_state
func.func @test_prepare_minus(%q : !quantum.bit) -> !quantum.bit {
    %0 = qec.prepare minus %q : !quantum.bit
    return %0 : !quantum.bit

    // CHECK-LABEL: func.func @test_prepare_minus
    // CHECK: [[STATE:%.+]] = arith.constant dense<[(0.707{{.*}},0.000000e+00), (-0.707{{.*}},0.000000e+00)]> : tensor<2xcomplex<f64>>
    // CHECK: [[OUT:%.+]] = quantum.set_state([[STATE]]) {{.*}} : (tensor<2xcomplex<f64>>, !quantum.bit) -> !quantum.bit
    // CHECK: return [[OUT]]
}

// -----

// Test lowering qec.prepare plus_i to quantum.set_state
func.func @test_prepare_plus_i(%q : !quantum.bit) -> !quantum.bit {
    %0 = qec.prepare plus_i %q : !quantum.bit
    return %0 : !quantum.bit

    // CHECK-LABEL: func.func @test_prepare_plus_i
    // CHECK: [[STATE:%.+]] = arith.constant dense<[(0.707{{.*}},0.000000e+00), (0.000000e+00,0.707{{.*}})]> : tensor<2xcomplex<f64>>
    // CHECK: [[OUT:%.+]] = quantum.set_state([[STATE]]) {{.*}} : (tensor<2xcomplex<f64>>, !quantum.bit) -> !quantum.bit
    // CHECK: return [[OUT]]
}

// -----

// Test lowering qec.prepare minus_i to quantum.set_state
func.func @test_prepare_minus_i(%q : !quantum.bit) -> !quantum.bit {
    %0 = qec.prepare minus_i %q : !quantum.bit
    return %0 : !quantum.bit

    // CHECK-LABEL: func.func @test_prepare_minus_i
    // CHECK: [[STATE:%.+]] = arith.constant dense<[(0.707{{.*}},0.000000e+00), (0.000000e+00,-0.707{{.*}})]> : tensor<2xcomplex<f64>>
    // CHECK: [[OUT:%.+]] = quantum.set_state([[STATE]]) {{.*}} : (tensor<2xcomplex<f64>>, !quantum.bit) -> !quantum.bit
    // CHECK: return [[OUT]]
}

// -----

// Test lowering qec.fabricate magic to quantum.alloc_qb + quantum.set_state
func.func @test_fabricate_magic() -> !quantum.bit {
    %0 = qec.fabricate magic : !quantum.bit
    return %0 : !quantum.bit

    // CHECK-LABEL: func.func @test_fabricate_magic
    // CHECK: [[STATE:%.+]] = arith.constant dense<[(0.707{{.*}},0.000000e+00), (5.000000e-01,5.000000e-01)]> : tensor<2xcomplex<f64>>
    // CHECK: [[Q:%.+]] = quantum.alloc_qb : !quantum.bit
    // CHECK: [[OUT:%.+]] = quantum.set_state([[STATE]]) [[Q]] : (tensor<2xcomplex<f64>>, !quantum.bit) -> !quantum.bit
    // CHECK: return [[OUT]]
}

// -----

// Test lowering qec.fabricate magic_conj to quantum.alloc_qb + quantum.set_state
func.func @test_fabricate_magic_conj() -> !quantum.bit {
    %0 = qec.fabricate magic_conj : !quantum.bit
    return %0 : !quantum.bit

    // CHECK-LABEL: func.func @test_fabricate_magic_conj
    // CHECK: [[STATE:%.+]] = arith.constant dense<[(0.707{{.*}},0.000000e+00), (5.000000e-01,-5.000000e-01)]> : tensor<2xcomplex<f64>>
    // CHECK: [[Q:%.+]] = quantum.alloc_qb : !quantum.bit
    // CHECK: [[OUT:%.+]] = quantum.set_state([[STATE]]) [[Q]] : (tensor<2xcomplex<f64>>, !quantum.bit) -> !quantum.bit
    // CHECK: return [[OUT]]
}

// -----

// Test lowering qec.fabricate plus_i to quantum.alloc_qb + quantum.set_state
func.func @test_fabricate_plus_i() -> !quantum.bit {
    %0 = qec.fabricate plus_i : !quantum.bit
    return %0 : !quantum.bit

    // CHECK-LABEL: func.func @test_fabricate_plus_i
    // CHECK: [[STATE:%.+]] = arith.constant dense<[(0.707{{.*}},0.000000e+00), (0.000000e+00,0.707{{.*}})]> : tensor<2xcomplex<f64>>
    // CHECK: [[Q:%.+]] = quantum.alloc_qb : !quantum.bit
    // CHECK: [[OUT:%.+]] = quantum.set_state([[STATE]]) [[Q]] : (tensor<2xcomplex<f64>>, !quantum.bit) -> !quantum.bit
    // CHECK: return [[OUT]]
}

// -----

// Test lowering qec.prepare with multiple qubits
func.func @test_prepare_multiple_qubits(%q1 : !quantum.bit, %q2 : !quantum.bit) -> (!quantum.bit, !quantum.bit) {
    %0, %1 = qec.prepare zero %q1, %q2 : !quantum.bit, !quantum.bit
    return %0, %1 : !quantum.bit, !quantum.bit

    // CHECK-LABEL: func.func @test_prepare_multiple_qubits
    // CHECK: [[STATE:%.+]] = arith.constant dense<[(1.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)]> : tensor<2xcomplex<f64>>
    // CHECK: [[OUT1:%.+]] = quantum.set_state([[STATE]]) {{.*}} : (tensor<2xcomplex<f64>>, !quantum.bit) -> !quantum.bit
    // CHECK: [[OUT2:%.+]] = quantum.set_state([[STATE]]) {{.*}} : (tensor<2xcomplex<f64>>, !quantum.bit) -> !quantum.bit
    // CHECK: return [[OUT1]], [[OUT2]]
}

// -----

// Test lowering qec.fabricate with multiple qubits
func.func @test_fabricate_multiple_qubits() -> (!quantum.bit, !quantum.bit) {
    %0, %1 = qec.fabricate magic : !quantum.bit, !quantum.bit
    return %0, %1 : !quantum.bit, !quantum.bit

    // CHECK-LABEL: func.func @test_fabricate_multiple_qubits
    // CHECK: [[STATE:%.+]] = arith.constant dense<[(0.707{{.*}},0.000000e+00), (5.000000e-01,5.000000e-01)]> : tensor<2xcomplex<f64>>
    // CHECK: [[OUT1:%.+]] = quantum.set_state([[STATE]]) {{.*}} : (tensor<2xcomplex<f64>>, !quantum.bit) -> !quantum.bit
    // CHECK: [[OUT2:%.+]] = quantum.set_state([[STATE]]) {{.*}} : (tensor<2xcomplex<f64>>, !quantum.bit) -> !quantum.bit
    // CHECK: return [[OUT1]], [[OUT2]]
}
