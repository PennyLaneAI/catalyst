// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt %s --split-input-file --verify-diagnostics

//////////////////////
// MemoryManagement //
//////////////////////

%c0 = arith.constant 0 : i64
%c5 = arith.constant 5 : i64

%r1 = quantum.alloc(%c5) : !quantum.reg
%r2 = quantum.alloc(5) : !quantum.reg

%q1 = quantum.extract %r1[%c0] : !quantum.reg -> !quantum.bit
%q2 = quantum.extract %r1[1] : !quantum.reg -> !quantum.bit

quantum.insert %r1[%c0], %q1 : !quantum.reg, !quantum.bit
quantum.insert %r1[1], %q2 : !quantum.reg, !quantum.bit

quantum.dealloc %r1 : !quantum.reg

// -----

// expected-error @below {{expected attribute value}}
%r = quantum.alloc() : !quantum.reg

// -----

// expected-error @below {{failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 1}}
%r = quantum.alloc(0) : !quantum.reg

// -----

// expected-error @+2 {{failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0}}
%r = quantum.alloc(5) : !quantum.reg
%q = quantum.extract %r[-1] : !quantum.reg -> !quantum.bit

// -----

///////////
// Gates //
///////////

func.func @custom(%f : f64, %q1 : !quantum.bit, %q2 : !quantum.bit) {
    %q3 = quantum.custom "Hadamard"() %q1 : !quantum.bit
    %q4 = quantum.custom "RZ"(%f) %q1 : !quantum.bit
    %q5, %q6 = quantum.custom "CNOT"() %q1, %q2 : !quantum.bit, !quantum.bit

    // expected-error@+1 {{number of qubits in input and output must be the same}}
    %err = quantum.custom "CNOT"() %q1, %q2 : !quantum.bit

    return
}

// -----

func.func @multirz1(%theta : f64) {
    // expected-error@+1 {{must have at least 1 qubit}}
    %err = quantum.multirz(%theta) : !quantum.bit

    return
}

// -----

func.func @multirz2(%q0 : !quantum.bit, %q1 : !quantum.bit, %theta : f64) {
    // expected-error@+1 {{number of qubits in input and output must be the same}}
    %err = quantum.multirz(%theta) %q0, %q1 : !quantum.bit

    return
}

// -----

func.func @multirz3(%q0 : !quantum.bit, %theta : f64) {
    // expected-error@+1 {{number of qubits in input and output must be the same}}
    %err:2 = quantum.multirz(%theta) %q0 : !quantum.bit, !quantum.bit

    return
}

// -----

func.func @unitary1(%m : tensor<4x4xcomplex<f64>>) {
    // expected-error@+1 {{'quantum.unitary' op must have at least 1 qubit}}
    %err = quantum.unitary(%m: tensor<4x4xcomplex<f64>>) : !quantum.bit

    return
}

// -----

func.func @unitary2(%q0 : !quantum.bit, %q1 : !quantum.bit,  %m : tensor<4x4xcomplex<f64>>) {
    // expected-error@+1 {{number of qubits in input and output must be the same}}
    %err = quantum.unitary(%m: tensor<4x4xcomplex<f64>>) %q0, %q1 : !quantum.bit

    return
}

// -----

func.func @unitary3(%q0 : !quantum.bit, %q1 : !quantum.bit, %m : tensor<4x4xcomplex<f64>>) {
    // expected-error@+1 {{'quantum.unitary' op The Unitary matrix must be of size 2^(num_qubits) * 2^(num_qubits)}}
    quantum.unitary(%m: tensor<4x4xcomplex<f64>>) %q0 : !quantum.bit

    quantum.unitary(%m: tensor<4x4xcomplex<f64>>) %q0, %q1 : !quantum.bit, !quantum.bit

    return
}


//////////////////
// Measurements //
//////////////////

func.func @compbasis(%q0 : !quantum.bit, %q1 : !quantum.bit, %q2 : !quantum.bit) {
    %obs = quantum.compbasis %q0, %q1, %q2 : !quantum.obs

    return
}

// -----

func.func @namedobs(%q : !quantum.bit) {
    %0 = quantum.namedobs %q[0] : !quantum.obs // Identity
    %1 = quantum.namedobs %q[1] : !quantum.obs // PauliX
    %2 = quantum.namedobs %q[2] : !quantum.obs // PauliY
    %3 = quantum.namedobs %q[3] : !quantum.obs // PauliZ
    %4 = quantum.namedobs %q[4] : !quantum.obs // Hadamard

    // expected-error@+1 {{8-bit signless integer attribute whose minimum value is 0 whose maximum value is 4}}
    %err = quantum.namedobs %q[5] : !quantum.obs  // namedobs range from 0-4 for I, X, Y, Z, H

    return
}

// -----

func.func @hermitian(%q : !quantum.bit, %m1 : tensor<1x1xcomplex<f64>>, %m2 : tensor<2x2xcomplex<f64>>) {
    // expected-error@+1 {{'quantum.hermitian' op The Hermitian matrix must be of size 2^(num_qubits) * 2^(num_qubits)}}
    %0 = quantum.hermitian(%m1: tensor<1x1xcomplex<f64>>) %q : !quantum.obs

    %1 = quantum.hermitian(%m2: tensor<2x2xcomplex<f64>>) %q : !quantum.obs

    return
}

// -----

func.func @tensorobs(%q0 : !quantum.bit, %q1 : !quantum.bit, %q2 : !quantum.bit) {
    %o1 = quantum.namedobs %q0[0] :  !quantum.obs
    %o2 = quantum.namedobs %q1[2] :  !quantum.obs
    %o3 = quantum.namedobs %q2[4] :  !quantum.obs
    %obs = quantum.tensor %o1, %o2, %o3 : !quantum.obs

    return
}

// -----

func.func @sample1(%q : !quantum.bit) {
    %obs = quantum.namedobs %q[0] : !quantum.obs

    // expected-error@+1 {{return tensor must have 1D static shape equal to (number of shots)}}
    %err = quantum.sample %obs { shots=1000 } : tensor<1xf64>

    %samples = quantum.sample %obs { shots=1000 } : tensor<1000xf64>

    return
}

// -----

func.func @sample2(%q : !quantum.bit) {
    %obs = quantum.compbasis %q : !quantum.obs

    // expected-error@+1 {{return tensor must have 2D static shape equal to (number of shots, number of qubits in observable)}}
    %err = quantum.sample %obs { shots=1000 } : tensor<1000xf64>

    %samples = quantum.sample %obs { shots=1000 } : tensor<1000x1xf64>

    return
}

// -----

func.func @sample3(%q : !quantum.bit) {
    %obs = quantum.compbasis %q : !quantum.obs

    %alloc0 = memref.alloc() : memref<1000xf64>
    // expected-error@+1 {{return tensor must have 2D static shape equal to (number of shots, number of qubits in observable)}}
    quantum.sample %obs in(%alloc0 : memref<1000xf64>) { shots = 1000 }

    %alloc1 = memref.alloc() : memref<1000x1xf64>
    quantum.sample %obs in(%alloc1 : memref<1000x1xf64>) { shots = 1000 }

    return
}

// -----

func.func @sample4(%q : !quantum.bit) {
    %obs = quantum.compbasis %q : !quantum.obs

    %alloc = memref.alloc() : memref<1000xf64>
    // expected-error@+1 {{cannot name an operation with no results}}
    %err = quantum.sample %obs { shots=1000 } in (%alloc : memref<1000xf64>) : tensor<1000xf64>

    %samples = quantum.sample %obs { shots=1000 } : tensor<1000x1xf64>

    return
}

// -----

func.func @counts1(%q0 : !quantum.bit, %q1 : !quantum.bit) {
    %obs = quantum.namedobs %q0[1] : !quantum.obs

    // expected-error@+1 {{number of eigenvalues or counts did not match observable}}
    %err:2 = quantum.counts %obs { shots=1000 } : tensor<4xf64>, tensor<4xi64>

    %counts:2 = quantum.counts %obs { shots=1000 } : tensor<2xf64>, tensor<2xi64>

    return
}

// -----

func.func @counts2(%q0 : !quantum.bit, %q1 : !quantum.bit) {
    %obs = quantum.compbasis %q0, %q1 : !quantum.obs

    // expected-error@+1 {{number of eigenvalues or counts did not match observable}}
    %err:2 = quantum.counts %obs { shots=1000 } : tensor<2xf64>, tensor<2xi64>

    %counts:2 = quantum.counts %obs { shots=1000 } : tensor<4xf64>, tensor<4xi64>

    return
}

// -----

func.func @counts3(%q0 : !quantum.bit, %q1 : !quantum.bit) {
    %obs = quantum.namedobs %q0[1] : !quantum.obs

    %in_eigvals_1 = memref.alloc() : memref<4xf64>
    %in_counts_1 = memref.alloc() : memref<4xi64>
    // expected-error@+1 {{number of eigenvalues or counts did not match observable}}
    quantum.counts %obs in(%in_eigvals_1 : memref<4xf64>, %in_counts_1 : memref<4xi64>) { shots=1000 }

    %in_eigvals_2 = memref.alloc() : memref<2xf64>
    %in_counts_2 = memref.alloc() : memref<2xi64>
    quantum.counts %obs in(%in_eigvals_2 : memref<2xf64>, %in_counts_2 : memref<2xi64>) { shots=1000 }

    return
}

// -----

func.func @probs1(%q0 : !quantum.bit, %q1 : !quantum.bit) {
    %obs = quantum.compbasis %q0, %q1 : !quantum.obs

    // expected-error@+1 {{return tensor must have static length equal to 2^(number of qubits)}}
    %err = quantum.probs %obs : tensor<2xf64>

    %probs = quantum.probs %obs : tensor<4xf64>

    return
}

// -----

func.func @probs2(%q0 : !quantum.bit, %q1 : !quantum.bit) {
    %obs = quantum.compbasis %q0, %q1 : !quantum.obs

    %in_probs1 = memref.alloc() : memref<2xf64>
    // expected-error@+1 {{return tensor must have static length equal to 2^(number of qubits)}}
    quantum.probs %obs in(%in_probs1 : memref<2xf64>)

    %in_probs2 = memref.alloc() : memref<4xf64>
    quantum.probs %obs in(%in_probs2 : memref<4xf64>)

    return
}

// -----

func.func @state1(%q0 : !quantum.bit, %q1 : !quantum.bit) {
    %obs = quantum.compbasis %q0, %q1 : !quantum.obs

    // expected-error@+1 {{return tensor must have static length equal to 2^(number of qubits)}}
    %err = quantum.state %obs : tensor<?xcomplex<f64>>

    %state = quantum.state %obs : tensor<4xcomplex<f64>>

    return
}

// -----

func.func @state2(%q0 : !quantum.bit, %q1 : !quantum.bit) {
    %obs = quantum.compbasis %q0, %q1 : !quantum.obs

    %alloc1 = memref.alloc() : memref<2xcomplex<f64>>
    // expected-error@+1 {{return tensor must have static length equal to 2^(number of qubits)}}
    quantum.state %obs in(%alloc1 : memref<2xcomplex<f64>>)

    %alloc2 = memref.alloc() : memref<4xcomplex<f64>>
    quantum.state %obs in(%alloc2 : memref<4xcomplex<f64>>)

    return
}
