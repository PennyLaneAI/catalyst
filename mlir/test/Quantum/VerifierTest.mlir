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

// expected-error @below {{failed to satisfy constraint: 64-bit signless integer attribute whose value is non-negative}}
%r = quantum.alloc(-1) : !quantum.reg

// -----

// expected-error @+2 {{failed to satisfy constraint: 64-bit signless integer attribute whose value is non-negative}}
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

    // expected-error@+1 {{number of qubits in input (2) and output (1) must be the same}}
    %err = quantum.custom "CNOT"() %q1, %q2 : !quantum.bit

    return
}

// -----

func.func @multirz2(%q0 : !quantum.bit, %q1 : !quantum.bit, %theta : f64) {
    // expected-error@+1 {{number of qubits in input (2) and output (1) must be the same}}
    %err = quantum.multirz(%theta) %q0, %q1 : !quantum.bit

    return
}

// -----

func.func @multirz3(%q0 : !quantum.bit, %theta : f64) {
    // expected-error@+1 {{number of qubits in input (1) and output (2) must be the same}}
    %err:2 = quantum.multirz(%theta) %q0 : !quantum.bit, !quantum.bit

    return
}

// -----

func.func @unitary2(%q0 : !quantum.bit, %q1 : !quantum.bit,  %m : tensor<4x4xcomplex<f64>>) {
    // expected-error@+1 {{number of qubits in input (2) and output (1) must be the same}}
    %err = quantum.unitary(%m: tensor<4x4xcomplex<f64>>) %q0, %q1 : !quantum.bit

    return
}

// -----

func.func @unitary3(%q0 : !quantum.bit, %q1 : !quantum.bit, %m : tensor<4x4xcomplex<f64>>) {
    // expected-error@+1 {{The Unitary matrix must be of size 2^(num_qubits) * 2^(num_qubits)}}
    quantum.unitary(%m: tensor<4x4xcomplex<f64>>) %q0 : !quantum.bit

    quantum.unitary(%m: tensor<4x4xcomplex<f64>>) %q0, %q1 : !quantum.bit, !quantum.bit

    return
}

// -----

func.func @controlled1(%1 : !quantum.bit, %2 : !quantum.bit, %3 : !quantum.bit) {
    %true = llvm.mlir.constant (1 : i1) :i1
    %cst = llvm.mlir.constant (6.000000e-01 : f64) : f64
    %cst_0 = llvm.mlir.constant (9.000000e-01 : f64) : f64
    %cst_1 = llvm.mlir.constant (3.000000e-01 : f64) : f64
    // expected-error@+1 {{number of controlling qubits in input (1) and output (0) must be the same}}
    %out_qubits:2  = quantum.custom "Rot"(%cst, %cst_1, %cst_0) %2 ctrls (%3) ctrlvals (%true) : !quantum.bit, !quantum.bit
    return
}

// -----

func.func @controlled2(%1 : !quantum.bit, %2 : !quantum.bit, %3 : !quantum.bit) {
    %true = llvm.mlir.constant (1 : i1) :i1
    %cst = llvm.mlir.constant (6.000000e-01 : f64) : f64
    %cst_0 = llvm.mlir.constant (9.000000e-01 : f64) : f64
    %cst_1 = llvm.mlir.constant (3.000000e-01 : f64) : f64
    // expected-error@+1 {{number of controlling qubits in input (2) and controlling values (1) must be the same}}
    %out_qubits:3  = quantum.custom "Rot"(%cst, %cst_1, %cst_0) %2 ctrls (%3, %3) ctrlvals (%true) : !quantum.bit, !quantum.bit, !quantum.bit
    return
}


//////////////////
// Measurements //
//////////////////

func.func @compbasis1(%q0 : !quantum.bit, %q1 : !quantum.bit, %q2 : !quantum.bit, %r : !quantum.reg) {
    %obs_q = quantum.compbasis qubits %q0, %q1, %q2 : !quantum.obs
    %obs_r = quantum.compbasis qreg %r : !quantum.obs
    return
}

// -----

func.func @compbasis2(%q0 : !quantum.bit, %q1 : !quantum.bit, %r : !quantum.reg) {
    // expected-error@+1 {{computational basis op cannot simultaneously take in both qubits and quregs}}
    %obs = quantum.compbasis qubits %q0, %q1 qreg %r : !quantum.obs

    return
}

// -----

func.func @namedobs(%q : !quantum.bit) {
    %0 = quantum.namedobs %q[Identity] : !quantum.obs
    %1 = quantum.namedobs %q[PauliX] : !quantum.obs
    %2 = quantum.namedobs %q[PauliY] : !quantum.obs
    %3 = quantum.namedobs %q[PauliZ] : !quantum.obs
    %4 = quantum.namedobs %q[Hadamard] : !quantum.obs

    // expected-error@+2 {{expected valid keyword}}
    // expected-error@+1 {{failed to parse NamedObservableAttr parameter}}
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
    %o1 = quantum.namedobs %q0[Identity] :  !quantum.obs
    %o2 = quantum.namedobs %q1[PauliY] :  !quantum.obs
    %o3 = quantum.namedobs %q2[Hadamard] :  !quantum.obs
    %obs = quantum.tensor %o1, %o2, %o3 : !quantum.obs

    return
}

// -----

func.func @sample1(%q : !quantum.bit) {
    %obs = quantum.compbasis qubits %q : !quantum.obs
    %alloc = memref.alloc() : memref<1000xf64>

    // expected-error@+1 {{either tensors must be returned or memrefs must be used as inputs}}
    quantum.sample %obs in (%alloc : memref<1000xf64>) : tensor<1000xf64>

    return
}

// -----

func.func @sample2(%q : !quantum.bit) {
    %obs = quantum.compbasis qubits %q : !quantum.obs

    // expected-error@+1 {{either tensors must be returned or memrefs must be used as inputs}}
    quantum.sample %obs

    return
}

// -----

func.func @sample3(%q0 : !quantum.bit, %q1 : !quantum.bit, %c : i64) {
    %obs = quantum.compbasis qubits %q0, %q1 : !quantum.obs

    // expected-error@+1 {{with static return shapes should not specify dynamic shape in arguments}}
    quantum.sample %obs shape %c : tensor<2x2xf64>

    return
}

// -----

func.func @sample4(%q0 : !quantum.bit, %q1 : !quantum.bit) {
    %obs = quantum.compbasis qubits %q0, %q1 : !quantum.obs

    // expected-error@+1 {{with dynamic return shapes must specify dynamic shape in arguments}}
    quantum.sample %obs : tensor<4x?xf64>

    return
}

// -----

func.func @sample_good(%q0 : !quantum.bit, %q1 : !quantum.bit, %c : i64, %in_sample1 : memref<1x?xf64>, %in_sample2 : memref<1x4xf64>) {
    %obs = quantum.compbasis qubits %q0, %q1 : !quantum.obs

    // smoke test for good cases
    quantum.sample %obs shape %c in(%in_sample1 : memref<1x?xf64>)
    quantum.sample %obs in(%in_sample1 : memref<1x?xf64>)
    quantum.sample %obs in(%in_sample2 : memref<1x4xf64>)
    quantum.sample %obs : tensor<1x4xf64>
    quantum.sample %obs shape %c : tensor<1x?xf64>
    quantum.sample %obs shape %c : tensor<?x?xf64>
    return
}

// -----

func.func @counts1(%q0 : !quantum.bit, %q1 : !quantum.bit) {
    %obs = quantum.namedobs %q0[PauliX] : !quantum.obs

    %counts:2 = quantum.counts %obs : tensor<2xf64>, tensor<2xi64>

    // expected-error@+1 {{number of eigenvalues or counts did not match observable}}
    %err:2 = quantum.counts %obs : tensor<4xf64>, tensor<4xi64>

    return
}

// -----

func.func @counts2(%q0 : !quantum.bit, %q1 : !quantum.bit) {
    %obs = quantum.compbasis qubits %q0, %q1 : !quantum.obs

    %counts:2 = quantum.counts %obs : tensor<4xf64>, tensor<4xi64>

    // expected-error@+1 {{number of eigenvalues or counts did not match observable}}
    %err:2 = quantum.counts %obs : tensor<2xf64>, tensor<2xi64>

    return
}

// -----

func.func @counts3(%q0 : !quantum.bit, %q1 : !quantum.bit) {
    %obs = quantum.namedobs %q0[PauliX] : !quantum.obs

    %in_eigvals_1 = memref.alloc() : memref<4xf64>
    %in_counts_1 = memref.alloc() : memref<4xi64>

    %in_eigvals_2 = memref.alloc() : memref<2xf64>
    %in_counts_2 = memref.alloc() : memref<2xi64>
    quantum.counts %obs in(%in_eigvals_2 : memref<2xf64>, %in_counts_2 : memref<2xi64>)

    // expected-error@+1 {{number of eigenvalues or counts did not match observable}}
    quantum.counts %obs in(%in_eigvals_1 : memref<4xf64>, %in_counts_1 : memref<4xi64>)

    return
}

// -----

func.func @counts4(%q0 : !quantum.bit, %q1 : !quantum.bit) {
    %obs = quantum.namedobs %q0[PauliX] : !quantum.obs

    // expected-error@+1 {{either tensors must be returned or memrefs must be used as inputs}}
    quantum.counts %obs

    return
}

// -----

func.func @counts5(%q0 : !quantum.bit, %q1 : !quantum.bit) {
    %obs = quantum.namedobs %q0[PauliX] : !quantum.obs

    %in_eigvals = memref.alloc() : memref<2xf64>
    %in_counts = memref.alloc() : memref<2xi64>
    // expected-error@+1 {{either tensors must be returned or memrefs must be used as inputs}}
    quantum.counts %obs in(%in_eigvals : memref<2xf64>, %in_counts : memref<2xi64>) : tensor<2xf64>, tensor<2xi64>

    return
}

// -----

func.func @counts6(%q0 : !quantum.bit, %q1 : !quantum.bit, %c : i64) {
    %obs = quantum.compbasis qubits %q0, %q1 : !quantum.obs

    // expected-error@+1 {{with static return shapes should not specify dynamic shape in arguments}}
    quantum.counts %obs shape %c : tensor<4xf64>, tensor<4xi64>

    return
}

// -----

func.func @counts7(%q0 : !quantum.bit, %q1 : !quantum.bit) {
    %obs = quantum.compbasis qubits %q0, %q1 : !quantum.obs

    // expected-error@+1 {{with dynamic return shapes must specify dynamic shape in arguments}}
    quantum.counts %obs : tensor<?xf64>, tensor<?xi64>

    return
}

// -----

func.func @probs1(%q0 : !quantum.bit, %q1 : !quantum.bit) {
    %obs = quantum.compbasis qubits %q0, %q1 : !quantum.obs

    %c4 = arith.constant 4 : i64
    %c4i = index.casts %c4 : i64 to index
    %in_probs1 = memref.alloc(%c4i) : memref<?xf64>
    // expected-error@+1 {{either tensors must be returned or memrefs must be used as inputs}}
    quantum.probs %obs in(%in_probs1 : memref<?xf64>) : tensor<?xf64>

    return
}

// -----

func.func @probs2(%q0 : !quantum.bit, %q1 : !quantum.bit) {
    %obs = quantum.compbasis qubits %q0, %q1 : !quantum.obs

    // expected-error@+1 {{either tensors must be returned or memrefs must be used as inputs}}
    quantum.probs %obs

    return
}

// -----

func.func @probs3(%q0 : !quantum.bit, %q1 : !quantum.bit, %c : i64) {
    %obs = quantum.compbasis qubits %q0, %q1 : !quantum.obs

    // expected-error@+1 {{with static return shapes should not specify dynamic shape in arguments}}
    quantum.probs %obs shape %c : tensor<4xf64>

    return
}

// -----

func.func @probs4(%q0 : !quantum.bit, %q1 : !quantum.bit) {
    %obs = quantum.compbasis qubits %q0, %q1 : !quantum.obs

    // expected-error@+1 {{with dynamic return shapes must specify dynamic shape in arguments}}
    quantum.probs %obs : tensor<?xf64>

    return
}

// -----

func.func @probs_good(%q0 : !quantum.bit, %q1 : !quantum.bit, %c : i64, %in_probs1 : memref<?xf64>, %in_probs2 : memref<4xf64>) {
    %obs = quantum.compbasis qubits %q0, %q1 : !quantum.obs

    // smoke test for good cases
    quantum.probs %obs shape %c in(%in_probs1 : memref<?xf64>)
    quantum.probs %obs in(%in_probs1 : memref<?xf64>)
    quantum.probs %obs in(%in_probs2 : memref<4xf64>)
    quantum.probs %obs : tensor<4xf64>
    quantum.probs %obs shape %c : tensor<?xf64>
    return
}

// -----

func.func @state1(%q0 : !quantum.bit, %q1 : !quantum.bit) {
    %obs = quantum.compbasis qubits %q0, %q1 : !quantum.obs

    %c4 = arith.constant 4 : i64
    %c4i = index.casts %c4 : i64 to index
    %in_state1 = memref.alloc(%c4i) : memref<?xcomplex<f64>>
    // expected-error@+1 {{either tensors must be returned or memrefs must be used as inputs}}
    quantum.state %obs in(%in_state1 : memref<?xcomplex<f64>>) : tensor<?xcomplex<f64>>

    return
}

// -----

func.func @state2(%q0 : !quantum.bit, %q1 : !quantum.bit) {
    %obs = quantum.compbasis qubits %q0, %q1 : !quantum.obs

    // expected-error@+1 {{either tensors must be returned or memrefs must be used as inputs}}
    quantum.state %obs

    return
}

// -----

func.func @state3(%q0 : !quantum.bit, %q1 : !quantum.bit, %c : i64) {
    %obs = quantum.compbasis qubits %q0, %q1 : !quantum.obs

    // expected-error@+1 {{with static return shapes should not specify dynamic shape in arguments}}
    quantum.state %obs shape %c : tensor<4xcomplex<f64>>

    return
}

// -----

func.func @state4(%q0 : !quantum.bit, %q1 : !quantum.bit) {
    %obs = quantum.compbasis qubits %q0, %q1 : !quantum.obs

    // expected-error@+1 {{with dynamic return shapes must specify dynamic shape in arguments}}
    quantum.state %obs : tensor<?xcomplex<f64>>

    return
}

// -----

func.func @state_good(%q0 : !quantum.bit, %q1 : !quantum.bit, %c : i64, %in_state1 : memref<?xcomplex<f64>>, %in_state2 : memref<4xcomplex<f64>>) {
    %obs = quantum.compbasis qubits %q0, %q1 : !quantum.obs

    // smoke test for good cases
    quantum.state %obs shape %c in(%in_state1 : memref<?xcomplex<f64>>)
    quantum.state %obs in(%in_state1 : memref<?xcomplex<f64>>)
    quantum.state %obs in(%in_state2 : memref<4xcomplex<f64>>)
    quantum.state %obs : tensor<4xcomplex<f64>>
    quantum.state %obs shape %c : tensor<?xcomplex<f64>>
    return
}
