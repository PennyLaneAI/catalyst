// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Test verifiers.
//
// RUN: quantum-opt --split-input-file --verify-diagnostics %s

func.func @test_controlled1(%q0: !qref.bit, %q1: !qref.bit) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{number of controlling qubits in input (1) and controlling values (2) must be the same}}
    qref.custom "PauliZ"() %q0 ctrls (%q1) ctrlvals (%true, %true) : !qref.bit ctrls !qref.bit
    return
}

// -----

func.func @test_controlled2(%q0: !qref.bit) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{number of controlling qubits in input (0) and controlling values (1) must be the same}}
    qref.custom "PauliZ"() %q0 ctrls () ctrlvals (%true) : !qref.bit
    return
}

// -----

func.func @test_controlled3(%q0: !qref.bit, %q1: !qref.bit) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{number of controlling qubits in input (1) and controlling values (0) must be the same}}
    qref.custom "PauliZ"() %q0 ctrls (%q1) ctrlvals () : !qref.bit ctrls !qref.bit
    return
}

// -----

func.func @test_duplicate_qubits1(%q0: !qref.bit) {
	// expected-error@+1 {{all qubits on a quantum gate must be distinct (including controls)}}
    qref.custom "CNOT"() %q0, %q0 : !qref.bit, !qref.bit
    return
}

// -----

func.func @test_duplicate_qubits2(%q0: !qref.bit) {
	%true = llvm.mlir.constant (1 : i1) :i1
	// expected-error@+1 {{all qubits on a quantum gate must be distinct (including controls)}}
    qref.custom "PauliX"() %q0 ctrls (%q0) ctrlvals (%true) : !qref.bit ctrls !qref.bit
    return
}

// -----

func.func @test_paulirot_length_mismatch(%q0: !qref.bit, %angle: f64) {
    // expected-error@+1 {{length of Pauli word (2) and number of qubits (1) must be the same}}
    qref.paulirot ["Z", "X"](%angle) %q0 : !qref.bit
    return
}

// -----

func.func @test_paulirot_bad_pauli_word(%q0: !qref.bit, %angle: f64) {
    // expected-error@+1 {{Only "X", "Y", "Z", and "I" are valid Pauli words.}}
    qref.paulirot ["bad"](%angle) %q0 : !qref.bit
    return
}

// -----

func.func @test_paulirot_control(%q0: !qref.bit, %q1: !qref.bit, %angle: f64) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{number of controlling qubits in input (1) and controlling values (2) must be the same}}
    qref.paulirot ["Z"](%angle) %q0 ctrls (%q1) ctrlvals (%true, %true) : !qref.bit ctrls !qref.bit
    return
}

// -----

func.func @test_paulirot_duplicate_qubits(%q0: !qref.bit, %angle: f64) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{all qubits on a quantum gate must be distinct (including controls)}}
    qref.paulirot ["Z", "I"](%angle) %q0, %q0 : !qref.bit, !qref.bit
    return
}

// -----

func.func @test_gphase_control(%q0: !qref.bit, %param: f64) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{number of controlling qubits in input (1) and controlling values (2) must be the same}}
    qref.gphase(%param) ctrls (%q0) ctrlvals (%true, %true) : f64 ctrls !qref.bit
    return
}

// -----

func.func @test_multirz_control(%q0: !qref.bit, %q1: !qref.bit, %theta: f64) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{number of controlling qubits in input (1) and controlling values (2) must be the same}}
    qref.multirz(%theta) %q0 ctrls (%q1) ctrlvals (%true, %true) : !qref.bit ctrls !qref.bit
    return
}

// -----

func.func @test_multirz_duplicate_qubits(%q0: !qref.bit, %theta: f64) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{all qubits on a quantum gate must be distinct (including controls)}}
    qref.multirz(%theta) %q0, %q0 : !qref.bit, !qref.bit
    return
}

// -----

func.func @test_pcphase_control(%q0: !qref.bit, %q1: !qref.bit, %theta: f64, %dim: f64) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{number of controlling qubits in input (1) and controlling values (2) must be the same}}
    qref.pcphase(%theta, %dim) %q0 ctrls (%q1) ctrlvals (%true, %true) : !qref.bit ctrls !qref.bit
    return
}

// -----

func.func @test_pcphase_duplicate_qubits(%q0: !qref.bit, %theta: f64, %dim: f64) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{all qubits on a quantum gate must be distinct (including controls)}}
    qref.pcphase(%theta, %dim) %q0, %q0 : !qref.bit, !qref.bit
    return
}

// -----

func.func @test_unitary_bad_matrix_shape(%q0: !qref.bit, %matrix: tensor<37x42xcomplex<f64>>) {
    // expected-error@+1 {{The Unitary matrix must be of size 2^(num_qubits) * 2^(num_qubits)}}
    qref.unitary (%matrix : tensor<37x42xcomplex<f64>>) %q0 : !qref.bit
    return
}

// -----

func.func @test_unitary_control(%q0: !qref.bit, %q1: !qref.bit, %matrix: tensor<2x2xcomplex<f64>>) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{number of controlling qubits in input (1) and controlling values (2) must be the same}}
    qref.unitary(%matrix: tensor<2x2xcomplex<f64>>) %q0 ctrls (%q1) ctrlvals (%true, %true) : !qref.bit ctrls !qref.bit
    return
}

// -----

func.func @test_unitary_duplicate_qubits(%q0: !qref.bit, %matrix: tensor<4x4xcomplex<f64>>) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{all qubits on a quantum gate must be distinct (including controls)}}
    qref.unitary(%matrix: tensor<4x4xcomplex<f64>>) %q0, %q0 : !qref.bit, !qref.bit
    return
}

// -----

func.func @test_namedobs_op_bad_attribute(%q0: !qref.bit) {
    // expected-error@+2 {{expected catalyst::quantum::NamedObservable to be one of: Identity, PauliX, PauliY, PauliZ, Hadamard}}
    // expected-error@+1 {{failed to parse NamedObservableAttr parameter 'value' which is to be a `catalyst::quantum::NamedObservable`}}
    %0 = qref.namedobs %q0 [ bad] : !quantum.obs
    return
}

// -----

func.func @test_alloc_bad_dynamic_size(%arg0 : i64) {
    // expected-error@+1 {{expected result to have dynamic allocation size !qref.qreg<?>}}
    %0 = qref.alloc(%arg0) : !qref.reg<20>
    return
}

// -----

func.func @test_alloc_bad_static_size() {
    // expected-error@+1 {{expected result to have allocation size !qref.qreg<2>}}
    %0 = qref.alloc(2) : !qref.reg<37>
    return
}

// -----

func.func @test_adjoint_op_no_MP(%r: !qref.reg<2>)
{
    // expected-error@+1 {{quantum measurements are not allowed in the adjoint regions}}
    qref.adjoint(%r) : !qref.reg<2> {
    ^bb0(%arg0: !qref.reg<2>):
        %q1 = qref.get %arg0[1] : !qref.reg<2> -> !qref.bit
        %obs = qref.namedobs %q1 [ PauliX] : !quantum.obs
        %expval = quantum.expval %obs : f64
    }
    return
}
