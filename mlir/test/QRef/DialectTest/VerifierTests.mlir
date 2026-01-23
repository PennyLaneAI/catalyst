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

// -----

func.func @test_controlled1(%q0: !qref.qubit_ref, %q1: !qref.qubit_ref) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{number of controlling wires in input (1) and controlling values (2) must be the same}}
    qref.custom "PauliZ"() %q0 ctrls (%q1) ctrlvals (%true, %true) : !qref.qubit_ref ctrls !qref.qubit_ref
    return
}

// -----

func.func @test_controlled2(%q0: !qref.qubit_ref) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{number of controlling wires in input (0) and controlling values (1) must be the same}}
    qref.custom "PauliZ"() %q0 ctrls () ctrlvals (%true) : !qref.qubit_ref
    return
}

// -----

func.func @test_controlled3(%q0: !qref.qubit_ref, %q1: !qref.qubit_ref) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{number of controlling wires in input (1) and controlling values (0) must be the same}}
    qref.custom "PauliZ"() %q0 ctrls (%q1) ctrlvals () : !qref.qubit_ref ctrls !qref.qubit_ref
    return
}

// -----

func.func @test_duplicate_wires1(%q0: !qref.qubit_ref) {
	// expected-error@+1 {{all wires on a quantum gate must be distinct (including controls)}}
    qref.custom "CNOT"() %q0, %q0 : !qref.qubit_ref, !qref.qubit_ref
    return
}

// -----

func.func @test_duplicate_wires2(%q0: !qref.qubit_ref) {
	%true = llvm.mlir.constant (1 : i1) :i1
	// expected-error@+1 {{all wires on a quantum gate must be distinct (including controls)}}
    qref.custom "PauliX"() %q0 ctrls (%q0) ctrlvals (%true) : !qref.qubit_ref ctrls !qref.qubit_ref
    return
}

// -----

func.func @test_paulirot_length_mismatch(%q0: !qref.qubit_ref, %angle: f64) {
    // expected-error@+1 {{length of Pauli word (2) and number of wires (1) must be the same}}
    qref.paulirot ["Z", "X"](%angle) %q0 : !qref.qubit_ref
    return
}

// -----

func.func @test_paulirot_bad_pauli_word(%q0: !qref.qubit_ref, %angle: f64) {
    // expected-error@+1 {{Only "X", "Y", "Z", and "I" are valid Pauli words.}}
    qref.paulirot ["bad"](%angle) %q0 : !qref.qubit_ref
    return
}

// -----

func.func @test_paulirot_control(%q0: !qref.qubit_ref, %q1: !qref.qubit_ref, %angle: f64) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{number of controlling wires in input (1) and controlling values (2) must be the same}}
    qref.paulirot ["Z"](%angle) %q0 ctrls (%q1) ctrlvals (%true, %true) : !qref.qubit_ref ctrls !qref.qubit_ref
    return
}

// -----

func.func @test_paulirot_duplicate_wires(%q0: !qref.qubit_ref, %angle: f64) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{all wires on a quantum gate must be distinct (including controls)}}
    qref.paulirot ["Z", "I"](%angle) %q0, %q0 : !qref.qubit_ref, !qref.qubit_ref
    return
}

// -----

func.func @test_gphase_control(%q0: !qref.qubit_ref, %param: f64) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{number of controlling wires in input (1) and controlling values (2) must be the same}}
    qref.gphase(%param) ctrls (%q0) ctrlvals (%true, %true) : f64 ctrls !qref.qubit_ref
    return
}

// -----

func.func @test_multirz_control(%q0: !qref.qubit_ref, %q1: !qref.qubit_ref, %theta: f64) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{number of controlling wires in input (1) and controlling values (2) must be the same}}
    qref.multirz(%theta) %q0 ctrls (%q1) ctrlvals (%true, %true) : !qref.qubit_ref ctrls !qref.qubit_ref
    return
}

// -----

func.func @test_multirz_duplicate_wires(%q0: !qref.qubit_ref, %theta: f64) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{all wires on a quantum gate must be distinct (including controls)}}
    qref.multirz(%theta) %q0, %q0 : !qref.qubit_ref, !qref.qubit_ref
    return
}

// -----

func.func @test_pcphase_control(%q0: !qref.qubit_ref, %q1: !qref.qubit_ref, %theta: f64, %dim: f64) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{number of controlling wires in input (1) and controlling values (2) must be the same}}
    qref.pcphase(%theta, %dim) %q0 ctrls (%q1) ctrlvals (%true, %true) : !qref.qubit_ref ctrls !qref.qubit_ref
    return
}

// -----

func.func @test_pcphase_duplicate_wires(%q0: !qref.qubit_ref, %theta: f64, %dim: f64) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{all wires on a quantum gate must be distinct (including controls)}}
    qref.pcphase(%theta, %dim) %q0, %q0 : !qref.qubit_ref, !qref.qubit_ref
    return
}

// -----

func.func @test_unitary_bad_matrix_shape(%q0: !qref.qubit_ref, %matrix: tensor<37x42xcomplex<f64>>) {
    // expected-error@+1 {{The Unitary matrix must be of size 2^(num_wires) * 2^(num_wires)}}
    qref.unitary (%matrix : tensor<37x42xcomplex<f64>>) %q0 : !qref.qubit_ref
    return
}

// -----

func.func @test_unitary_control(%q0: !qref.qubit_ref, %q1: !qref.qubit_ref, %matrix: tensor<2x2xcomplex<f64>>) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{number of controlling wires in input (1) and controlling values (2) must be the same}}
    qref.unitary(%matrix: tensor<2x2xcomplex<f64>>) %q0 ctrls (%q1) ctrlvals (%true, %true) : !qref.qubit_ref ctrls !qref.qubit_ref
    return
}

// -----

func.func @test_unitary_duplicate_wires(%q0: !qref.qubit_ref, %matrix: tensor<4x4xcomplex<f64>>) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{all wires on a quantum gate must be distinct (including controls)}}
    qref.unitary(%matrix: tensor<4x4xcomplex<f64>>) %q0, %q0 : !qref.qubit_ref, !qref.qubit_ref
    return
}

// -----

func.func @test_namedobs_op_bad_attribute(%q0: !qref.qubit_ref) {
    // expected-error@+2 {{expected catalyst::quantum::NamedObservable to be one of: Identity, PauliX, PauliY, PauliZ, Hadamard}}
    // expected-error@+1 {{failed to parse NamedObservableAttr parameter 'value' which is to be a `catalyst::quantum::NamedObservable`}}
    %0 = qref.namedobs %q0 [ bad] : !quantum.obs
    return
}
