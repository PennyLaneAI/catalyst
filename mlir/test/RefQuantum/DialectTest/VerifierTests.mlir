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

func.func @test_controlled1(%w0: i64, %w1: i64) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{number of controlling wires in input (1) and controlling values (2) must be the same}}
    ref_quantum.custom "PauliZ"() %w0 ctrls (%w1) ctrlvals (%true, %true) : i64 ctrls i64
    return
}

// -----

func.func @test_controlled2(%w0: i64) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{number of controlling wires in input (0) and controlling values (1) must be the same}}
    ref_quantum.custom "PauliZ"() %w0 ctrls () ctrlvals (%true) : i64
    return
}

// -----

func.func @test_controlled3(%w0: i64, %w1: i64) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{number of controlling wires in input (1) and controlling values (0) must be the same}}
    ref_quantum.custom "PauliZ"() %w0 ctrls (%w1) ctrlvals () : i64 ctrls i64
    return
}

// -----

func.func @test_duplicate_wires1(%w0: i64) {
	// expected-error@+1 {{all wires on a quantum gate must be distinct (including controls)}}
    ref_quantum.custom "CNOT"() %w0, %w0 : i64, i64
    return
}

// -----

func.func @test_duplicate_wires2(%w0: i64) {
	%true = llvm.mlir.constant (1 : i1) :i1
	// expected-error@+1 {{all wires on a quantum gate must be distinct (including controls)}}
    ref_quantum.custom "PauliX"() %w0 ctrls (%w0) ctrlvals (%true) : i64 ctrls i64
    return
}

// -----

func.func @test_paulirot_length_mismatch(%w0: i64, %angle: f64) {
    // expected-error@+1 {{length of Pauli word (2) and number of wires (1) must be the same}}
    ref_quantum.paulirot ["Z", "X"](%angle) %w0 : i64
    return
}

// -----

func.func @test_paulirot_bad_pauli_word(%w0: i64, %angle: f64) {
    // expected-error@+1 {{Only "X", "Y", "Z", and "I" are valid Pauli words.}}
    ref_quantum.paulirot ["bad"](%angle) %w0 : i64
    return
}

// -----

func.func @test_paulirot_control(%w0: i64, %w1: i64, %angle: f64) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{number of controlling wires in input (1) and controlling values (2) must be the same}}
    ref_quantum.paulirot ["Z"](%angle) %w0 ctrls (%w1) ctrlvals (%true, %true) : i64 ctrls i64
    return
}

// -----

func.func @test_paulirot_duplicate_wires(%w0: i64, %angle: f64) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{all wires on a quantum gate must be distinct (including controls)}}
    ref_quantum.paulirot ["Z", "I"](%angle) %w0, %w0 : i64, i64
    return
}

// -----

func.func @test_gphase_control(%w0: i64, %param: f64) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{number of controlling wires in input (1) and controlling values (2) must be the same}}
    ref_quantum.gphase(%param) ctrls (%w0) ctrlvals (%true, %true) : f64 ctrls i64
    return
}

// -----

func.func @test_multirz_control(%w0: i64, %w1: i64, %theta: f64) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{number of controlling wires in input (1) and controlling values (2) must be the same}}
    ref_quantum.multirz(%theta) %w0 ctrls (%w1) ctrlvals (%true, %true) : i64 ctrls i64
    return
}

// -----

func.func @test_multirz_duplicate_wires(%w0: i64, %theta: f64) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{all wires on a quantum gate must be distinct (including controls)}}
    ref_quantum.multirz(%theta) %w0, %w0 : i64, i64
    return
}

// -----

func.func @test_pcphase_control(%w0: i64, %w1: i64, %theta: f64, %dim: f64) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{number of controlling wires in input (1) and controlling values (2) must be the same}}
    ref_quantum.pcphase(%theta, %dim) %w0 ctrls (%w1) ctrlvals (%true, %true) : i64 ctrls i64
    return
}

// -----

func.func @test_pcphase_duplicate_wires(%w0: i64, %theta: f64, %dim: f64) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{all wires on a quantum gate must be distinct (including controls)}}
    ref_quantum.pcphase(%theta, %dim) %w0, %w0 : i64, i64
    return
}

// -----

func.func @test_unitary_bad_matrix_shape(%w0: i64, %matrix: tensor<37x42xcomplex<f64>>) {
    // expected-error@+1 {{The Unitary matrix must be of size 2^(num_wires) * 2^(num_wires)}}
    ref_quantum.unitary (%matrix : tensor<37x42xcomplex<f64>>) %w0 : i64
    return
}

// -----

func.func @test_unitary_control(%w0: i64, %w1: i64, %matrix: tensor<2x2xcomplex<f64>>) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{number of controlling wires in input (1) and controlling values (2) must be the same}}
    ref_quantum.unitary(%matrix: tensor<2x2xcomplex<f64>>) %w0 ctrls (%w1) ctrlvals (%true, %true) : i64 ctrls i64
    return
}

// -----

func.func @test_unitary_duplicate_wires(%w0: i64, %matrix: tensor<4x4xcomplex<f64>>) {
    %true = llvm.mlir.constant (1 : i1) :i1
    // expected-error@+1 {{all wires on a quantum gate must be distinct (including controls)}}
    ref_quantum.unitary(%matrix: tensor<4x4xcomplex<f64>>) %w0, %w0 : i64, i64
    return
}

// -----

func.func @test_namedobs_op_bad_attribute(%w0: i64) {
    // expected-error@+2 {{expected catalyst::quantum::NamedObservable to be one of: Identity, PauliX, PauliY, PauliZ, Hadamard}}
    // expected-error@+1 {{failed to parse NamedObservableAttr parameter 'value' which is to be a `catalyst::quantum::NamedObservable`}}
    %0 = ref_quantum.namedobs %w0 [ bad] : !quantum.obs
    return
}
