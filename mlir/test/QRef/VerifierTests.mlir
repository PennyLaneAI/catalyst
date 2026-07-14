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
    qref.gphase(%param) ctrls (%q0) ctrlvals (%true, %true) : ctrls !qref.bit
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
    // expected-error@+1 {{expected result to have static allocation size !qref.qreg<2>}}
    %0 = qref.alloc(2) : !qref.reg<37>
    return
}

// -----

func.func @test_adjoint_op_no_MP(%r: !qref.reg<2>)
{
    // expected-error@+1 {{quantum measurements are not allowed in the adjoint regions}}
    qref.adjoint {
    ^bb0():
        %q1 = qref.get %r[1] : !qref.reg<2> -> !qref.bit
        %obs = qref.namedobs %q1 [ PauliX] : !quantum.obs
        %expval = quantum.expval %obs : f64
    }
    return
}

// -----

func.func @test_adjoint_with_args(%r: !qref.reg<2>, %q: !qref.bit)
{
    // expected-error@+1 {{qref.adjoint op must have no arguments on its block}}
    qref.adjoint {
    ^bb0(%arg0: !qref.reg<2>):
        %q1 = qref.get %arg0[1] : !qref.reg<2> -> !qref.bit
        qref.custom "Hadamard"() %q1 : !qref.bit
    }
    return
}

// -----

func.func @test_hermitian_bad_matrix_shape(%q0: !qref.bit, %matrix: tensor<20x20xcomplex<f64>>) {
    // expected-error@+1 {{The Hermitian matrix must be of size 2^(num_qubits) * 2^(num_qubits)}}
    %obs = qref.hermitian(%matrix : tensor<20x20xcomplex<f64>>) %q0 : !quantum.obs
    return
}

// -----


//////////////////////
// qref.operator //
//////////////////////


func.func @operator_basic_qubits(%q0 : !qref.bit, %q1 : !qref.bit) {
    "qref.operator"(%q0, %q1) <{op_name = "basic_qubits", operandSegmentSizes = array<i32: 0, 0, 2, 0, 0, 0, 0, 0, 0>}> : (!qref.bit, !qref.bit) -> ()
    return
}

// -----

func.func @operator_custom_basic_qubits(%q0 : !qref.bit, %q1 : !qref.bit) {
    qref.operator "custom_basic_qubits"() qubits(%q0, %q1)
    return
}

// -----

func.func @operator_with_control_qubits(%q : !qref.bit, %cq : !qref.bit, %cv : i1) {
    "qref.operator"(%q, %cq, %cv) <{op_name = "ctrl_qubits", operandSegmentSizes = array<i32: 0, 0, 1, 1, 1, 0, 0, 0, 0>}> : (!qref.bit, !qref.bit, i1) -> ()
    return
}

// -----

func.func @operator_custom_with_control_qubits(%q : !qref.bit, %cq : !qref.bit, %cv : i1) {
    qref.operator "custom_ctrl_qubits"() qubits(%q)
      ctrls(%cq) ctrl_vals(%cv)
    return
}

// -----

func.func @operator_with_registers(%r : !qref.reg<6>, %idx0 : tensor<2xi64>, %idx1 : tensor<1xi64>) {
    "qref.operator"(%r, %idx0, %idx1) <{op_name = "with_registers", operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 1, 2, 0, 0>}> : (!qref.reg<6>, tensor<2xi64>, tensor<1xi64>) -> ()
    return
}

// -----

func.func @operator_custom_with_registers(%r : !qref.reg<?>, %idx0 : tensor<2xi64>, %idx1 : tensor<1xi64>) {
    qref.operator "custom_with_registers"()
      quregs(%r : !qref.reg<?>) indices(%idx0 : tensor<2xi64>, %idx1 : tensor<1xi64>)
    return
}

// -----

func.func @operator_with_registers_and_controls(%r : !qref.reg<4>, %idx : tensor<2xi64>, %cidx : tensor<2xi64>, %cval : tensor<2xi1>) {
    "qref.operator"(%r, %idx, %cidx, %cval) <{op_name = "reg_ctrls", operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 1, 1, 1, 1>}> : (!qref.reg<4>, tensor<2xi64>, tensor<2xi64>, tensor<2xi1>) -> ()
    return
}

// -----

func.func @operator_custom_with_registers_and_controls(%r : !qref.reg<?>, %idx : tensor<2xi64>, %cidx : tensor<2xi64>, %cval : tensor<2xi1>) {
    qref.operator "custom_reg_ctrls"()
      quregs(%r : !qref.reg<?>) indices(%idx : tensor<2xi64>)
      ctrls(%cidx : tensor<2xi64>) ctrl_vals(%cval : tensor<2xi1>)
    return
}

// -----

func.func @operator_qubits_with_maps(%p0 : f64, %p1 : i64, %q0 : !qref.bit, %q1 : !qref.bit) {
    "qref.operator"(%p0, %p1, %q0, %q1) <{op_name = "qubit_maps", param_map = {p0 = array<i64: 0>, p1 = array<i64: 1>}, qubit_map = {pair = array<i64: 0, 1>}, operandSegmentSizes = array<i32: 2, 0, 2, 0, 0, 0, 0, 0, 0>}> : (f64, i64, !qref.bit, !qref.bit) -> ()
    return
}

// -----

func.func @operator_custom_qubits_with_maps(%p0 : f64, %p1 : i64, %q0 : !qref.bit, %q1 : !qref.bit) {
    qref.operator "custom_qubit_maps"(%p0 : f64, %p1 : i64) qubits(%q0, %q1)
      param_map = {p0 = [0], p1 = [1]}
      qubit_map = {pair = [0, 1]}
    return
}

// -----

func.func @operator_registers_with_maps(%p0 : f64, %p1 : i64, %r : !qref.reg<3>, %idx0 : tensor<2xi64>, %idx1 : tensor<1xi64>) {
    "qref.operator"(%p0, %p1, %r, %idx0, %idx1) <{op_name = "reg_maps", param_map = {p0 = array<i64: 0>, p1 = array<i64: 1>}, qubit_map = {qi0 = array<i64: 0>, qi1 = array<i64: 1>}, operandSegmentSizes = array<i32: 2, 0, 0, 0, 0, 1, 2, 0, 0>}> : (f64, i64, !qref.reg<3>, tensor<2xi64>, tensor<1xi64>) -> ()
    return
}

// -----

func.func @operator_custom_registers_with_maps(%p0 : f64, %p1 : i64, %r : !qref.reg<?>, %idx0 : tensor<2xi64>, %idx1 : tensor<1xi64>) {
    qref.operator "custom_reg_maps"(%p0 : f64, %p1 : i64)
      quregs(%r : !qref.reg<?>) indices(%idx0 : tensor<2xi64>, %idx1 : tensor<1xi64>)
      param_map = {p0 = [0], p1 = [1]}
      qubit_map = {qi0 = [0], qi1 = [1]}
    return
}

// -----

func.func @operator_register_multi_index_entry(%r : !qref.reg<3>, %idx0 : tensor<2xi64>, %idx1 : tensor<1xi64>) {
    qref.operator "custom_multi_index_entry"() quregs(%r : !qref.reg<3>)
      indices(%idx0 : tensor<2xi64>, %idx1 : tensor<1xi64>)
      qubit_map = {wires = [0, 1]}
    return
}

// -----

func.func @operator_custom_multi_param_entry(%p0 : f64, %p1 : f64, %q0 : !qref.bit, %q1 : !qref.bit) {
    qref.operator "custom_multi_param_entry"(%p0 : f64, %p1 : f64) qubits(%q0, %q1)
      param_map = {angles = [0, 1]}
      qubit_map = {pair = [0, 1]}
    return
}

// -----

func.func @operator_basic_with_static_data(%p : f64, %q : !qref.bit) {
    "qref.operator"(%p, %q) <{op_name = "with_static_data", static_data = {pauli_string = "XYZ", conditioning = 1 : i64}, adjoint = unit, operandSegmentSizes = array<i32: 1, 0, 1, 0, 0, 0, 0, 0, 0>}> : (f64, !qref.bit) -> ()
    return
}

// -----

func.func @operator_custom_basic_with_static_data(%p : f64, %q : !qref.bit) {
    qref.operator "custom_with_static_data"(%p : f64) adj qubits(%q)
      static_data = {pauli_string = "XYZ", conditioning = 1 : i64}
    return
}

// -----

func.func @operator_custom_with_uid_and_forward(%fwd : i64, %q0 : !qref.bit, %q1 : !qref.bit) {
    qref.operator "custom_uid_forward"() qubits(%q0, %q1)
      UID(7) forward(%fwd : i64)
    return
}

// -----

func.func @operator_no_mode() {
    // CHECK: qref.operator "no_qubits_or_qreg"() qubits()
    "qref.operator"() <{op_name = "no_qubits_or_qreg", operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0>}> : () -> ()
    return
}

// -----

func.func @operator_invalid_param_map_coverage(%p0 : f64, %p1 : i64, %q0 : !qref.bit, %q1 : !qref.bit) {
    // expected-error@+1 {{param_map must cover all params when provided: expected 2, got 1}}
    "qref.operator"(%p0, %p1, %q0, %q1) <{op_name = "bad_param_map", param_map = {p0 = array<i64: 0>}, qubit_map = {}, operandSegmentSizes = array<i32: 2, 0, 2, 0, 0, 0, 0, 0, 0>}> : (f64, i64, !qref.bit, !qref.bit) -> ()
    return
}

// -----

func.func @operator_custom_invalid_param_map_coverage(%p0 : f64, %p1 : i64, %q0 : !qref.bit, %q1 : !qref.bit) {
    // expected-error@+1 {{param_map must cover all params when provided: expected 2, got 1}}
    qref.operator "custom_bad_param_map"(%p0 : f64, %p1 : i64) qubits(%q0, %q1)
      param_map = {p0 = [0]}
    return
}

// -----

func.func @operator_invalid_qubit_map_coverage(%r : !qref.reg<3>, %idx0 : tensor<2xi64>, %idx1 : tensor<1xi64>) {
    // expected-error@+1 {{qubit_map must cover all index arrays in register mode: expected 2, got 1}}
    "qref.operator"(%r, %idx0, %idx1) <{op_name = "bad_qubit_map", param_map = {}, qubit_map = {qi0 = array<i64: 0>}, operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 1, 2, 0, 0>}> : (!qref.reg<3>, tensor<2xi64>, tensor<1xi64>) -> ()
    return
}

// -----

func.func @operator_custom_invalid_qubit_map_coverage(%r : !qref.reg<?>, %idx0 : tensor<2xi64>, %idx1 : tensor<1xi64>) {
    // expected-error@+1 {{qubit_map must cover all index arrays in register mode: expected 2, got 1}}
    qref.operator "custom_bad_qubit_map"()
      quregs(%r : !qref.reg<?>) indices(%idx0 : tensor<2xi64>, %idx1 : tensor<1xi64>)
      qubit_map = {qi0 = [0]}
    return
}

// -----

func.func @operator_invalid_qubit_map_sum(%q0 : !qref.bit, %q1 : !qref.bit) {
    // expected-error@+1 {{qubit_map must cover all qubit values in qubit mode: expected 2, got 1}}
    "qref.operator"(%q0, %q1) <{op_name = "bad_qubit_map_sum", param_map = {}, qubit_map = {pair = array<i64: 0>}, operandSegmentSizes = array<i32: 0, 0, 2, 0, 0, 0, 0, 0, 0>}> : (!qref.bit, !qref.bit) -> ()
    return
}

// -----

func.func @operator_custom_invalid_qubit_map_sum(%q0 : !qref.bit, %q1 : !qref.bit) {
    // expected-error@+1 {{qubit_map must cover all qubit values in qubit mode: expected 2, got 1}}
    qref.operator "custom_bad_qubit_map_sum"() qubits(%q0, %q1)
      qubit_map = {pair = [0]}
    return
}

// -----

func.func @operator_invalid_qubit_map_union(%q0 : !qref.bit, %q1 : !qref.bit) {
    // expected-error@+1 {{qubit_map must cover all qubit values in qubit mode: expected 2, got 1}}
    "qref.operator"(%q0, %q1) <{op_name = "bad_qubit_map_union", param_map = {}, qubit_map = {a = array<i64: 0>, b = array<i64: 0>}, operandSegmentSizes = array<i32: 0, 0, 2, 0, 0, 0, 0, 0, 0>}> : (!qref.bit, !qref.bit) -> ()
    return
}

// -----

func.func @operator_custom_invalid_qubit_map_union(%q0 : !qref.bit, %q1 : !qref.bit) {
    // expected-error@+1 {{qubit_map must cover all qubit values in qubit mode: expected 2, got 1}}
    qref.operator "custom_bad_qubit_map_union"() qubits(%q0, %q1)
      qubit_map = {a = [0], b = [0]}
    return
}

// -----

func.func @operator_invalid_register_qubit_map_oob(%r : !qref.reg<3>, %idx0 : tensor<2xi64>, %idx1 : tensor<1xi64>) {
    // expected-error@+1 {{qubit_map index is out of bounds with respect to index arrays: 2 is not in [0, 2)}}
    "qref.operator"(%r, %idx0, %idx1) <{op_name = "bad_register_map_oob", param_map = {}, qubit_map = {qi0 = array<i64: 0>, qi1 = array<i64: 2>}, operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 1, 2, 0, 0>}> : (!qref.reg<3>, tensor<2xi64>, tensor<1xi64>) -> ()
    return
}

// -----

func.func @operator_custom_invalid_register_qubit_map_oob(%r : !qref.reg<?>, %idx0 : tensor<2xi64>, %idx1 : tensor<1xi64>) {
    // expected-error@+1 {{qubit_map index is out of bounds with respect to index arrays: 2 is not in [0, 2)}}
    qref.operator "custom_bad_register_map_oob"()
      quregs(%r : !qref.reg<?>) indices(%idx0 : tensor<2xi64>, %idx1 : tensor<1xi64>)
      qubit_map = {qi0 = [0], qi1 = [2]}
    return
}

// -----

func.func @operator_invalid_forward_args_without_uid(%fwd : i64, %q0 : !qref.bit, %q1 : !qref.bit) {
    // expected-error@+1 {{forward_args can only be present when UID is provided}}
    "qref.operator"(%fwd, %q0, %q1) <{op_name = "bad_forward_no_uid", operandSegmentSizes = array<i32: 0, 1, 2, 0, 0, 0, 0, 0, 0>}> : (i64, !qref.bit, !qref.bit) -> ()
    return
}

// -----

func.func @operator_invalid_both_modes(%q : !qref.bit, %r : !qref.reg<2>, %idx : tensor<2xi64>) {
    // expected-error@+1 {{must use either qubits or registers, but not both}}
    "qref.operator"(%q, %r, %idx) <{op_name = "bad_both", operandSegmentSizes = array<i32: 0, 0, 1, 0, 0, 1, 1, 0, 0>}> : (!qref.bit, !qref.reg<2>, tensor<2xi64>) -> ()
    return
}

// -----

func.func @operator_invalid_ctrl_qubit_value_mismatch(%q : !qref.bit, %cq0 : !qref.bit, %cq1 : !qref.bit, %cv : i1) {
    // expected-error@+1 {{number of controlling qubits in input (2) and controlling values (1) must be the same}}
    "qref.operator"(%q, %cq0, %cq1, %cv) <{op_name = "bad_ctrl_qval", operandSegmentSizes = array<i32: 0, 0, 1, 2, 1, 0, 0, 0, 0>}> : (!qref.bit, !qref.bit, !qref.bit, i1) -> ()
    return
}

// -----

func.func @operator_invalid_control_mode_mix(%cq : !qref.bit, %ctrlv : i1, %r : !qref.reg<4>, %idx : tensor<2xi64>, %cidx : tensor<2xi64>, %cval : tensor<2xi1>) {
    // expected-error@+1 {{cannot mix qubit controls (ctrl_qubits/ctrl_values) with register controls (arr_ctrl_indices/arr_ctrl_values)}}
    "qref.operator"(%cq, %ctrlv, %r, %idx, %cidx, %cval) <{op_name = "bad_ctrl_mix", operandSegmentSizes = array<i32: 0, 0, 0, 1, 1, 1, 1, 1, 1>}> : (!qref.bit, i1, !qref.reg<4>, tensor<2xi64>, tensor<2xi64>, tensor<2xi1>) -> ()
    return
}

// -----

func.func @operator_invalid_ctrl_pair_presence(%r : !qref.reg<2>, %idx : tensor<2xi64>, %cidx : tensor<2xi64>) {
    // expected-error@+1 {{arr_ctrl_indices and arr_ctrl_values must either both be present or both absent}}
    "qref.operator"(%r, %idx, %cidx) <{op_name = "bad_ctrl_pair", operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 1, 1, 1, 0>}> : (!qref.reg<2>, tensor<2xi64>, tensor<2xi64>) -> ()
    return
}

// -----

func.func @operator_invalid_ctrl_static_length(%r : !qref.reg<2>, %idx : tensor<2xi64>, %cidx : tensor<2xi64>, %cval : tensor<1xi1>) {
    // expected-error@+1 {{number of input control qubits (2) and control values (1) must be the same}}
    "qref.operator"(%r, %idx, %cidx, %cval) <{op_name = "bad_ctrl_len", operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 1, 1, 1, 1>}> : (!qref.reg<2>, tensor<2xi64>, tensor<2xi64>, tensor<1xi1>) -> ()
    return
}

// -----

func.func @operator_custom_invalid_ctrl_static_length(%r : !qref.reg<?>, %idx : tensor<2xi64>, %cidx : tensor<2xi64>, %cval : tensor<1xi1>) {
    // expected-error@+1 {{number of input control qubits (2) and control values (1) must be the same}}
    qref.operator "custom_bad_ctrl_len"() quregs(%r : !qref.reg<?>) indices(%idx : tensor<2xi64>)
      ctrls(%cidx : tensor<2xi64>) ctrl_vals(%cval : tensor<1xi1>)
    return
}

// -----
