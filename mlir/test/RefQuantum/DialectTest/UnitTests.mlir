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

// Test basic parsing.
//
// RUN: quantum-opt --split-input-file --verify-diagnostics %s


func.func @test_alloc(%arg0 : i64) {

    // Static
    %0 = ref_quantum.alloc(2) : !ref_quantum.allocation

    // Dynamic
    %1 = ref_quantum.alloc(%arg0) : !ref_quantum.allocation

    return
}

// -----

func.func @test_dealloc(%arg0 : !ref_quantum.allocation) {
    ref_quantum.dealloc %arg0 : !ref_quantum.allocation
    return
}

// -----

func.func @test_set_state(%arg0 : tensor<2xcomplex<f64>>, %q0: !ref_quantum.qubit_ref) {
    ref_quantum.set_state(%arg0) %q0 : tensor<2xcomplex<f64>>, !ref_quantum.qubit_ref
    return
}

// -----

func.func @test_basis_state(%arg0 : tensor<1xi1>, %q0: !ref_quantum.qubit_ref) {
    ref_quantum.set_basis_state(%arg0) %q0 : tensor<1xi1>, !ref_quantum.qubit_ref
    return
}

// -----

func.func @test_custom_op(%q0: !ref_quantum.qubit_ref, %q1: !ref_quantum.qubit_ref,
    %q2: !ref_quantum.qubit_ref, %q3: !ref_quantum.qubit_ref, %param0: f64, %param1: f64) {

    // Basic
    ref_quantum.custom "Hadamard"() %q0 : !ref_quantum.qubit_ref
    ref_quantum.custom "CNOT"() %q0, %q1 : !ref_quantum.qubit_ref, !ref_quantum.qubit_ref

    // With params
    ref_quantum.custom "RX"(%param0) %q0 : !ref_quantum.qubit_ref
    ref_quantum.custom "Rot"(%param0, %param1, %param1) %q0 : !ref_quantum.qubit_ref

    // With adjoint
    ref_quantum.custom "PauliX"() %q0 adj : !ref_quantum.qubit_ref
    ref_quantum.custom "CNOT"() %q0, %q1 adj : !ref_quantum.qubit_ref, !ref_quantum.qubit_ref

    // With control
    %true = llvm.mlir.constant (1 : i1) :i1
    %false = llvm.mlir.constant (0 : i1) :i1
    ref_quantum.custom "PauliZ"() %q0 ctrls (%q1) ctrlvals (%true) : !ref_quantum.qubit_ref ctrls !ref_quantum.qubit_ref
    ref_quantum.custom "RY"(%param0) %q0 ctrls (%q1) ctrlvals (%true) : !ref_quantum.qubit_ref ctrls !ref_quantum.qubit_ref
    ref_quantum.custom "SWAP"() %q0, %q1 ctrls (%q2, %q3) ctrlvals (%true, %false) : !ref_quantum.qubit_ref, !ref_quantum.qubit_ref ctrls !ref_quantum.qubit_ref, !ref_quantum.qubit_ref

    // With params, control and adjoint altogether
    ref_quantum.custom "Rot"(%param0, %param1, %param1) %q0, %q1 adj ctrls (%q2, %q3) ctrlvals (%true, %false) : !ref_quantum.qubit_ref, !ref_quantum.qubit_ref ctrls !ref_quantum.qubit_ref, !ref_quantum.qubit_ref

    return
}


// -----

func.func @test_paulirot_op(%q0: !ref_quantum.qubit_ref, %q1: !ref_quantum.qubit_ref,
    %q2: !ref_quantum.qubit_ref, %q3: !ref_quantum.qubit_ref, %angle: f64) {

    // Basic
    ref_quantum.paulirot ["Z"](%angle) %q0 : !ref_quantum.qubit_ref
    ref_quantum.paulirot ["Z", "X"](%angle) %q0, %q1 : !ref_quantum.qubit_ref, !ref_quantum.qubit_ref

    // With adjoint
    ref_quantum.paulirot ["Z", "X", "I"](%angle) %q0, %q1, %q2 adj : !ref_quantum.qubit_ref, !ref_quantum.qubit_ref, !ref_quantum.qubit_ref

    // With control
    %true = llvm.mlir.constant (1 : i1) :i1
    %false = llvm.mlir.constant (0 : i1) :i1
    ref_quantum.paulirot ["Y", "I"](%angle) %q0, %q1 ctrls (%q2, %q3) ctrlvals (%true, %false) : !ref_quantum.qubit_ref, !ref_quantum.qubit_ref ctrls !ref_quantum.qubit_ref, !ref_quantum.qubit_ref

    // With params, control and adjoint altogether
    ref_quantum.paulirot ["I", "X"](%angle) %q0, %q1 adj ctrls (%q2, %q3) ctrlvals (%true, %false) : !ref_quantum.qubit_ref, !ref_quantum.qubit_ref ctrls !ref_quantum.qubit_ref, !ref_quantum.qubit_ref

    return
}

// -----

func.func @test_global_phase(%q0: !ref_quantum.qubit_ref, %cv: i1, %param: f64) {

    // Basic
    ref_quantum.gphase(%param) : f64

    // With adjoint
    ref_quantum.gphase(%param) adj : f64

    // With control
    ref_quantum.gphase(%param) ctrls (%q0) ctrlvals (%cv) : f64 ctrls !ref_quantum.qubit_ref

    // With control and adjoint
    ref_quantum.gphase(%param) adj ctrls (%q0) ctrlvals (%cv) : f64 ctrls !ref_quantum.qubit_ref

    return
}

// -----

func.func @test_multirz(%q0: !ref_quantum.qubit_ref, %q1: !ref_quantum.qubit_ref,
    %q2: !ref_quantum.qubit_ref, %q3: !ref_quantum.qubit_ref, %theta: f64) {

    // Basic
    ref_quantum.multirz (%theta) %q0 : !ref_quantum.qubit_ref
    ref_quantum.multirz (%theta) %q0, %q1 : !ref_quantum.qubit_ref, !ref_quantum.qubit_ref

    // With adjoint
    ref_quantum.multirz (%theta) %q0, %q1, %q2 adj : !ref_quantum.qubit_ref, !ref_quantum.qubit_ref, !ref_quantum.qubit_ref

    // With control
    %true = llvm.mlir.constant (1 : i1) :i1
    %false = llvm.mlir.constant (0 : i1) :i1
    ref_quantum.multirz (%theta) %q0 ctrls (%q1) ctrlvals (%true) : !ref_quantum.qubit_ref ctrls !ref_quantum.qubit_ref
    ref_quantum.multirz (%theta) %q0, %q1 ctrls (%q2, %q3) ctrlvals (%true, %false) : !ref_quantum.qubit_ref, !ref_quantum.qubit_ref ctrls !ref_quantum.qubit_ref, !ref_quantum.qubit_ref

    // With control and adjoint
    ref_quantum.multirz (%theta) %q0, %q1 adj ctrls (%q2, %q3) ctrlvals (%true, %false) : !ref_quantum.qubit_ref, !ref_quantum.qubit_ref ctrls !ref_quantum.qubit_ref, !ref_quantum.qubit_ref

    return
}

// -----

func.func @test_pcphase(%q0: !ref_quantum.qubit_ref, %q1: !ref_quantum.qubit_ref,
    %q2: !ref_quantum.qubit_ref, %q3: !ref_quantum.qubit_ref, %theta: f64, %dim: f64) {

    // Basic
    ref_quantum.pcphase (%theta, %dim) %q0 : !ref_quantum.qubit_ref
    ref_quantum.pcphase (%theta, %dim) %q0, %q1, %q2 : !ref_quantum.qubit_ref, !ref_quantum.qubit_ref, !ref_quantum.qubit_ref

    // With adjoint
    ref_quantum.pcphase (%theta, %dim) %q0, %q1 adj : !ref_quantum.qubit_ref, !ref_quantum.qubit_ref

    // With control
    %true = llvm.mlir.constant (1 : i1) :i1
    %false = llvm.mlir.constant (0 : i1) :i1
    ref_quantum.pcphase (%theta, %dim) %q0 ctrls (%q1) ctrlvals (%true) : !ref_quantum.qubit_ref ctrls !ref_quantum.qubit_ref
    ref_quantum.pcphase (%theta, %dim) %q0, %q1 ctrls (%q2, %q3) ctrlvals (%true, %false) : !ref_quantum.qubit_ref, !ref_quantum.qubit_ref ctrls !ref_quantum.qubit_ref, !ref_quantum.qubit_ref

    // With control and adjoint
    ref_quantum.pcphase (%theta, %dim) %q0, %q1 adj ctrls (%q2, %q3) ctrlvals (%true, %false) : !ref_quantum.qubit_ref, !ref_quantum.qubit_ref ctrls !ref_quantum.qubit_ref, !ref_quantum.qubit_ref

    return
}

// -----

func.func @test_qubit_unitary(%q0: !ref_quantum.qubit_ref, %q1: !ref_quantum.qubit_ref,
    %q2: !ref_quantum.qubit_ref, %q3: !ref_quantum.qubit_ref) {

    // Basic
    %matrix22 = tensor.empty() : tensor<2x2xcomplex<f64>>
    %matrix44 = tensor.empty() : tensor<4x4xcomplex<f64>>

    ref_quantum.unitary (%matrix22 : tensor<2x2xcomplex<f64>>) %q0 : !ref_quantum.qubit_ref
    ref_quantum.unitary (%matrix44 : tensor<4x4xcomplex<f64>>) %q0, %q1 : !ref_quantum.qubit_ref, !ref_quantum.qubit_ref

    // With adjoint
    ref_quantum.unitary (%matrix22 : tensor<2x2xcomplex<f64>>) %q0 adj : !ref_quantum.qubit_ref

    // With control
    %true = llvm.mlir.constant (1 : i1) :i1
    %false = llvm.mlir.constant (0 : i1) :i1
    ref_quantum.unitary (%matrix22 : tensor<2x2xcomplex<f64>>) %q0 ctrls (%q1) ctrlvals (%true) : !ref_quantum.qubit_ref ctrls !ref_quantum.qubit_ref
    ref_quantum.unitary (%matrix44 : tensor<4x4xcomplex<f64>>) %q0, %q1 ctrls (%q2, %q3) ctrlvals (%true, %false) : !ref_quantum.qubit_ref, !ref_quantum.qubit_ref ctrls !ref_quantum.qubit_ref, !ref_quantum.qubit_ref

    // With control and adjoint
    ref_quantum.unitary (%matrix44 : tensor<4x4xcomplex<f64>>) %q0, %q1 adj ctrls (%q2, %q3) ctrlvals (%true, %false) : !ref_quantum.qubit_ref, !ref_quantum.qubit_ref ctrls !ref_quantum.qubit_ref, !ref_quantum.qubit_ref

    return
}

// -----

func.func @test_namedobs_op(%q0: !ref_quantum.qubit_ref) {

    %ox = ref_quantum.namedobs %q0 [ PauliX] : !quantum.obs
    %oy = ref_quantum.namedobs %q0 [ PauliY] : !quantum.obs
    %oz = ref_quantum.namedobs %q0 [ PauliZ] : !quantum.obs
    %oi = ref_quantum.namedobs %q0 [ Identity] : !quantum.obs
    %oh = ref_quantum.namedobs %q0 [ Hadamard] : !quantum.obs

    return
}

// -----

func.func @test_expval_circuit(%q0: !ref_quantum.qubit_ref) -> f64 {
    ref_quantum.custom "Hadamard"() %q0 : !ref_quantum.qubit_ref
    %obs = ref_quantum.namedobs %q0 [ PauliX] : !quantum.obs
    %expval = quantum.expval %obs : f64
    return %expval : f64
}
