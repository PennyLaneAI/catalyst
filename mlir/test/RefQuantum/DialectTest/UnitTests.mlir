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


func.func @test_set_state(%arg0 : tensor<2xcomplex<f64>>, %w0: i64) {
    ref_quantum.set_state(%arg0) %w0 : tensor<2xcomplex<f64>>, i64
    return
}

// -----

func.func @test_basis_state(%arg0 : tensor<1xi1>, %w0: i64) {
    ref_quantum.set_basis_state(%arg0) %w0 : tensor<1xi1>, i64
    return
}

// -----

func.func @test_custom_op(%w0: i64, %w1: i64, %w2: i64, %w3: i64, %param0: f64, %param1: f64) {

    // Basic
    ref_quantum.custom "Hadamard"() %w0 : i64
    ref_quantum.custom "CNOT"() %w0, %w1 : i64, i64

    // With params
    ref_quantum.custom "RX"(%param0) %w0 : i64
    ref_quantum.custom "Rot"(%param0, %param1, %param1) %w0 : i64

    // With adjoint
    ref_quantum.custom "PauliX"() %w0 adj : i64
    ref_quantum.custom "CNOT"() %w0, %w1 adj : i64, i64

    // With control
    %true = llvm.mlir.constant (1 : i1) :i1
    %false = llvm.mlir.constant (0 : i1) :i1
    ref_quantum.custom "PauliZ"() %w0 ctrls (%w1) ctrlvals (%true) : i64 ctrls i64
    ref_quantum.custom "RY"(%param0) %w0 ctrls (%w1) ctrlvals (%true) : i64 ctrls i64
    ref_quantum.custom "SWAP"() %w0, %w1 ctrls (%w2, %w3) ctrlvals (%true, %false) : i64, i64 ctrls i64, i64

    // With params, control and adjoint altogether
    ref_quantum.custom "Rot"(%param0, %param1, %param1) %w0, %w1 adj ctrls (%w2, %w3) ctrlvals (%true, %false) : i64, i64 ctrls i64, i64

    return
}


// -----

func.func @test_paulirot_op(%w0: i64, %w1: i64, %w2: i64, %w3: i64, %angle: f64) {

    // Basic
    ref_quantum.paulirot ["Z"](%angle) %w0 : i64
    ref_quantum.paulirot ["Z", "X"](%angle) %w0, %w1 : i64, i64

    // With adjoint
    ref_quantum.paulirot ["Z", "X", "I"](%angle) %w0, %w1, %w2 adj : i64, i64, i64

    // With control
    %true = llvm.mlir.constant (1 : i1) :i1
    %false = llvm.mlir.constant (0 : i1) :i1
    ref_quantum.paulirot ["Y", "I"](%angle) %w0, %w1 ctrls (%w2, %w3) ctrlvals (%true, %false) : i64, i64 ctrls i64, i64

    // With params, control and adjoint altogether
    ref_quantum.paulirot ["I", "X"](%angle) %w0, %w1 adj ctrls (%w2, %w3) ctrlvals (%true, %false) : i64, i64 ctrls i64, i64

    return
}

// -----

func.func @test_global_phase(%w0: i64, %cv: i1, %param: f64) {

    // Basic
    ref_quantum.gphase(%param) : f64

    // With adjoint
    ref_quantum.gphase(%param) adj : f64

    // With control
    ref_quantum.gphase(%param) ctrls (%w0) ctrlvals (%cv) : f64 ctrls i64

    // With control and adjoint
    ref_quantum.gphase(%param) adj ctrls (%w0) ctrlvals (%cv) : f64 ctrls i64

    return
}

// -----

func.func @test_multirz(%w0: i64, %w1: i64, %w2: i64, %w3: i64, %theta: f64) {

    // Basic
    ref_quantum.multirz (%theta) %w0 : i64
    ref_quantum.multirz (%theta) %w0, %w1 : i64, i64

    // With adjoint
    ref_quantum.multirz (%theta) %w0, %w1, %w2 adj : i64, i64, i64

    // With control
    %true = llvm.mlir.constant (1 : i1) :i1
    %false = llvm.mlir.constant (0 : i1) :i1
    ref_quantum.multirz (%theta) %w0 ctrls (%w1) ctrlvals (%true) : i64 ctrls i64
    ref_quantum.multirz (%theta) %w0, %w1 ctrls (%w2, %w3) ctrlvals (%true, %false) : i64, i64 ctrls i64, i64

    // With control and adjoint
    ref_quantum.multirz (%theta) %w0, %w1 adj ctrls (%w2, %w3) ctrlvals (%true, %false) : i64, i64 ctrls i64, i64

    return
}

// -----

func.func @test_pcphase(%w0: i64, %w1: i64, %w2: i64, %w3: i64, %theta: f64, %dim: f64) {

    // Basic
    ref_quantum.pcphase (%theta, %dim) %w0 : i64
    ref_quantum.pcphase (%theta, %dim) %w0, %w1, %w2 : i64, i64, i64

    // With adjoint
    ref_quantum.pcphase (%theta, %dim) %w0, %w1 adj : i64, i64

    // With control
    %true = llvm.mlir.constant (1 : i1) :i1
    %false = llvm.mlir.constant (0 : i1) :i1
    ref_quantum.pcphase (%theta, %dim) %w0 ctrls (%w1) ctrlvals (%true) : i64 ctrls i64
    ref_quantum.pcphase (%theta, %dim) %w0, %w1 ctrls (%w2, %w3) ctrlvals (%true, %false) : i64, i64 ctrls i64, i64

    // With control and adjoint
    ref_quantum.pcphase (%theta, %dim) %w0, %w1 adj ctrls (%w2, %w3) ctrlvals (%true, %false) : i64, i64 ctrls i64, i64

    return
}

// -----

func.func @test_qubit_unitary(%w0: i64, %w1: i64, %w2: i64, %w3: i64) {

    // Basic
    %matrix22 = tensor.empty() : tensor<2x2xcomplex<f64>>
    %matrix44 = tensor.empty() : tensor<4x4xcomplex<f64>>

    ref_quantum.unitary (%matrix22 : tensor<2x2xcomplex<f64>>) %w0 : i64
    ref_quantum.unitary (%matrix44 : tensor<4x4xcomplex<f64>>) %w0, %w1 : i64, i64

    // With adjoint
    ref_quantum.unitary (%matrix22 : tensor<2x2xcomplex<f64>>) %w0 adj : i64

    // With control
    %true = llvm.mlir.constant (1 : i1) :i1
    %false = llvm.mlir.constant (0 : i1) :i1
    ref_quantum.unitary (%matrix22 : tensor<2x2xcomplex<f64>>) %w0 ctrls (%w1) ctrlvals (%true) : i64 ctrls i64
    ref_quantum.unitary (%matrix44 : tensor<4x4xcomplex<f64>>) %w0, %w1 ctrls (%w2, %w3) ctrlvals (%true, %false) : i64, i64 ctrls i64, i64

    // With control and adjoint
    ref_quantum.unitary (%matrix44 : tensor<4x4xcomplex<f64>>) %w0, %w1 adj ctrls (%w2, %w3) ctrlvals (%true, %false) : i64, i64 ctrls i64, i64

    return
}

// -----

func.func @test_namedobs_op(%w0: i64) {

    %ox = ref_quantum.namedobs %w0 [ PauliX] : !quantum.obs
    %oy = ref_quantum.namedobs %w0 [ PauliY] : !quantum.obs
    %oz = ref_quantum.namedobs %w0 [ PauliZ] : !quantum.obs
    %oi = ref_quantum.namedobs %w0 [ Identity] : !quantum.obs
    %oh = ref_quantum.namedobs %w0 [ Hadamard] : !quantum.obs

    return
}

// -----

func.func @test_expval_circuit() -> f64 {
    %0 = arith.constant 0 : i64
    ref_quantum.custom "Hadamard"() %0 : i64
    %obs = ref_quantum.namedobs %0 [ PauliX] : !quantum.obs
    %expval = quantum.expval %obs : f64
    return %expval : f64
}
