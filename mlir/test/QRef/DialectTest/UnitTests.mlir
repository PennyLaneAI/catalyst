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
    %0 = qref.alloc(2) : !qref.allocation

    // Dynamic
    %1 = qref.alloc(%arg0) : !qref.allocation

    return
}

// -----

func.func @test_dealloc(%arg0 : !qref.allocation) {
    qref.dealloc %arg0 : !qref.allocation
    return
}

// -----

func.func @test_make_reference(%arg0 : !qref.allocation, %arg1: i64) {

    // Static
    %0 = qref.make_reference %arg0, 3 : !qref.allocation -> !qref.qubit_ref

    // Dynamic
    %1 = qref.make_reference %arg0, %arg1 : !qref.allocation, i64 -> !qref.qubit_ref

    return
}

// -----

func.func @test_set_state(%arg0 : tensor<2xcomplex<f64>>, %q0: !qref.qubit_ref) {
    qref.set_state(%arg0) %q0 : tensor<2xcomplex<f64>>, !qref.qubit_ref
    return
}

// -----

func.func @test_basis_state(%arg0 : tensor<1xi1>, %q0: !qref.qubit_ref) {
    qref.set_basis_state(%arg0) %q0 : tensor<1xi1>, !qref.qubit_ref
    return
}

// -----

func.func @test_custom_op(%q0: !qref.qubit_ref, %q1: !qref.qubit_ref,
    %q2: !qref.qubit_ref, %q3: !qref.qubit_ref, %param0: f64, %param1: f64) {

    // Basic
    qref.custom "Hadamard"() %q0 : !qref.qubit_ref
    qref.custom "CNOT"() %q0, %q1 : !qref.qubit_ref, !qref.qubit_ref

    // With params
    qref.custom "RX"(%param0) %q0 : !qref.qubit_ref
    qref.custom "Rot"(%param0, %param1, %param1) %q0 : !qref.qubit_ref

    // With adjoint
    qref.custom "PauliX"() %q0 adj : !qref.qubit_ref
    qref.custom "CNOT"() %q0, %q1 adj : !qref.qubit_ref, !qref.qubit_ref

    // With control
    %true = llvm.mlir.constant (1 : i1) :i1
    %false = llvm.mlir.constant (0 : i1) :i1
    qref.custom "PauliZ"() %q0 ctrls (%q1) ctrlvals (%true) : !qref.qubit_ref ctrls !qref.qubit_ref
    qref.custom "RY"(%param0) %q0 ctrls (%q1) ctrlvals (%true) : !qref.qubit_ref ctrls !qref.qubit_ref
    qref.custom "SWAP"() %q0, %q1 ctrls (%q2, %q3) ctrlvals (%true, %false) : !qref.qubit_ref, !qref.qubit_ref ctrls !qref.qubit_ref, !qref.qubit_ref

    // With params, control and adjoint altogether
    qref.custom "Rot"(%param0, %param1, %param1) %q0, %q1 adj ctrls (%q2, %q3) ctrlvals (%true, %false) : !qref.qubit_ref, !qref.qubit_ref ctrls !qref.qubit_ref, !qref.qubit_ref

    return
}


// -----

func.func @test_paulirot_op(%q0: !qref.qubit_ref, %q1: !qref.qubit_ref,
    %q2: !qref.qubit_ref, %q3: !qref.qubit_ref, %angle: f64) {

    // Basic
    qref.paulirot ["Z"](%angle) %q0 : !qref.qubit_ref
    qref.paulirot ["Z", "X"](%angle) %q0, %q1 : !qref.qubit_ref, !qref.qubit_ref

    // With adjoint
    qref.paulirot ["Z", "X", "I"](%angle) %q0, %q1, %q2 adj : !qref.qubit_ref, !qref.qubit_ref, !qref.qubit_ref

    // With control
    %true = llvm.mlir.constant (1 : i1) :i1
    %false = llvm.mlir.constant (0 : i1) :i1
    qref.paulirot ["Y", "I"](%angle) %q0, %q1 ctrls (%q2, %q3) ctrlvals (%true, %false) : !qref.qubit_ref, !qref.qubit_ref ctrls !qref.qubit_ref, !qref.qubit_ref

    // With params, control and adjoint altogether
    qref.paulirot ["I", "X"](%angle) %q0, %q1 adj ctrls (%q2, %q3) ctrlvals (%true, %false) : !qref.qubit_ref, !qref.qubit_ref ctrls !qref.qubit_ref, !qref.qubit_ref

    return
}

// -----

func.func @test_global_phase(%q0: !qref.qubit_ref, %cv: i1, %param: f64) {

    // Basic
    qref.gphase(%param) : f64

    // With adjoint
    qref.gphase(%param) adj : f64

    // With control
    qref.gphase(%param) ctrls (%q0) ctrlvals (%cv) : f64 ctrls !qref.qubit_ref

    // With control and adjoint
    qref.gphase(%param) adj ctrls (%q0) ctrlvals (%cv) : f64 ctrls !qref.qubit_ref

    return
}

// -----

func.func @test_multirz(%q0: !qref.qubit_ref, %q1: !qref.qubit_ref,
    %q2: !qref.qubit_ref, %q3: !qref.qubit_ref, %theta: f64) {

    // Basic
    qref.multirz (%theta) %q0 : !qref.qubit_ref
    qref.multirz (%theta) %q0, %q1 : !qref.qubit_ref, !qref.qubit_ref

    // With adjoint
    qref.multirz (%theta) %q0, %q1, %q2 adj : !qref.qubit_ref, !qref.qubit_ref, !qref.qubit_ref

    // With control
    %true = llvm.mlir.constant (1 : i1) :i1
    %false = llvm.mlir.constant (0 : i1) :i1
    qref.multirz (%theta) %q0 ctrls (%q1) ctrlvals (%true) : !qref.qubit_ref ctrls !qref.qubit_ref
    qref.multirz (%theta) %q0, %q1 ctrls (%q2, %q3) ctrlvals (%true, %false) : !qref.qubit_ref, !qref.qubit_ref ctrls !qref.qubit_ref, !qref.qubit_ref

    // With control and adjoint
    qref.multirz (%theta) %q0, %q1 adj ctrls (%q2, %q3) ctrlvals (%true, %false) : !qref.qubit_ref, !qref.qubit_ref ctrls !qref.qubit_ref, !qref.qubit_ref

    return
}

// -----

func.func @test_pcphase(%q0: !qref.qubit_ref, %q1: !qref.qubit_ref,
    %q2: !qref.qubit_ref, %q3: !qref.qubit_ref, %theta: f64, %dim: f64) {

    // Basic
    qref.pcphase (%theta, %dim) %q0 : !qref.qubit_ref
    qref.pcphase (%theta, %dim) %q0, %q1, %q2 : !qref.qubit_ref, !qref.qubit_ref, !qref.qubit_ref

    // With adjoint
    qref.pcphase (%theta, %dim) %q0, %q1 adj : !qref.qubit_ref, !qref.qubit_ref

    // With control
    %true = llvm.mlir.constant (1 : i1) :i1
    %false = llvm.mlir.constant (0 : i1) :i1
    qref.pcphase (%theta, %dim) %q0 ctrls (%q1) ctrlvals (%true) : !qref.qubit_ref ctrls !qref.qubit_ref
    qref.pcphase (%theta, %dim) %q0, %q1 ctrls (%q2, %q3) ctrlvals (%true, %false) : !qref.qubit_ref, !qref.qubit_ref ctrls !qref.qubit_ref, !qref.qubit_ref

    // With control and adjoint
    qref.pcphase (%theta, %dim) %q0, %q1 adj ctrls (%q2, %q3) ctrlvals (%true, %false) : !qref.qubit_ref, !qref.qubit_ref ctrls !qref.qubit_ref, !qref.qubit_ref

    return
}

// -----

func.func @test_qubit_unitary(%q0: !qref.qubit_ref, %q1: !qref.qubit_ref,
    %q2: !qref.qubit_ref, %q3: !qref.qubit_ref) {

    // Basic
    %matrix22 = tensor.empty() : tensor<2x2xcomplex<f64>>
    %matrix44 = tensor.empty() : tensor<4x4xcomplex<f64>>

    qref.unitary (%matrix22 : tensor<2x2xcomplex<f64>>) %q0 : !qref.qubit_ref
    qref.unitary (%matrix44 : tensor<4x4xcomplex<f64>>) %q0, %q1 : !qref.qubit_ref, !qref.qubit_ref

    // With adjoint
    qref.unitary (%matrix22 : tensor<2x2xcomplex<f64>>) %q0 adj : !qref.qubit_ref

    // With control
    %true = llvm.mlir.constant (1 : i1) :i1
    %false = llvm.mlir.constant (0 : i1) :i1
    qref.unitary (%matrix22 : tensor<2x2xcomplex<f64>>) %q0 ctrls (%q1) ctrlvals (%true) : !qref.qubit_ref ctrls !qref.qubit_ref
    qref.unitary (%matrix44 : tensor<4x4xcomplex<f64>>) %q0, %q1 ctrls (%q2, %q3) ctrlvals (%true, %false) : !qref.qubit_ref, !qref.qubit_ref ctrls !qref.qubit_ref, !qref.qubit_ref

    // With control and adjoint
    qref.unitary (%matrix44 : tensor<4x4xcomplex<f64>>) %q0, %q1 adj ctrls (%q2, %q3) ctrlvals (%true, %false) : !qref.qubit_ref, !qref.qubit_ref ctrls !qref.qubit_ref, !qref.qubit_ref

    return
}

// -----

func.func @test_namedobs_op(%q0: !qref.qubit_ref) {

    %ox = qref.namedobs %q0 [ PauliX] : !quantum.obs
    %oy = qref.namedobs %q0 [ PauliY] : !quantum.obs
    %oz = qref.namedobs %q0 [ PauliZ] : !quantum.obs
    %oi = qref.namedobs %q0 [ Identity] : !quantum.obs
    %oh = qref.namedobs %q0 [ Hadamard] : !quantum.obs

    return
}

// -----

func.func @test_expval_circuit() -> f64 {
    %a = qref.alloc(2) : !qref.allocation
    %q0 = qref.make_reference %a, 0 : !qref.allocation -> !qref.qubit_ref
    %q1 = qref.make_reference %a, 1 : !qref.allocation -> !qref.qubit_ref
    qref.custom "Hadamard"() %q0 : !qref.qubit_ref
    qref.custom "CNOT"() %q0, %q1 : !qref.qubit_ref, !qref.qubit_ref
    qref.custom "Hadamard"() %q0 : !qref.qubit_ref
    %obs = qref.namedobs %q1 [ PauliX] : !quantum.obs
    %expval = quantum.expval %obs : f64
    qref.dealloc %a : !qref.allocation
    return %expval : f64
}

// -----

func.func @test_circuit_with_loop(%nqubits: i64) -> f64 {
    %a = qref.alloc(%nqubits) : !qref.allocation

    %start = arith.constant 0 : index
    %step = arith.constant 1 : index
    %stop = index.casts %nqubits : i64 to index

    scf.for %i = %start to %stop step %step {
        %int = index.casts %i : index to i64
        %this_q = qref.make_reference %a, %int : !qref.allocation, i64 -> !qref.qubit_ref
        qref.custom "Hadamard"() %this_q : !qref.qubit_ref
        scf.yield
    }

    %q0 = qref.make_reference %a, 0 : !qref.allocation -> !qref.qubit_ref
    %obs = qref.namedobs %q0 [ PauliX] : !quantum.obs
    %expval = quantum.expval %obs : f64
    qref.dealloc %a : !qref.allocation
    return %expval : f64
}
