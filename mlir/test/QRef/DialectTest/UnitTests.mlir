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
    %0 = qref.alloc(2) : !qref.reg<2>

    // Dynamic
    %1 = qref.alloc(%arg0) : !qref.reg<?>

    return
}

// -----

func.func @test_dealloc(%arg0 : !qref.reg<10>) {
    qref.dealloc %arg0 : !qref.reg<10>
    return
}

// -----

func.func @test_get(%arg0 : !qref.reg<10>, %arg1: i64) {

    // Static
    %0 = qref.get %arg0[3] : !qref.reg<10> -> !qref.bit

    // Dynamic
    %1 = qref.get %arg0[%arg1] : !qref.reg<10>, i64 -> !qref.bit

    return
}

// -----

func.func @test_set_state(%arg0 : tensor<2xcomplex<f64>>, %q0: !qref.bit) {
    qref.set_state(%arg0) %q0 : tensor<2xcomplex<f64>>, !qref.bit
    return
}

// -----

func.func @test_basis_state(%arg0 : tensor<1xi1>, %q0: !qref.bit) {
    qref.set_basis_state(%arg0) %q0 : tensor<1xi1>, !qref.bit
    return
}

// -----

func.func @test_custom_op(%q0: !qref.bit, %q1: !qref.bit,
    %q2: !qref.bit, %q3: !qref.bit, %param0: f64, %param1: f64) {

    // Basic
    qref.custom "Hadamard"() %q0 : !qref.bit
    qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

    // With params
    qref.custom "RX"(%param0) %q0 : !qref.bit
    qref.custom "Rot"(%param0, %param1, %param1) %q0 : !qref.bit

    // With adjoint
    qref.custom "PauliX"() %q0 adj : !qref.bit
    qref.custom "CNOT"() %q0, %q1 adj : !qref.bit, !qref.bit

    // With control
    %true = llvm.mlir.constant (1 : i1) :i1
    %false = llvm.mlir.constant (0 : i1) :i1
    qref.custom "PauliZ"() %q0 ctrls (%q1) ctrlvals (%true) : !qref.bit ctrls !qref.bit
    qref.custom "RY"(%param0) %q0 ctrls (%q1) ctrlvals (%true) : !qref.bit ctrls !qref.bit
    qref.custom "SWAP"() %q0, %q1 ctrls (%q2, %q3) ctrlvals (%true, %false) : !qref.bit, !qref.bit ctrls !qref.bit, !qref.bit

    // With params, control and adjoint altogether
    qref.custom "Rot"(%param0, %param1, %param1) %q0, %q1 adj ctrls (%q2, %q3) ctrlvals (%true, %false) : !qref.bit, !qref.bit ctrls !qref.bit, !qref.bit

    return
}


// -----

func.func @test_paulirot_op(%q0: !qref.bit, %q1: !qref.bit,
    %q2: !qref.bit, %q3: !qref.bit, %angle: f64) {

    // Basic
    qref.paulirot ["Z"](%angle) %q0 : !qref.bit
    qref.paulirot ["Z", "X"](%angle) %q0, %q1 : !qref.bit, !qref.bit

    // With adjoint
    qref.paulirot ["Z", "X", "I"](%angle) %q0, %q1, %q2 adj : !qref.bit, !qref.bit, !qref.bit

    // With control
    %true = llvm.mlir.constant (1 : i1) :i1
    %false = llvm.mlir.constant (0 : i1) :i1
    qref.paulirot ["Y", "I"](%angle) %q0, %q1 ctrls (%q2, %q3) ctrlvals (%true, %false) : !qref.bit, !qref.bit ctrls !qref.bit, !qref.bit

    // With params, control and adjoint altogether
    qref.paulirot ["I", "X"](%angle) %q0, %q1 adj ctrls (%q2, %q3) ctrlvals (%true, %false) : !qref.bit, !qref.bit ctrls !qref.bit, !qref.bit

    return
}

// -----

func.func @test_global_phase(%q0: !qref.bit, %cv: i1, %param: f64) {

    // Basic
    qref.gphase(%param) : f64

    // With adjoint
    qref.gphase(%param) adj : f64

    // With control
    qref.gphase(%param) ctrls (%q0) ctrlvals (%cv) : f64 ctrls !qref.bit

    // With control and adjoint
    qref.gphase(%param) adj ctrls (%q0) ctrlvals (%cv) : f64 ctrls !qref.bit

    return
}

// -----

func.func @test_multirz(%q0: !qref.bit, %q1: !qref.bit,
    %q2: !qref.bit, %q3: !qref.bit, %theta: f64) {

    // Basic
    qref.multirz (%theta) %q0 : !qref.bit
    qref.multirz (%theta) %q0, %q1 : !qref.bit, !qref.bit

    // With adjoint
    qref.multirz (%theta) %q0, %q1, %q2 adj : !qref.bit, !qref.bit, !qref.bit

    // With control
    %true = llvm.mlir.constant (1 : i1) :i1
    %false = llvm.mlir.constant (0 : i1) :i1
    qref.multirz (%theta) %q0 ctrls (%q1) ctrlvals (%true) : !qref.bit ctrls !qref.bit
    qref.multirz (%theta) %q0, %q1 ctrls (%q2, %q3) ctrlvals (%true, %false) : !qref.bit, !qref.bit ctrls !qref.bit, !qref.bit

    // With control and adjoint
    qref.multirz (%theta) %q0, %q1 adj ctrls (%q2, %q3) ctrlvals (%true, %false) : !qref.bit, !qref.bit ctrls !qref.bit, !qref.bit

    return
}

// -----

func.func @test_pcphase(%q0: !qref.bit, %q1: !qref.bit,
    %q2: !qref.bit, %q3: !qref.bit, %theta: f64, %dim: f64) {

    // Basic
    qref.pcphase (%theta, %dim) %q0 : !qref.bit
    qref.pcphase (%theta, %dim) %q0, %q1, %q2 : !qref.bit, !qref.bit, !qref.bit

    // With adjoint
    qref.pcphase (%theta, %dim) %q0, %q1 adj : !qref.bit, !qref.bit

    // With control
    %true = llvm.mlir.constant (1 : i1) :i1
    %false = llvm.mlir.constant (0 : i1) :i1
    qref.pcphase (%theta, %dim) %q0 ctrls (%q1) ctrlvals (%true) : !qref.bit ctrls !qref.bit
    qref.pcphase (%theta, %dim) %q0, %q1 ctrls (%q2, %q3) ctrlvals (%true, %false) : !qref.bit, !qref.bit ctrls !qref.bit, !qref.bit

    // With control and adjoint
    qref.pcphase (%theta, %dim) %q0, %q1 adj ctrls (%q2, %q3) ctrlvals (%true, %false) : !qref.bit, !qref.bit ctrls !qref.bit, !qref.bit

    return
}

// -----

func.func @test_qubit_unitary(%q0: !qref.bit, %q1: !qref.bit,
    %q2: !qref.bit, %q3: !qref.bit) {

    // Basic
    %matrix22 = tensor.empty() : tensor<2x2xcomplex<f64>>
    %matrix44 = tensor.empty() : tensor<4x4xcomplex<f64>>

    qref.unitary (%matrix22 : tensor<2x2xcomplex<f64>>) %q0 : !qref.bit
    qref.unitary (%matrix44 : tensor<4x4xcomplex<f64>>) %q0, %q1 : !qref.bit, !qref.bit

    // With adjoint
    qref.unitary (%matrix22 : tensor<2x2xcomplex<f64>>) %q0 adj : !qref.bit

    // With control
    %true = llvm.mlir.constant (1 : i1) :i1
    %false = llvm.mlir.constant (0 : i1) :i1
    qref.unitary (%matrix22 : tensor<2x2xcomplex<f64>>) %q0 ctrls (%q1) ctrlvals (%true) : !qref.bit ctrls !qref.bit
    qref.unitary (%matrix44 : tensor<4x4xcomplex<f64>>) %q0, %q1 ctrls (%q2, %q3) ctrlvals (%true, %false) : !qref.bit, !qref.bit ctrls !qref.bit, !qref.bit

    // With control and adjoint
    qref.unitary (%matrix44 : tensor<4x4xcomplex<f64>>) %q0, %q1 adj ctrls (%q2, %q3) ctrlvals (%true, %false) : !qref.bit, !qref.bit ctrls !qref.bit, !qref.bit

    return
}

// -----

func.func @test_computational_basis_op(%q0: !qref.bit, %q1: !qref.bit, %r: !qref.reg<5>)
{
    %obs_q = qref.compbasis qubits %q0, %q1 : !quantum.obs
    %obs_r = qref.compbasis (qreg %r : !qref.reg<5>) : !quantum.obs
    func.return
}

// -----

func.func @test_namedobs_op(%q0: !qref.bit) {

    %ox = qref.namedobs %q0 [ PauliX] : !quantum.obs
    %oy = qref.namedobs %q0 [ PauliY] : !quantum.obs
    %oz = qref.namedobs %q0 [ PauliZ] : !quantum.obs
    %oi = qref.namedobs %q0 [ Identity] : !quantum.obs
    %oh = qref.namedobs %q0 [ Hadamard] : !quantum.obs

    return
}

// -----

func.func @test_hermitian_op(%q0: !qref.bit, %matrix: tensor<2x2xcomplex<f64>>) {
    %obs = qref.hermitian(%matrix : tensor<2x2xcomplex<f64>>) %q0 : !quantum.obs
    func.return
}

// -----

func.func @test_measure_op(%q0: !qref.bit) {
    %mres = qref.measure %q0 : i1
    func.return
}

// -----

func.func @test_expval_circuit() -> f64 {
    %a = qref.alloc(2) : !qref.reg<2>
    %q0 = qref.get %a[0] : !qref.reg<2> -> !qref.bit
    %q1 = qref.get %a[1] : !qref.reg<2> -> !qref.bit
    qref.custom "Hadamard"() %q0 : !qref.bit
    qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
    qref.custom "Hadamard"() %q0 : !qref.bit
    %obs = qref.namedobs %q1 [ PauliX] : !quantum.obs
    %expval = quantum.expval %obs : f64
    qref.dealloc %a : !qref.reg<2>
    return %expval : f64
}

// -----

func.func @test_circuit_with_loop(%nqubits: i64) -> f64 {
    %a = qref.alloc(%nqubits) : !qref.reg<?>

    %start = arith.constant 0 : index
    %step = arith.constant 1 : index
    %stop = index.casts %nqubits : i64 to index

    scf.for %i = %start to %stop step %step {
        %int = index.casts %i : index to i64
        %this_q = qref.get %a[%int] : !qref.reg<?>, i64 -> !qref.bit
        qref.custom "Hadamard"() %this_q : !qref.bit
        scf.yield
    }

    %q0 = qref.get %a[0] : !qref.reg<?> -> !qref.bit
    %obs = qref.namedobs %q0 [ PauliX] : !quantum.obs
    %expval = quantum.expval %obs : f64
    qref.dealloc %a : !qref.reg<?>
    return %expval : f64
}
