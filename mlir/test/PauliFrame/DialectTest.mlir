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

// RUN: quantum-opt --split-input-file --verify-diagnostics %s

func.func @test_pauli_frame_init_single_qubit(%q0 : !quantum.bit) {
    %q1 = pauli_frame.init %q0 : !quantum.bit
    func.return
}

// -----

func.func @test_pauli_frame_init_multi_qubit(%q00 : !quantum.bit, %q10 : !quantum.bit) {
    %q01, %q11 = pauli_frame.init %q00, %q10 : !quantum.bit, !quantum.bit
    func.return
}

// -----

func.func @test_pauli_frame_init_qreg(%qreg : !quantum.reg) {
    %out_qreg = pauli_frame.init_qreg %qreg : !quantum.reg
    func.return
}

// -----

func.func @test_pauli_frame_read(%q0 : !quantum.bit) {
    %record_x, %record_z, %out_qubit = pauli_frame.read %q0 : i1, i1, !quantum.bit
    func.return
}

// -----

func.func @test_pauli_frame_update_single_qubit(%q0 : !quantum.bit) {
    %q1 = pauli_frame.update [0, 0] %q0 : !quantum.bit
    %q2 = pauli_frame.update [0, 1] %q1 : !quantum.bit
    %q3 = pauli_frame.update [1, 0] %q2 : !quantum.bit
    %q4 = pauli_frame.update [1, 1] %q3 : !quantum.bit
    func.return
}

// -----

func.func @test_pauli_frame_update_multi_qubit(%q00 : !quantum.bit, %q10 : !quantum.bit) {
    %q01, %q11 = pauli_frame.update [0, 0] %q00, %q10 : !quantum.bit, !quantum.bit
    %q02, %q12 = pauli_frame.update [0, 1] %q01, %q11 : !quantum.bit, !quantum.bit
    %q03, %q13 = pauli_frame.update [1, 0] %q02, %q12 : !quantum.bit, !quantum.bit
    %q04, %q14 = pauli_frame.update [1, 1] %q03, %q13 : !quantum.bit, !quantum.bit
    func.return
}

// -----

func.func @test_pauli_frame_update_with_clifford(%q00 : !quantum.bit, %q10 : !quantum.bit) {
    %q01, %q11 = pauli_frame.update_with_clifford [CNOT] %q00, %q10 : !quantum.bit, !quantum.bit
    %q02 = pauli_frame.update_with_clifford [Hadamard] %q01 : !quantum.bit
    %q12 = pauli_frame.update_with_clifford [S] %q11 : !quantum.bit
    func.return
}

// -----

func.func @test_pauli_frame_correct_measurement(%mres : i1, %q0 : !quantum.bit) {
    %out_mres, %q1 = pauli_frame.correct_measurement %mres, %q0 : i1, !quantum.bit
    func.return
}

// -----

func.func @test_pauli_frame_flush(%q0 : !quantum.bit) {
    %x, %z, %q1 = pauli_frame.flush %q0 : i1, i1, !quantum.bit
    func.return
}

// -----

func.func @test_pauli_frame_set_single_qubit(%q0 : !quantum.bit) {
    %q1 = pauli_frame.set [0, 0] %q0 : !quantum.bit
    %q2 = pauli_frame.set [0, 1] %q1 : !quantum.bit
    %q3 = pauli_frame.set [1, 0] %q2 : !quantum.bit
    %q4 = pauli_frame.set [1, 1] %q3 : !quantum.bit
    func.return
}

// -----

func.func @test_pauli_frame_set_multi_qubit(%q00 : !quantum.bit, %q10 : !quantum.bit) {
    %q01, %q11 = pauli_frame.set [0, 0] %q00, %q10 : !quantum.bit, !quantum.bit
    %q02, %q12 = pauli_frame.set [0, 1] %q01, %q11 : !quantum.bit, !quantum.bit
    %q03, %q13 = pauli_frame.set [1, 0] %q02, %q12 : !quantum.bit, !quantum.bit
    %q04, %q14 = pauli_frame.set [1, 1] %q03, %q13 : !quantum.bit, !quantum.bit
    func.return
}
