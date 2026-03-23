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

// RUN: quantum-opt --split-input-file --verify-diagnostics %s

func.func @test_extract_no_idx(%r : !qecl.hyperreg<3 x 1>) {
    // expected-error@below {{expected to have a non-null index}}
    %b = "qecl.extract_block"(%r) <{}> : (!qecl.hyperreg<3 x 1>) -> !qecl.codeblock<1>
    return
}

// -----

func.func @test_insert_no_idx(%r : !qecl.hyperreg<3 x 1>, %b : !qecl.codeblock<1>) {
    // expected-error@below {{expected to have a non-null index}}
    %r1 = "qecl.insert_block"(%r, %b) <{}> : (!qecl.hyperreg<3 x 1>, !qecl.codeblock<1>) -> !qecl.hyperreg<3 x 1>
    return
}

// -----

func.func @test_extract_negative_index(%r : !qecl.hyperreg<3 x 1>) {
    // expected-error@below {{attribute 'idx_attr' failed to satisfy constraint}}
    %b = qecl.extract_block %r[-1] : !qecl.hyperreg<3 x 1> -> !qecl.codeblock<1>
    return
}

// -----

func.func @test_insert_negative_index(%r : !qecl.hyperreg<3 x 1>, %b : !qecl.codeblock<1>) {
    // expected-error@below {{attribute 'idx_attr' failed to satisfy constraint}}
    %r1 = qecl.insert_block %r[-1], %b : !qecl.hyperreg<3 x 1>, !qecl.codeblock<1>
    return
}

// -----

func.func @test_extract_index_out_of_bounds(%r : !qecl.hyperreg<3 x 1>) {
    // expected-error@below {{out-of-bounds index attribute}}
    %b = qecl.extract_block %r[3] : !qecl.hyperreg<3 x 1> -> !qecl.codeblock<1>
    return
}

// -----

func.func @test_insert_index_out_of_bounds(%r : !qecl.hyperreg<3 x 1>, %b : !qecl.codeblock<1>) {
    // expected-error@below {{out-of-bounds index attribute}}
    %r1 = qecl.insert_block %r[3], %b : !qecl.hyperreg<3 x 1>, !qecl.codeblock<1>
    return
}

// -----

func.func @test_extract_type_mismatch(%r : !qecl.hyperreg<3 x 1>) {
    // expected-error@below {{expected hyper-register and codeblock types to have same value of k}}
    %b1 = qecl.extract_block %r[0] : !qecl.hyperreg<3 x 1> -> !qecl.codeblock<2>
    return
}

// -----

func.func @test_insert_type_mismatch(%r : !qecl.hyperreg<3 x 1>, %b : !qecl.codeblock<2>) {
    // expected-error@below {{expected hyper-register and codeblock types to have same value of k}}
    %b1 = qecl.insert_block %r[0], %b : !qecl.hyperreg<3 x 1>, !qecl.codeblock<2>
    return
}

// -----

func.func @test_gate_op_index_out_of_bounds_identity(%b : !qecl.codeblock<1>) {
    // expected-error@below {{out-of-bounds index attribute}}
    %b1 = qecl.identity %b[1] : !qecl.codeblock<1>
    return
}

// -----

func.func @test_gate_op_index_out_of_bounds_pauli_x(%b : !qecl.codeblock<1>) {
    // expected-error@below {{out-of-bounds index attribute}}
    %b1 = qecl.x %b[1] : !qecl.codeblock<1>
    return
}

// -----

func.func @test_gate_op_index_out_of_bounds_pauli_y(%b : !qecl.codeblock<1>) {
    // expected-error@below {{out-of-bounds index attribute}}
    %b1 = qecl.y %b[1] : !qecl.codeblock<1>
    return
}

// -----

func.func @test_gate_op_index_out_of_bounds_pauli_z(%b : !qecl.codeblock<1>) {
    // expected-error@below {{out-of-bounds index attribute}}
    %b1 = qecl.z %b[1] : !qecl.codeblock<1>
    return
}

// -----

func.func @test_gate_op_index_out_of_bounds_hadamard(%b : !qecl.codeblock<1>) {
    // expected-error@below {{out-of-bounds index attribute}}
    %b1 = qecl.hadamard %b[1] : !qecl.codeblock<1>
    return
}

// -----

func.func @test_gate_op_index_out_of_bounds_s(%b : !qecl.codeblock<1>) {
    // expected-error@below {{out-of-bounds index attribute}}
    %b1 = qecl.s %b[1] : !qecl.codeblock<1>
    return
}

// -----

func.func @test_gate_op_index_out_of_bounds_cnot_ctrl(%b0 : !qecl.codeblock<1>, %b1 : !qecl.codeblock<1>) {
    // expected-error@below {{out-of-bounds index attribute}}
    %b2, %b3 = qecl.cnot %b0[1], %b1[0] : !qecl.codeblock<1>, !qecl.codeblock<1>
    return
}

// -----

func.func @test_gate_op_index_out_of_bounds_cnot_trgt(%b0 : !qecl.codeblock<1>, %b1 : !qecl.codeblock<1>) {
    // expected-error@below {{out-of-bounds index attribute}}
    %b2, %b3 = qecl.cnot %b0[0], %b1[1] : !qecl.codeblock<1>, !qecl.codeblock<1>
    return
}

// -----

func.func @test_measure_op_index_out_of_bounds(%b : !qecl.codeblock<1>) {
    // expected-error@below {{out-of-bounds index attribute}}
    %mres, %b1 = qecl.measure %b[1] : i1, !qecl.codeblock<1>
    return
}
