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

func.func @test_qubit_data(%arg0 : !qecp.qubit<data>) {
    func.return
}

// -----

func.func @test_qubit_aux(%arg0 : !qecp.qubit<aux>) {
    func.return
}

// -----

func.func @test_codeblock(%arg0 : !qecp.codeblock<1 x 7>) {
    func.return
}

// -----

// TODO: Currently we must write parameters as '<3 x 1 x 7>' (with spaces)
func.func @test_hyperreg(%arg0 : !qecp.hyperreg<3 x 1 x 7>) {
    func.return
}

// -----

func.func @test_alloc() {
    %0 = qecp.alloc() : !qecp.hyperreg<3 x 1 x 7>
    func.return
}

// -----

func.func @test_dealloc(%arg0 : !qecp.hyperreg<3 x 1 x 7>) {
    qecp.dealloc %arg0 : !qecp.hyperreg<3 x 1 x 7>
    func.return
}

// -----

func.func @test_extract_block_static_idx(%arg0 : !qecp.hyperreg<3 x 1 x 7>) {
    %0 = qecp.extract_block %arg0[ 0] : !qecp.hyperreg<3 x 1 x 7> -> !qecp.codeblock<1 x 7>
    func.return
}

// -----

func.func @test_extract_block_dyn_idx(%arg0 : !qecp.hyperreg<3 x 1 x 7>, %arg1 : index) {
    %0 = qecp.extract_block %arg0[ %arg1] : !qecp.hyperreg<3 x 1 x 7> -> !qecp.codeblock<1 x 7>
    func.return
}

// -----

func.func @test_insert_block_static_idx(%arg0 : !qecp.hyperreg<3 x 1 x 7>, %arg1 : !qecp.codeblock<1 x 7>) {
    %0 = qecp.insert_block %arg0[ 0], %arg1 : !qecp.hyperreg<3 x 1 x 7>, !qecp.codeblock<1 x 7>
    func.return
}

// -----

func.func @test_insert_block_dyn_idx(%arg0 : !qecp.hyperreg<3 x 1 x 7>, %arg1 : index, %arg2 : !qecp.codeblock<1 x 7>) {
    %0 = qecp.insert_block %arg0[ %arg1], %arg2 : !qecp.hyperreg<3 x 1 x 7>, !qecp.codeblock<1 x 7>
    func.return
}

// -----

func.func @test_extract_qubit_static_idx(%arg0 : !qecp.codeblock<1 x 7>) {
    %0 = qecp.extract %arg0[ 0] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
    func.return
}

// -----

func.func @test_extract_qubit_dyn_idx(%arg0 : !qecp.codeblock<1 x 7>, %arg1 : index) {
    %0 = qecp.extract %arg0[ %arg1] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
    func.return
}

// -----

func.func @test_insert_qubit_static_idx(%arg0 : !qecp.codeblock<1 x 7>, %arg1 : !qecp.qubit<data>) {
    %0 = qecp.insert %arg0[ 0], %arg1 : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
    func.return
}

// -----

func.func @test_insert_qubit_dyn_idx(%arg0 : !qecp.codeblock<1 x 7>, %arg1 : index, %arg2 : !qecp.qubit<data>) {
    %0 = qecp.insert %arg0[ %arg1], %arg2 : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
    func.return
}

// -----

func.func @test_alloc_aux() {
    %0 = qecp.alloc_aux : !qecp.qubit<aux>
    func.return
}

// -----

func.func @test_dealloc_aux(%arg0 : !qecp.qubit<aux>) {
    qecp.dealloc_aux %arg0 : !qecp.qubit<aux>
    func.return
}

// -----

func.func @test_gate_op_identity(%arg0 : !qecp.qubit<data>, %arg1 : !qecp.qubit<aux>) {
    %0 = qecp.identity %arg0 : !qecp.qubit<data>
    %1 = qecp.identity %arg1 : !qecp.qubit<aux>
    func.return
}

// -----

func.func @test_gate_op_pauli_x(%arg0 : !qecp.qubit<data>, %arg1 : !qecp.qubit<aux>) {
    %0 = qecp.x %arg0 : !qecp.qubit<data>
    %1 = qecp.x %arg1 : !qecp.qubit<aux>
    func.return
}

// -----

func.func @test_gate_op_pauli_y(%arg0 : !qecp.qubit<data>, %arg1 : !qecp.qubit<aux>) {
    %0 = qecp.y %arg0 : !qecp.qubit<data>
    %1 = qecp.y %arg1 : !qecp.qubit<aux>
    func.return
}

// -----

func.func @test_gate_op_pauli_z(%arg0 : !qecp.qubit<data>, %arg1 : !qecp.qubit<aux>) {
    %0 = qecp.z %arg0 : !qecp.qubit<data>
    %1 = qecp.z %arg1 : !qecp.qubit<aux>
    func.return
}

// -----

func.func @test_gate_op_hadamard(%arg0 : !qecp.qubit<data>, %arg1 : !qecp.qubit<aux>) {
    %0 = qecp.hadamard %arg0 : !qecp.qubit<data>
    %1 = qecp.hadamard %arg1 : !qecp.qubit<aux>
    func.return
}

// -----

func.func @test_gate_op_s(%arg0 : !qecp.qubit<data>, %arg1 : !qecp.qubit<aux>) {
    %0 = qecp.s %arg0 : !qecp.qubit<data>
    %1 = qecp.s %0 adj : !qecp.qubit<data>
    %2 = qecp.s %arg1 : !qecp.qubit<aux>
    %3 = qecp.s %2 adj : !qecp.qubit<aux>
    func.return
}

// -----

func.func @test_gate_op_rot(%phi : f64, %theta : f64,  %omega : f64, %arg0 : !qecp.qubit<data>, %arg1 : !qecp.qubit<aux>) {
    %0 = qecp.rot(%phi, %theta, %omega) %arg0 : !qecp.qubit<data>
    %1 = qecp.rot(%phi, %theta, %omega) %0 : !qecp.qubit<data>
    %2 = qecp.rot(%phi, %theta, %omega) %arg1 : !qecp.qubit<aux>
    %3 = qecp.rot(%phi, %theta, %omega) %2 : !qecp.qubit<aux>
    func.return
}

// -----

func.func @test_gate_op_cnot(
    %arg0 : !qecp.qubit<data>, %arg1 : !qecp.qubit<data>,
    %arg2 : !qecp.qubit<aux>, %arg3 : !qecp.qubit<aux>
) {
    %0, %1 = qecp.cnot %arg0, %arg1 : !qecp.qubit<data>, !qecp.qubit<data>
    %2, %3 = qecp.cnot %arg2, %arg3 : !qecp.qubit<aux>, !qecp.qubit<aux>
    %4, %5 = qecp.cnot %0, %2 : !qecp.qubit<data>, !qecp.qubit<aux>
    %6, %7 = qecp.cnot %3, %1 : !qecp.qubit<aux>, !qecp.qubit<data>
    func.return
}

// -----

func.func @test_tanner_graph_type(%arg0 : !qecp.tanner_graph<8, 6, i32>) {
    func.return
}

// -----

func.func @test_assemble_tanner_graph_tensor(%arg0 : tensor<8xi32>, %arg1 : tensor<6xi32>) {
    %0 = qecp.assemble_tanner %arg0, %arg1 : tensor<8xi32>, tensor<6xi32> -> !qecp.tanner_graph<8, 6, i32>
    func.return
}

// -----

func.func @test_measure(%arg0 : !qecp.qubit<data>, %arg1 : !qecp.qubit<aux>) {
    %mres0, %0 = qecp.measure %arg0 : i1, !qecp.qubit<data>
    %mres1, %1 = qecp.measure %arg1 : i1, !qecp.qubit<aux>
    func.return
}

// -----

func.func @test_assemble_tanner_graph_memref(%arg0 : memref<8xi32>, %arg1 : memref<6xi32>) {
    %0 = qecp.assemble_tanner %arg0, %arg1 : memref<8xi32>, memref<6xi32> -> !qecp.tanner_graph<8, 6, i32>
    func.return
}

// -----

func.func @test_decode_esm_css(%arg0 : !qecp.tanner_graph<8, 6, i32>, %arg1 : tensor<2xi1>) {
    %0 = qecp.decode_esm_css(%arg0 : !qecp.tanner_graph<8, 6, i32>) %arg1 : tensor<2xi1> -> tensor<1xindex>
    func.return
}

// -----

func.func @test_decode_physical_meas(%arg1 : tensor<7xi1>) {
    %0 = qecp.decode_physical_meas %arg1 : tensor<7xi1> -> tensor<1xi1>
    func.return
}
