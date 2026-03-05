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

func.func @test_alloc_aux() {
    %0 = qecp.alloc_aux : !qecp.qubit<aux>
    func.return
}

// -----

func.func @test_dealloc_aux(%arg0 : !qecp.qubit<aux>) {
    qecp.dealloc_aux %arg0 : !qecp.qubit<aux>
    func.return
}
