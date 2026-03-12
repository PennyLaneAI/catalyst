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

func.func @test_codeblock(%arg0 : !qecl.codeblock<1>) {
    func.return
}

// -----

// TODO: Currently we must write parameters as '<3 x 1>' (with spaces)
func.func @test_hyperreg(%arg0 : !qecl.hyperreg<3 x 1>) {
    func.return
}

// -----

func.func @test_alloc() {
    %0 = qecl.alloc() : !qecl.hyperreg<3 x 1>
    func.return
}

// -----

func.func @test_dealloc(%arg0 : !qecl.hyperreg<3 x 1>) {
    qecl.dealloc %arg0 : !qecl.hyperreg<3 x 1>
    func.return
}

// -----

func.func @test_extract_block_static_idx(%arg0 : !qecl.hyperreg<3 x 1>) {
    %0 = qecl.extract_block %arg0[ 0] : !qecl.hyperreg<3 x 1> -> !qecl.codeblock<1>
    func.return
}

// -----

func.func @test_extract_block_dyn_idx(%arg0 : !qecl.hyperreg<3 x 1>, %arg1 : index) {
    %0 = qecl.extract_block %arg0[ %arg1] : !qecl.hyperreg<3 x 1> -> !qecl.codeblock<1>
    func.return
}

// -----

func.func @test_insert_block_static_idx(%arg0 : !qecl.hyperreg<3 x 1>, %arg1 : !qecl.codeblock<1>) {
    %0 = qecl.insert_block %arg0[ 0], %arg1 : !qecl.hyperreg<3 x 1>, !qecl.codeblock<1>
    func.return
}

// -----

func.func @test_insert_block_dyn_idx(%arg0 : !qecl.hyperreg<3 x 1>, %arg1 : index, %arg2 : !qecl.codeblock<1>) {
    %0 = qecl.insert_block %arg0[ %arg1], %arg2 : !qecl.hyperreg<3 x 1>, !qecl.codeblock<1>
    func.return
}

// -----

func.func @test_encode_block(%arg0 : !qecl.codeblock<1>) {
    %0 = qecl.encode [zero] %arg0 : !qecl.codeblock<1>
    func.return
}

// -----

func.func @test_qec_cycle(%arg0 : !qecl.codeblock<1>) {
    %0 = qecl.qec %arg0 : !qecl.codeblock<1>
    func.return
}
