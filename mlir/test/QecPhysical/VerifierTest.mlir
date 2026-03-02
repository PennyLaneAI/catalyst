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

func.func @test_extract_no_idx(%r : !qecp.hyperreg<3 x 1 x 7>) {
    // expected-error@below {{expected to have a non-null index}}
    %b = "qecp.extract_block"(%r) <{}> : (!qecp.hyperreg<3 x 1 x 7>) -> !qecp.codeblock<1 x 7>
    return
}

// -----

func.func @test_insert_no_idx(%r : !qecp.hyperreg<3 x 1 x 7>, %b : !qecp.codeblock<1 x 7>) {
    // expected-error@below {{expected to have a non-null index}}
    %r1 = "qecp.insert_block"(%r, %b) <{}> : (!qecp.hyperreg<3 x 1 x 7>, !qecp.codeblock<1 x 7>) -> !qecp.hyperreg<3 x 1 x 7>
    return
}

// -----

func.func @test_extract_negative_index(%r : !qecp.hyperreg<3 x 1 x 7>) {
    // expected-error@below {{attribute 'idx_attr' failed to satisfy constraint}}
    %b = qecp.extract_block %r[-1] : !qecp.hyperreg<3 x 1 x 7> -> !qecp.codeblock<1 x 7>
    return
}

// -----

func.func @test_insert_negative_index(%r : !qecp.hyperreg<3 x 1 x 7>, %b : !qecp.codeblock<1 x 7>) {
    // expected-error@below {{attribute 'idx_attr' failed to satisfy constraint}}
    %r1 = qecp.insert_block %r[-1], %b : !qecp.hyperreg<3 x 1 x 7>, !qecp.codeblock<1 x 7>
    return
}

// -----

func.func @test_extract_index_out_of_bounds(%r : !qecp.hyperreg<3 x 1 x 7>) {
    // expected-error@below {{out-of-bounds index attribute}}
    %b = qecp.extract_block %r[3] : !qecp.hyperreg<3 x 1 x 7> -> !qecp.codeblock<1 x 7>
    return
}

// -----

func.func @test_insert_index_out_of_bounds(%r : !qecp.hyperreg<3 x 1 x 7>, %b : !qecp.codeblock<1 x 7>) {
    // expected-error@below {{out-of-bounds index attribute}}
    %r1 = qecp.insert_block %r[3], %b : !qecp.hyperreg<3 x 1 x 7>, !qecp.codeblock<1 x 7>
    return
}

// -----

func.func @test_extract_type_mismatch_k(%r : !qecp.hyperreg<3 x 1 x 7>) {
    // expected-error@below {{expected hyper-register and codeblock types to have same value of k}}
    %b1 = qecp.extract_block %r[0] : !qecp.hyperreg<3 x 1 x 7> -> !qecp.codeblock<2 x 7>
    return
}

// -----

func.func @test_extract_type_mismatch_n(%r : !qecp.hyperreg<3 x 1 x 7>) {
    // expected-error@below {{expected hyper-register and codeblock types to have same value of n}}
    %b1 = qecp.extract_block %r[0] : !qecp.hyperreg<3 x 1 x 7> -> !qecp.codeblock<1 x 5>
    return
}

// -----

func.func @test_insert_type_mismatch_k(%r : !qecp.hyperreg<3 x 1 x 7>, %b : !qecp.codeblock<2 x 7>) {
    // expected-error@below {{expected hyper-register and codeblock types to have same value of k}}
    %b1 = qecp.insert_block %r[0], %b : !qecp.hyperreg<3 x 1 x 7>, !qecp.codeblock<2 x 7>
    return
}

// -----

func.func @test_insert_type_mismatch_n(%r : !qecp.hyperreg<3 x 1 x 7>, %b : !qecp.codeblock<1 x 5>) {
    // expected-error@below {{expected hyper-register and codeblock types to have same value of n}}
    %b1 = qecp.insert_block %r[0], %b : !qecp.hyperreg<3 x 1 x 7>, !qecp.codeblock<1 x 5>
    return
}

// -----

func.func @test_alloc_aux_invalid_qubit() -> !qecp.qubit<data> {
    // expected-error@below {{expected a QEC physical qubit with role 'aux', but got 'data'}}
    %q = qecp.alloc_aux : !qecp.qubit<data>
    return %q : !qecp.qubit<data>
}

// -----

func.func @test_dealloc_aux_invalid_qubit(%q : !qecp.qubit<data>) {
    // expected-error@below {{expected a QEC physical qubit with role 'aux', but got 'data'}}
    qecp.dealloc_aux %q : !qecp.qubit<data>
    return
}
