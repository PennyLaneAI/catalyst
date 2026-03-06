// Copyright 2026 Xanadu qref Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --canonicalize --cse --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: test_alloc_dce
func.func @test_alloc_dce() {
    // CHECK-NOT: qecp.alloc
    %r = qecp.alloc() : !qecp.hyperreg<3 x 1 x 7>
    return
}

// -----

// CHECK-LABEL: test_alloc_no_cse
func.func @test_alloc_no_cse() -> (!qecp.hyperreg<3 x 1 x 7>, !qecp.hyperreg<3 x 1 x 7>) {
    // CHECK: qecp.alloc
    // CHECK-NEXT: qecp.alloc
    %r1 = qecp.alloc() : !qecp.hyperreg<3 x 1 x 7>
    %r2 = qecp.alloc() : !qecp.hyperreg<3 x 1 x 7>
    return %r1, %r2 : !qecp.hyperreg<3 x 1 x 7>, !qecp.hyperreg<3 x 1 x 7>
}

// -----

// CHECK-LABEL: test_alloc_dealloc_fold
func.func @test_alloc_dealloc_fold() {
    // CHECK-NOT: qecp.alloc
    // CHECK-NOT: qecp.dealloc
    %r = qecp.alloc() : !qecp.hyperreg<3 x 1 x 7>
    qecp.dealloc %r : !qecp.hyperreg<3 x 1 x 7>
    return
}

// -----

// CHECK-LABEL: test_alloc_dealloc_no_fold
func.func @test_alloc_dealloc_no_fold() -> !qecp.codeblock<1 x 7> {
    // CHECK: qecp.alloc
    // CHECK: qecp.dealloc
    %r = qecp.alloc() : !qecp.hyperreg<3 x 1 x 7>
    %b = qecp.extract_block %r[0] : !qecp.hyperreg<3 x 1 x 7> -> !qecp.codeblock<1 x 7>
    qecp.dealloc %r : !qecp.hyperreg<3 x 1 x 7>
    return %b : !qecp.codeblock<1 x 7>
}

// -----

// CHECK-LABEL: test_extract_insert_dce
// CHECK: ([[r0:%.+]]{{\s*}}: !qecp.hyperreg
func.func @test_extract_insert_dce(
    %r0 : !qecp.hyperreg<3 x 1 x 7>, %i : index
) -> !qecp.hyperreg<3 x 1 x 7> {
    // CHECK-NOT: qecp.extract_block
    // CHECK-NOT: qecp.insert_block
    %b0 = qecp.extract_block %r0[0] : !qecp.hyperreg<3 x 1 x 7> -> !qecp.codeblock<1 x 7>
    %r1 = qecp.insert_block %r0[0], %b0 : !qecp.hyperreg<3 x 1 x 7>, !qecp.codeblock<1 x 7>

    // CHECK-NOT: qecp.extract_block
    // CHECK-NOT: qecp.insert_block
    %b2 = qecp.extract_block %r1[%i] : !qecp.hyperreg<3 x 1 x 7> -> !qecp.codeblock<1 x 7>
    %r2 = qecp.insert_block %r1[%i], %b2 : !qecp.hyperreg<3 x 1 x 7>, !qecp.codeblock<1 x 7>

    // CHECK: return [[r0]]
    return %r2 : !qecp.hyperreg<3 x 1 x 7>
}

// -----

// CHECK-LABEL: test_extract_insert_no_dce
func.func @test_extract_insert_no_dce(
    %r0 : !qecp.hyperreg<3 x 1 x 7>, %i0 : index, %i1 : index
) -> !qecp.hyperreg<3 x 1 x 7> {
    // CHECK: qecp.extract_block
    // CHECK: qecp.insert_block
    %b0 = qecp.extract_block %r0[0] : !qecp.hyperreg<3 x 1 x 7> -> !qecp.codeblock<1 x 7>
    %r1 = qecp.insert_block %r0[1], %b0 : !qecp.hyperreg<3 x 1 x 7>, !qecp.codeblock<1 x 7>

    // CHECK: qecp.extract_block
    // CHECK: [[r2:%.+]] = qecp.insert_block
    %b2 = qecp.extract_block %r1[%i0] : !qecp.hyperreg<3 x 1 x 7> -> !qecp.codeblock<1 x 7>
    %r2 = qecp.insert_block %r1[%i1], %b2 : !qecp.hyperreg<3 x 1 x 7>, !qecp.codeblock<1 x 7>

    // CHECK: return [[r2]]
    return %r2 : !qecp.hyperreg<3 x 1 x 7>
}

// -----

// CHECK-LABEL: test_insert_extract_dce
// CHECK: ([[r0:%.+]]{{\s*}}: !qecp.hyperreg{{.*}}, [[b0:%.+]]{{\s*}}: !qecp.codeblock
func.func @test_insert_extract_dce(
    %r0 : !qecp.hyperreg<3 x 1 x 7>, %b0 : !qecp.codeblock<1 x 7>, %i : index
) -> !qecp.codeblock<1 x 7> {
    // CHECK-NOT: qecp.insert_block
    // CHECK-NOT: qecp.extract_block
    %r1 = qecp.insert_block %r0[0], %b0 : !qecp.hyperreg<3 x 1 x 7>, !qecp.codeblock<1 x 7>
    %b1 = qecp.extract_block %r1[0] : !qecp.hyperreg<3 x 1 x 7> -> !qecp.codeblock<1 x 7>

    // CHECK-NOT: qecp.insert_block
    // CHECK-NOT: qecp.extract_block
    %r2 = qecp.insert_block %r1[%i], %b1 : !qecp.hyperreg<3 x 1 x 7>, !qecp.codeblock<1 x 7>
    %b2 = qecp.extract_block %r2[%i] : !qecp.hyperreg<3 x 1 x 7> -> !qecp.codeblock<1 x 7>

    // CHECK: return [[b0]]
    return %b2 : !qecp.codeblock<1 x 7>
}

// -----

// CHECK-LABEL: test_insert_extract_no_dce
func.func @test_insert_extract_no_dce(
    %r0 : !qecp.hyperreg<3 x 1 x 7>, %b0 : !qecp.codeblock<1 x 7>, %i0 : index, %i1 : index
) -> !qecp.codeblock<1 x 7> {
    // CHECK: qecp.insert_block
    // CHECK: qecp.extract_block
    %r1 = qecp.insert_block %r0[0], %b0 : !qecp.hyperreg<3 x 1 x 7>, !qecp.codeblock<1 x 7>
    %b1 = qecp.extract_block %r1[1] : !qecp.hyperreg<3 x 1 x 7> -> !qecp.codeblock<1 x 7>

    // CHECK: qecp.insert_block
    // CHECK: [[b2:%.+]] = qecp.extract_block
    %r2 = qecp.insert_block %r1[%i0], %b1 : !qecp.hyperreg<3 x 1 x 7>, !qecp.codeblock<1 x 7>
    %b2 = qecp.extract_block %r2[%i1] : !qecp.hyperreg<3 x 1 x 7> -> !qecp.codeblock<1 x 7>

    // CHECK: return [[b2]]
    return %b2 : !qecp.codeblock<1 x 7>
}

// -----

// CHECK-LABEL: test_extract_block_const_idx_fold
func.func @test_extract_block_const_idx_fold() -> !qecp.codeblock<1 x 7> {
    // CHECK-NOT: arith.constant
    // CHECK: [[hreg:%.+]] = qecp.alloc
    // CHECK: qecp.extract_block [[hreg]][ 0]
    %0 = arith.constant 0 : index
    %r = qecp.alloc() : !qecp.hyperreg<3 x 1 x 7>
    %b = qecp.extract_block %r[%0] : !qecp.hyperreg<3 x 1 x 7> -> !qecp.codeblock<1 x 7>

    return %b : !qecp.codeblock<1 x 7>
}

// -----

// CHECK-LABEL: test_insert_block_const_idx_fold
func.func @test_insert_block_const_idx_fold(%b : !qecp.codeblock<1 x 7>) -> !qecp.hyperreg<3 x 1 x 7> {
    // CHECK-NOT: arith.constant
    // CHECK: [[hreg:%.+]] = qecp.alloc
    // CHECK: qecp.insert_block [[hreg]][ 0]
    %0 = arith.constant 0 : index
    %r0 = qecp.alloc() : !qecp.hyperreg<3 x 1 x 7>
    %r1 = qecp.insert_block %r0[%0], %b : !qecp.hyperreg<3 x 1 x 7>, !qecp.codeblock<1 x 7>

    return %r1 : !qecp.hyperreg<3 x 1 x 7>
}

// -----

// CHECK-LABEL: test_extract_qubit_const_idx_fold
func.func @test_extract_qubit_const_idx_fold() -> !qecp.qubit<data> {
    // CHECK-NOT: arith.constant
    // CHECK: [[block:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 7>
    // CHECK: qecp.extract [[block]][ 0]
    %0 = arith.constant 0 : index
    %b = "test.op"() : () -> !qecp.codeblock<1 x 7>
    %q = qecp.extract %b[%0] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>

    return %q : !qecp.qubit<data>
}

// -----

// CHECK-LABEL: test_insert_qubit_const_idx_fold
func.func @test_insert_qubit_const_idx_fold() -> !qecp.codeblock<1 x 7> {
    // CHECK-NOT: arith.constant
    // CHECK: [[qubit:%.+]] = "test.op"() : () -> !qecp.qubit<data>
    // CHECK: [[block:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 7>
    // CHECK: qecp.insert [[block]][ 0], [[qubit]]
    %0 = arith.constant 0 : index
    %q = "test.op"() : () -> !qecp.qubit<data>
    %b = "test.op"() : () -> !qecp.codeblock<1 x 7>
    %b1 = qecp.insert %b[%0], %q : !qecp.codeblock<1 x 7>, !qecp.qubit<data>

    return %b1 : !qecp.codeblock<1 x 7>
}

// -----

// CHECK-LABEL: test_alloc_aux_dce
func.func @test_alloc_aux_dce() {
    // CHECK-NOT: qecp.alloc_aux
    %q = qecp.alloc_aux : !qecp.qubit<aux>
    return
}

// -----

// CHECK-LABEL: test_alloc_aux_no_cse
func.func @test_alloc_aux_no_cse() -> (!qecp.qubit<aux>, !qecp.qubit<aux>) {
    // CHECK: qecp.alloc_aux
    // CHECK-NEXT: qecp.alloc_aux
    %q1 = qecp.alloc_aux : !qecp.qubit<aux>
    %q2 = qecp.alloc_aux : !qecp.qubit<aux>
    return %q1, %q2 : !qecp.qubit<aux>, !qecp.qubit<aux>
}

// -----

// CHECK-LABEL: test_alloc_dealloc_aux_fold
func.func @test_alloc_dealloc_aux_fold() {
    // CHECK-NOT: qecp.alloc_aux
    // CHECK-NOT: qecp.dealloc_aux
    %q = qecp.alloc_aux : !qecp.qubit<aux>
    qecp.dealloc_aux %q : !qecp.qubit<aux>
    return
}

// -----

// CHECK-LABEL: test_alloc_dealloc_aux_no_fold
func.func @test_alloc_dealloc_aux_no_fold() {
    // CHECK: qecp.alloc_aux
    // CHECK: qecp.dealloc_aux
    %q0 = qecp.alloc_aux : !qecp.qubit<aux>
    %q1 = "test.op"(%q0) : (!qecp.qubit<aux>) -> (!qecp.qubit<aux>)
    qecp.dealloc_aux %q1 : !qecp.qubit<aux>
    return
}
