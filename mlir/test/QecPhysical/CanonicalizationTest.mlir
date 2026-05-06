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
func.func @test_alloc_dealloc_no_fold() {
    // CHECK: qecp.alloc
    // CHECK: qecp.dealloc
    %r = qecp.alloc() : !qecp.hyperreg<3 x 1 x 7>
    %b = "test.op"(%r) : (!qecp.hyperreg<3 x 1 x 7>) -> !qecp.hyperreg<3 x 1 x 7>
    qecp.dealloc %r : !qecp.hyperreg<3 x 1 x 7>
    return
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
    %q1 = "test.op"(%q0) : (!qecp.qubit<aux>) -> !qecp.qubit<aux>
    qecp.dealloc_aux %q1 : !qecp.qubit<aux>
    return
}

// -----

// CHECK-LABEL: test_extract_insert_block_dce
func.func @test_extract_insert_block_dce(%i : index) -> !qecp.hyperreg<3 x 1 x 7> {
    // CHECK: [[r0:%.+]] = "test.op"() : () -> !qecp.hyperreg<3 x 1 x 7>
    %r0 = "test.op"() : () -> !qecp.hyperreg<3 x 1 x 7>

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

// CHECK-LABEL: test_extract_insert_block_no_dce
func.func @test_extract_insert_block_no_dce(%i0 : index, %i1 : index) -> !qecp.hyperreg<3 x 1 x 7> {
    %r0 = "test.op"() : () -> !qecp.hyperreg<3 x 1 x 7>

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

// CHECK-LABEL: test_insert_extract_block_dce
func.func @test_insert_extract_block_dce(%i : index) -> !qecp.codeblock<1 x 7> {
    // CHECK: [[r0:%.+]] = "test.op"() : () -> !qecp.hyperreg<3 x 1 x 7>
    // CHECK: [[b0:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 7>
    %r0 = "test.op"() : () -> !qecp.hyperreg<3 x 1 x 7>
    %b0 = "test.op"() : () -> !qecp.codeblock<1 x 7>

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

// CHECK-LABEL: test_insert_extract_block_no_dce
func.func @test_insert_extract_block_no_dce(%i0 : index, %i1 : index) -> !qecp.codeblock<1 x 7> {
    %r0 = "test.op"() : () -> !qecp.hyperreg<3 x 1 x 7>
    %b0 = "test.op"() : () -> !qecp.codeblock<1 x 7>

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

// CHECK-LABEL: test_extract_insert_qubit_dce
func.func @test_extract_insert_qubit_dce(%i : index) -> !qecp.codeblock<1 x 7> {
    // CHECK: [[r0:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 7>
    %r0 = "test.op"() : () -> !qecp.codeblock<1 x 7>

    // CHECK-NOT: qecp.extract
    // CHECK-NOT: qecp.insert_block
    %b0 = qecp.extract %r0[0] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
    %r1 = qecp.insert %r0[0], %b0 : !qecp.codeblock<1 x 7>, !qecp.qubit<data>

    // CHECK-NOT: qecp.extract
    // CHECK-NOT: qecp.insert
    %b2 = qecp.extract %r1[%i] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
    %r2 = qecp.insert %r1[%i], %b2 : !qecp.codeblock<1 x 7>, !qecp.qubit<data>

    // CHECK: return [[r0]]
    return %r2 : !qecp.codeblock<1 x 7>
}

// -----

// CHECK-LABEL: test_extract_insert_qubit_no_dce
func.func @test_extract_insert_qubit_no_dce(%i0 : index, %i1 : index) -> !qecp.codeblock<1 x 7> {
    %r0 = "test.op"() : () -> !qecp.codeblock<1 x 7>

    // CHECK: qecp.extract
    // CHECK: qecp.insert
    %b0 = qecp.extract %r0[0] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
    %r1 = qecp.insert %r0[1], %b0 : !qecp.codeblock<1 x 7>, !qecp.qubit<data>

    // CHECK: qecp.extract
    // CHECK: [[r2:%.+]] = qecp.insert
    %b2 = qecp.extract %r1[%i0] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>
    %r2 = qecp.insert %r1[%i1], %b2 : !qecp.codeblock<1 x 7>, !qecp.qubit<data>

    // CHECK: return [[r2]]
    return %r2 : !qecp.codeblock<1 x 7>
}

// -----

// CHECK-LABEL: test_insert_extract_qubit_dce
func.func @test_insert_extract_qubit_dce(%i : index) -> !qecp.qubit<data> {
    // CHECK: [[r0:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 7>
    // CHECK: [[b0:%.+]] = "test.op"() : () -> !qecp.qubit<data>
    %r0 = "test.op"() : () -> !qecp.codeblock<1 x 7>
    %b0 = "test.op"() : () -> !qecp.qubit<data>

    // CHECK-NOT: qecp.insert
    // CHECK-NOT: qecp.extract
    %r1 = qecp.insert %r0[0], %b0 : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
    %b1 = qecp.extract %r1[0] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>

    // CHECK-NOT: qecp.insert
    // CHECK-NOT: qecp.extract
    %r2 = qecp.insert %r1[%i], %b1 : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
    %b2 = qecp.extract %r2[%i] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>

    // CHECK: return [[b0]]
    return %b2 : !qecp.qubit<data>
}

// -----

// CHECK-LABEL: test_insert_extract_qubit_no_dce
func.func @test_insert_extract_qubit_no_dce(%i0 : index, %i1 : index) -> !qecp.qubit<data> {
    %r0 = "test.op"() : () -> !qecp.codeblock<1 x 7>
    %b0 = "test.op"() : () -> !qecp.qubit<data>

    // CHECK: qecp.insert
    // CHECK: qecp.extract
    %r1 = qecp.insert %r0[0], %b0 : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
    %b1 = qecp.extract %r1[1] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>

    // CHECK: qecp.insert
    // CHECK: [[b2:%.+]] = qecp.extract
    %r2 = qecp.insert %r1[%i0], %b1 : !qecp.codeblock<1 x 7>, !qecp.qubit<data>
    %b2 = qecp.extract %r2[%i1] : !qecp.codeblock<1 x 7> -> !qecp.qubit<data>

    // CHECK: return [[b2]]
    return %b2 : !qecp.qubit<data>
}

// -----

// CHECK-LABEL: test_extract_block_const_idx_fold
func.func @test_extract_block_const_idx_fold() -> !qecp.codeblock<1 x 7> {
    // CHECK-NOT: arith.constant
    // CHECK: [[hreg:%.+]] = "test.op"() : () -> !qecp.hyperreg<3 x 1 x 7>
    // CHECK: qecp.extract_block [[hreg]][ 0]
    %0 = arith.constant 0 : index
    %r = "test.op"() : () -> !qecp.hyperreg<3 x 1 x 7>
    %b = qecp.extract_block %r[%0] : !qecp.hyperreg<3 x 1 x 7> -> !qecp.codeblock<1 x 7>

    return %b : !qecp.codeblock<1 x 7>
}

// -----

// CHECK-LABEL: test_insert_block_const_idx_fold
func.func @test_insert_block_const_idx_fold() -> !qecp.hyperreg<3 x 1 x 7> {
    // CHECK-NOT: arith.constant
    // CHECK: [[hreg:%.+]] = "test.op"() : () -> !qecp.hyperreg<3 x 1 x 7>
    // CHECK: [[block:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 7>
    // CHECK: qecp.insert_block [[hreg]][ 0], [[block]]
    %0 = arith.constant 0 : index
    %r0 = "test.op"() : () -> !qecp.hyperreg<3 x 1 x 7>
    %b = "test.op"() : () -> !qecp.codeblock<1 x 7>
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
    // CHECK: [[block:%.+]] = "test.op"() : () -> !qecp.codeblock<1 x 7>
    // CHECK: [[qubit:%.+]] = "test.op"() : () -> !qecp.qubit<data>
    // CHECK: qecp.insert [[block]][ 0], [[qubit]]
    %0 = arith.constant 0 : index
    %b = "test.op"() : () -> !qecp.codeblock<1 x 7>
    %q = "test.op"() : () -> !qecp.qubit<data>
    %b1 = qecp.insert %b[%0], %q : !qecp.codeblock<1 x 7>, !qecp.qubit<data>

    return %b1 : !qecp.codeblock<1 x 7>
}
