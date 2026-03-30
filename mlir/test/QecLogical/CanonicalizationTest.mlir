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
    // CHECK-NOT: qecl.alloc
    %r = qecl.alloc() : !qecl.hyperreg<3 x 1>
    return
}

// -----

// CHECK-LABEL: test_alloc_no_cse
func.func @test_alloc_no_cse() -> (!qecl.hyperreg<3 x 1>, !qecl.hyperreg<3 x 1>) {
    // CHECK: qecl.alloc
    // CHECK-NEXT: qecl.alloc
    %r1 = qecl.alloc() : !qecl.hyperreg<3 x 1>
    %r2 = qecl.alloc() : !qecl.hyperreg<3 x 1>
    return %r1, %r2 : !qecl.hyperreg<3 x 1>, !qecl.hyperreg<3 x 1>
}

// -----

// CHECK-LABEL: test_alloc_dealloc_fold
func.func @test_alloc_dealloc_fold() {
    // CHECK-NOT: qecl.alloc
    // CHECK-NOT: qecl.dealloc
    %r = qecl.alloc() : !qecl.hyperreg<3 x 1>
    qecl.dealloc %r : !qecl.hyperreg<3 x 1>
    return
}

// -----

// CHECK-LABEL: test_alloc_dealloc_no_fold
func.func @test_alloc_dealloc_no_fold() -> !qecl.codeblock<1> {
    // CHECK: qecl.alloc
    // CHECK: qecl.dealloc
    %r = qecl.alloc() : !qecl.hyperreg<3 x 1>
    %b = qecl.extract_block %r[0] : !qecl.hyperreg<3 x 1> -> !qecl.codeblock<1>
    qecl.dealloc %r : !qecl.hyperreg<3 x 1>
    return %b : !qecl.codeblock<1>
}

// -----

// CHECK-LABEL: test_extract_insert_dce
// CHECK: ([[r0:%.+]]{{\s*}}: !qecl.hyperreg
func.func @test_extract_insert_dce(
    %r0 : !qecl.hyperreg<3 x 1>, %i : index
) -> !qecl.hyperreg<3 x 1> {
    // CHECK-NOT: qecl.extract_block
    // CHECK-NOT: qecl.insert_block
    %b0 = qecl.extract_block %r0[0] : !qecl.hyperreg<3 x 1> -> !qecl.codeblock<1>
    %r1 = qecl.insert_block %r0[0], %b0 : !qecl.hyperreg<3 x 1>, !qecl.codeblock<1>

    // CHECK-NOT: qecl.extract_block
    // CHECK-NOT: qecl.insert_block
    %b2 = qecl.extract_block %r1[%i] : !qecl.hyperreg<3 x 1> -> !qecl.codeblock<1>
    %r2 = qecl.insert_block %r1[%i], %b2 : !qecl.hyperreg<3 x 1>, !qecl.codeblock<1>

    // CHECK: return [[r0]]
    return %r2 : !qecl.hyperreg<3 x 1>
}

// -----

// CHECK-LABEL: test_extract_insert_no_dce
func.func @test_extract_insert_no_dce(
    %r0 : !qecl.hyperreg<3 x 1>, %i0 : index, %i1 : index
) -> !qecl.hyperreg<3 x 1> {
    // CHECK: qecl.extract_block
    // CHECK: qecl.insert_block
    %b0 = qecl.extract_block %r0[0] : !qecl.hyperreg<3 x 1> -> !qecl.codeblock<1>
    %r1 = qecl.insert_block %r0[1], %b0 : !qecl.hyperreg<3 x 1>, !qecl.codeblock<1>

    // CHECK: qecl.extract_block
    // CHECK: [[r2:%.+]] = qecl.insert_block
    %b2 = qecl.extract_block %r1[%i0] : !qecl.hyperreg<3 x 1> -> !qecl.codeblock<1>
    %r2 = qecl.insert_block %r1[%i1], %b2 : !qecl.hyperreg<3 x 1>, !qecl.codeblock<1>

    // CHECK: return [[r2]]
    return %r2 : !qecl.hyperreg<3 x 1>
}

// -----

// CHECK-LABEL: test_insert_extract_dce
// CHECK: ([[r0:%.+]]{{\s*}}: !qecl.hyperreg{{.*}}, [[b0:%.+]]{{\s*}}: !qecl.codeblock
func.func @test_insert_extract_dce(
    %r0 : !qecl.hyperreg<3 x 1>, %b0 : !qecl.codeblock<1>, %i : index
) -> !qecl.codeblock<1> {
    // CHECK-NOT: qecl.insert_block
    // CHECK-NOT: qecl.extract_block
    %r1 = qecl.insert_block %r0[0], %b0 : !qecl.hyperreg<3 x 1>, !qecl.codeblock<1>
    %b1 = qecl.extract_block %r1[0] : !qecl.hyperreg<3 x 1> -> !qecl.codeblock<1>

    // CHECK-NOT: qecl.insert_block
    // CHECK-NOT: qecl.extract_block
    %r2 = qecl.insert_block %r1[%i], %b1 : !qecl.hyperreg<3 x 1>, !qecl.codeblock<1>
    %b2 = qecl.extract_block %r2[%i] : !qecl.hyperreg<3 x 1> -> !qecl.codeblock<1>

    // CHECK: return [[b0]]
    return %b2 : !qecl.codeblock<1>
}

// -----

// CHECK-LABEL: test_insert_extract_no_dce
func.func @test_insert_extract_no_dce(
    %r0 : !qecl.hyperreg<3 x 1>, %b0 : !qecl.codeblock<1>, %i0 : index, %i1 : index
) -> !qecl.codeblock<1> {
    // CHECK: qecl.insert_block
    // CHECK: qecl.extract_block
    %r1 = qecl.insert_block %r0[0], %b0 : !qecl.hyperreg<3 x 1>, !qecl.codeblock<1>
    %b1 = qecl.extract_block %r1[1] : !qecl.hyperreg<3 x 1> -> !qecl.codeblock<1>

    // CHECK: qecl.insert_block
    // CHECK: [[b2:%.+]] = qecl.extract_block
    %r2 = qecl.insert_block %r1[%i0], %b1 : !qecl.hyperreg<3 x 1>, !qecl.codeblock<1>
    %b2 = qecl.extract_block %r2[%i1] : !qecl.hyperreg<3 x 1> -> !qecl.codeblock<1>

    // CHECK: return [[b2]]
    return %b2 : !qecl.codeblock<1>
}

// -----

// CHECK-LABEL: test_extract_const_idx_fold
func.func @test_extract_const_idx_fold() -> !qecl.codeblock<1> {
    // CHECK-NOT: arith.constant
    // CHECK: [[hreg:%.+]] = qecl.alloc
    // CHECK: qecl.extract_block [[hreg]][ 0]
    %0 = arith.constant 0 : index
    %r = qecl.alloc() : !qecl.hyperreg<3 x 1>
    %b = qecl.extract_block %r[%0] : !qecl.hyperreg<3 x 1> -> !qecl.codeblock<1>

    return %b : !qecl.codeblock<1>
}

// -----

// CHECK-LABEL: test_insert_const_idx_fold
func.func @test_insert_const_idx_fold(%b : !qecl.codeblock<1>) -> !qecl.hyperreg<3 x 1>{
    // CHECK-NOT: arith.constant
    // CHECK: [[hreg:%.+]] = qecl.alloc
    // CHECK: qecl.insert_block [[hreg]][ 0]
    %0 = arith.constant 0 : index
    %r0 = qecl.alloc() : !qecl.hyperreg<3 x 1>
    %r1 = qecl.insert_block %r0[%0], %b : !qecl.hyperreg<3 x 1>, !qecl.codeblock<1>

    return %r1 : !qecl.hyperreg<3 x 1>
}
