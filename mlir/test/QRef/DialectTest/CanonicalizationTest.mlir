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

// RUN: quantum-opt --canonicalize --cse %s | FileCheck %s

// CHECK-LABEL: test_alloc_dce
func.func @test_alloc_dce() {
    // CHECK-NOT: qref.alloc
    %r = qref.alloc(5) : !qref.reg<5>
    return
}

// -----

// CHECK-LABEL: test_alloc_no_cse
func.func @test_alloc_no_cse() -> (!qref.reg<4>, !qref.reg<4>){
    // CHECK: qref.alloc
    // CHECK-NEXT: qref.alloc
    %r1 = qref.alloc(4) : !qref.reg<4>
    %r2 = qref.alloc(4) : !qref.reg<4>
    return %r1, %r2 : !qref.reg<4>, !qref.reg<4>
}

// -----

// CHECK-LABEL: test_alloc_dealloc_fold
func.func @test_alloc_dealloc_fold() {
    // CHECK-NOT: qref.alloc
    // CHECK-NOT: qref.dealloc
    %r = qref.alloc(3) : !qref.reg<3>
    qref.dealloc %r : !qref.reg<3>
    return
}

// -----

// CHECK-LABEL: test_alloc_dealloc_no_fold
func.func @test_alloc_dealloc_no_fold() -> !qref.bit {
    // CHECK: qref.alloc
    // CHECK: qref.dealloc
    %r = qref.alloc(3) : !qref.reg<3>
    %q = qref.get %r[0] : !qref.reg<3> -> !qref.bit
    qref.dealloc %r : !qref.reg<3>
    return %q : !qref.bit<3>
}
