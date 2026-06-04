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

// Test conversion to reference semantics quantum dialect for flat circuits.

// RUN: quantum-opt --convert-to-reference-semantics --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: test_static_alloc
func.func @test_static_alloc() attributes {quantum.node} {
    // CHECK: [[qreg:%.+]] = qref.alloc( 2) : !qref.reg<2>
    // CHECK-NOT: quantum.alloc
    %0 = quantum.alloc( 2) : !quantum.reg

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<2>
    quantum.dealloc %0 : !quantum.reg

    return
}

// -----

// CHECK-LABEL: test_dynamic_alloc
func.func @test_dynamic_alloc(%arg0: i64) attributes {quantum.node} {
    // CHECK: [[qreg:%.+]] = qref.alloc(%arg0) : !qref.reg<?>
    // CHECK-NOT: quantum.alloc
    %0 = quantum.alloc(%arg0) : !quantum.reg

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<?>
    quantum.dealloc %0 : !quantum.reg

    return
}
