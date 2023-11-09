// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --buffer-deallocation --split-input-file %s | FileCheck %s

// CHECK-LABEL: @existing_deallocation
func.func @existing_deallocation() {
    // CHECK: [[reg:%.+]] = quantum.alloc
    %0 = quantum.alloc( 1) : !quantum.reg
    // CHECK: quantum.dealloc [[reg]]
    quantum.dealloc %0 : !quantum.reg
    return
}

// -----

// CHECK-LABEL: @missing_deallocation
func.func @missing_deallocation() {
    // CHECK: [[reg:%.+]] = quantum.alloc
    %0 = quantum.alloc( 1) : !quantum.reg
    // CHECK-NOT: quantum.dealloc [[reg]]
    return
}

// -----

// CHECK-LABEL: @returning_allocation
func.func @returning_allocation() -> !quantum.reg {
    // CHECK: [[reg:%.+]] = quantum.alloc
    %0 = quantum.alloc( 1) : !quantum.reg
    // CHECK-NEXT: return [[reg]]
    return %0 : !quantum.reg
}
