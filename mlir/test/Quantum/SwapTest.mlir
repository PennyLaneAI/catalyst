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

// RUN: quantum-opt --split-input-file %s | quantum-opt | FileCheck %s

// CHECK-LABEL: @test_swap_static
func.func @test_swap_static(%q_extra : !quantum.bit) -> (!quantum.reg, !quantum.bit) {
    %r = quantum.alloc(4) : !quantum.reg
    // CHECK: %{{.*}}, %{{.*}} = quantum.swap %{{.*}}[ 1], %{{.*}} : !quantum.reg, !quantum.bit
    %r1, %q_displaced = quantum.swap %r[1], %q_extra : !quantum.reg, !quantum.bit
    return %r1, %q_displaced : !quantum.reg, !quantum.bit
}

// -----

// CHECK-LABEL: @test_swap_dynamic
func.func @test_swap_dynamic(%q_extra : !quantum.bit, %i : i64) -> (!quantum.reg, !quantum.bit) {
    %r = quantum.alloc(4) : !quantum.reg
    // CHECK: %{{.*}}, %{{.*}} = quantum.swap %{{.*}}[%{{.*}}], %{{.*}} : !quantum.reg, !quantum.bit
    %r1, %q_displaced = quantum.swap %r[%i], %q_extra : !quantum.reg, !quantum.bit
    return %r1, %q_displaced : !quantum.reg, !quantum.bit
}
