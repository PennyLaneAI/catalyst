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

// RUN: quantum-opt --pass-pipeline='builtin.module(decompose-lowering{target-rules=my_X_decomp,my_Z_decomp})' --split-input-file -verify-diagnostics %s | FileCheck %s

// Test that decompose-lowering only applies the rules requested by the `target-rules` option when present

module @test_module {
    // CHECK: func.func private @my_X_decomp
    func.func private @my_X_decomp(%q: !quantum.bit) -> !quantum.bit attributes {target_gate="X"} {
        %angle = arith.constant 1.57 : f64
        %out = quantum.custom "RX"(%angle) %q : !quantum.bit
        return %out : !quantum.bit
    }

    func.func private @my_Y_decomp(%q: !quantum.bit) -> !quantum.bit attributes {target_gate="Y"} {
        %angle = arith.constant 1.57 : f64
        %out = quantum.custom "RY"(%angle) %q : !quantum.bit
        return %out : !quantum.bit
    }

    // CHECK: func.func private @my_Z_decomp
    func.func private @my_Z_decomp(%q: !quantum.bit) -> !quantum.bit attributes {target_gate="Z"} {
        %angle = arith.constant 1.57 : f64
        %out = quantum.custom "RZ"(%angle) %q : !quantum.bit
        return %out : !quantum.bit
    }

    // CHECK: [[q_alloc:%.+]] = quantum.alloc_qb
    // CHECK: [[x_out:%.+]] = func.call @my_X_decomp([[q_alloc]])
    // CHECK: [[y_out:%.+]] = quantum.custom "Y"() [[x_out]]
    // CHECK: [[z_out:%.+]] = func.call @my_Z_decomp([[y_out]])
    // CHECK: quantum.dealloc_qb [[z_out]]
    %0 = quantum.alloc_qb : !quantum.bit
    %1 = quantum.custom "X"() %0 : !quantum.bit
    %2 = quantum.custom "Y"() %1 : !quantum.bit
    %3 = quantum.custom "Z"() %2 : !quantum.bit
    quantum.dealloc_qb %3 : !quantum.bit
}
