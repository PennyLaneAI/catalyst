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

// RUN: quantum-opt --pass-pipeline='builtin.module(decompose-lowering{target-rules=my_X_decomp,my_Z_decomp})' --split-input-file -verify-diagnostics %s | FileCheck %s --check-prefixes=ALL,CALL
// RUN: quantum-opt --pass-pipeline='builtin.module(decompose-lowering{target-rules=my_X_decomp,my_Z_decomp inline-rule-body})' --split-input-file -verify-diagnostics %s | FileCheck %s --check-prefixes=ALL,INLINE

// Test that decompose-lowering only applies the rules requested by the `target-rules` option when present

// ALL: func.func private @my_X_decomp
func.func private @my_X_decomp(%q: !quantum.bit) -> !quantum.bit attributes {target_gate="X"} {
    %angle = arith.constant 1.57 : f64
    %out = quantum.custom "RX"(%angle) %q : !quantum.bit
    return %out : !quantum.bit
}

// ALL: func.func private @my_Y_decomp
func.func private @my_Y_decomp(%q: !quantum.bit) -> !quantum.bit attributes {target_gate="Y"} {
    %angle = arith.constant 1.57 : f64
    %out = quantum.custom "RY"(%angle) %q : !quantum.bit
    return %out : !quantum.bit
}

// ALL: func.func private @my_Z_decomp
func.func private @my_Z_decomp(%q: !quantum.bit) -> !quantum.bit attributes {target_gate="Z"} {
    %angle = arith.constant 1.57 : f64
    %out = quantum.custom "RZ"(%angle) %q : !quantum.bit
    return %out : !quantum.bit
}

func.func  @main_circuit() attributes {quantum.node} {
    // ALL: [[q:%.+]] = quantum.alloc_qb

    // CALL: [[x_out:%.+]] = call @my_X_decomp_1([[q]])
    // INLINE: [[x_out:%.+]] = quantum.custom "RX"(%{{.+}}) [[q]]

    // ALL: [[y_out:%.+]] = quantum.custom "Y"() [[x_out]]

    // CALL: [[z_out:%.+]] = call @my_Z_decomp_0([[y_out]])
    // INLINE: [[z_out:%.+]] = quantum.custom "RZ"(%{{.+}}) [[y_out]]

    // ALL: quantum.dealloc_qb [[z_out]]
    %0 = quantum.alloc_qb : !quantum.bit
    %1 = quantum.custom "X"() %0 : !quantum.bit
    %2 = quantum.custom "Y"() %1 : !quantum.bit
    %3 = quantum.custom "Z"() %2 : !quantum.bit
    quantum.dealloc_qb %3 : !quantum.bit
    return
}

// CALL: func.func private @my_Z_decomp_0
// CALL:   quantum.custom "RZ"
// CALL: func.func private @my_X_decomp_1
// CALL:   quantum.custom "RX"
