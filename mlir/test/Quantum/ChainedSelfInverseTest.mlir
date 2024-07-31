// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --pass-pipeline="builtin.module(remove-chained-self-inverse{func-name=test_chained_self_inverse})" --split-input-file -verify-diagnostics %s | FileCheck %s

// test chained Hadamard
// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse() -> !quantum.bit {
    // CHECK: quantum.alloc
    // CHECK: quantum.extract
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // CHECK-NOT: quantum.custom
    %2 = quantum.custom "Hadamard"() %1 : !quantum.bit
    %3 = quantum.custom "Hadamard"() %2 : !quantum.bit
    return %3 : !quantum.bit
}

// -----

// test chained Hadamard from block arg
// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse(%arg: !quantum.bit) -> !quantum.bit {
    // CHECK-NOT: quantum.custom
    %0 = quantum.custom "Hadamard"() %arg : !quantum.bit
    %1 = quantum.custom "Hadamard"() %0 : !quantum.bit
    return %1 : !quantum.bit
}
