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

// RUN: xdsl-opt --passes combine-global-phases --split-input-file --verify-diagnostics %s | FileCheck %s


// Test combining without control flow
// CHECK-LABEL: test_combine_global_phase
func.func @test_combine_global_phase(%arg0: f64, %arg1: f64) {
    // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
    %0 = "test.op"() : () -> !quantum.bit
    // CHECK: [[phi_sum:%.+]] = arith.addf %arg1, %arg0 : f64
    // CHECK: quantum.gphase([[phi_sum]])
    quantum.gphase(%arg0) :
    quantum.gphase(%arg1) :
    // CHECK: [[q1:%.+]] = quantum.custom "PauliX"() [[q0]] : !quantum.bit
    %2 = quantum.custom "PauliX"() %0 : !quantum.bit
    return
}