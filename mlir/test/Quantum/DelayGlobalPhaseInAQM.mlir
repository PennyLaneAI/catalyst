// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --delay-global-phase-in-AQM --split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: test_basic_case
func.func @test_basic_case(%phase: f64, %q: !quantum.bit, %shots: i64) {
    quantum.device shots(%shots) ["", "", ""] {auto_qubit_management}

    // CHECK: quantum.custom "Hadamard"
    // CHECK: quantum.gphase(%arg0) :
    quantum.gphase(%phase) :
    %2 = quantum.custom "Hadamard"() %q : !quantum.bit

    return
}

// -----
// CHECK-LABEL: test_sandwhiched
func.func @test_sandwhiched(%phase: f64, %q: !quantum.bit, %shots: i64) {
    quantum.device shots(%shots) ["", "", ""] {auto_qubit_management}

    // CHECK: quantum.custom "Hadamard"
    // CHECK: [[phase:%.+]] = arith.addf %arg0, %arg0 : f64
    // CHECK: quantum.gphase([[phase]]) :
    quantum.gphase(%phase) :
    %2 = quantum.custom "Hadamard"() %q : !quantum.bit
    quantum.gphase(%phase) :

    return
}

// -----
// CHECK-LABEL: test_adjoint
func.func @test_adjoint(%phase: f64, %q: !quantum.bit, %shots: i64) {
    quantum.device shots(%shots) ["", "", ""] {auto_qubit_management}

    // CHECK: quantum.custom "Hadamard"
    // CHECK: [[adj:%.+]] = arith.negf %arg0 : f64
    // CHECK: [[phase:%.+]] = arith.addf [[adj]], %arg0 : f64
    // CHECK: quantum.gphase([[phase]]) :
    quantum.gphase(%phase) {adjoint} :
    quantum.gphase(%phase) :
    %2 = quantum.custom "Hadamard"() %q : !quantum.bit

    return
}

// -----
// CHECK-LABEL: test_not_aqm
func.func @test_not_aqm(%phase: f64, %q: !quantum.bit, %shots: i64) {
    quantum.device shots(%shots) ["", "", ""]

    // CHECK: quantum.gphase
    // CHECK: quantum.custom "Hadamard"
    quantum.gphase(%phase) :
    %2 = quantum.custom "Hadamard"() %q : !quantum.bit

    return
}
