// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --loop-boundary --split-input-file -verify-diagnostics %s | FileCheck %s

func.func @test(%q: !quantum.bit) -> !quantum.bit {
    %start = arith.constant 0 : index
    %stop = arith.constant 10 : index
    %step = arith.constant 1 : index
    %phi = arith.constant 0.1 : f64

    // CHECK: func @test([[arg:%.+]]: !quantum.bit) -> !quantum.bit {
    // CHECK: [[qubit_0:%.+]] = quantum.custom "H"() [[arg]] : !quantum.bit 

    // CHECK: [[qubit_1:%.+]] = scf.for {{.*}} iter_args([[qubit_2:%.+]] = [[qubit_0]]) -> (!quantum.bit) {
    %qq = scf.for %i = %start to %stop step %step iter_args(%q_0 = %q) -> (!quantum.bit) {
        %q_1 = quantum.custom "H"() %q_0 : !quantum.bit
        // CHECK: [[qubit_3:%.+]] = quantum.custom "RY"{{.*}} [[qubit_2]] : !quantum.bit
        %q_2 = quantum.custom "RY"(%phi) %q_1 : !quantum.bit
        %q_3 = quantum.custom "H"() %q_2 : !quantum.bit
        // CHECK-NEXT: scf.yield [[qubit_3]] : !quantum.bit
        scf.yield %q_3 : !quantum.bit
    }

    // CHECK: [[qubit_4:%.+]] = quantum.custom "H"() [[qubit_1]] : !quantum.bit
    // CHECK: return [[qubit_4]]
    func.return %qq : !quantum.bit
}