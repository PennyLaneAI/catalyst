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

// RUN: quantum-opt --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @pauli
func.func @pauli(%coeffs: tensor<2xf64>, %time: tensor<f64>) {
    // CHECK: [[h:%.+]] = quantum.abs_hamiltonian(%arg0 : tensor<2xf64>) {kind = "pauli", structure = {words = ["XX", "Z"]}} : !quantum.abs_hamiltonian
    %h = quantum.abs_hamiltonian(%coeffs : tensor<2xf64>) {kind = "pauli", structure = {words = ["XX", "Z"]}} : !quantum.abs_hamiltonian
    // CHECK: quantum.custom_h [[h]], %arg1 {m = 1 : i64, n = 50 : i64} : tensor<f64>
    quantum.custom_h %h, %time {n = 50 : i64, m = 1 : i64} : tensor<f64>
    func.return
}

// -----

// CHECK-LABEL: func.func @lcu
func.func @lcu(%coeffs: tensor<2xf64>, %time: tensor<f64>) {
    // CHECK: quantum.abs_hamiltonian(%arg0 : tensor<2xf64>) {kind = "lcu"
    %h = quantum.abs_hamiltonian(%coeffs : tensor<2xf64>) {kind = "lcu", structure = {terms = ["X(0)", "Z(0)"]}} : !quantum.abs_hamiltonian
    quantum.custom_h %h, %time {n = 50 : i64, m = 1 : i64} : tensor<f64>
    func.return
}

// -----

// CHECK-LABEL: func.func @cdf
func.func @cdf(%core: tensor<2x2x2x3x3xf64>, %leaf: tensor<2x2x3x3xf64>, %nuc: tensor<f64>, %time: tensor<f64>) {
    // CHECK: quantum.abs_hamiltonian(%arg0, %arg1, %arg2 : tensor<2x2x2x3x3xf64>, tensor<2x2x3x3xf64>, tensor<f64>) {kind = "cdf", structure = {L = 2 : i64, M = 2 : i64, N = 3 : i64}} : !quantum.abs_hamiltonian
    %h = quantum.abs_hamiltonian(%core, %leaf, %nuc : tensor<2x2x2x3x3xf64>, tensor<2x2x3x3xf64>, tensor<f64>) {kind = "cdf", structure = {L = 2 : i64, M = 2 : i64, N = 3 : i64}} : !quantum.abs_hamiltonian
    quantum.custom_h %h, %time {n = 50 : i64, m = 1 : i64} : tensor<f64>
    func.return
}
