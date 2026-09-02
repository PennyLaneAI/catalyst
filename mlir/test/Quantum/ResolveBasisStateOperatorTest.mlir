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

// RUN: quantum-opt --resolve-basis-state-operator --verify-diagnostics %s | FileCheck %s


// CHECK-LABEL: test_single_qubit_set_state
func.func @test_single_qubit_set_state(%arg0: !quantum.bit, %arg1: tensor<2xcomplex<f64>>) -> !quantum.bit {
    // CHECK: [[q:%.+]] = quantum.set_state(%arg1) %arg0 : (tensor<2xcomplex<f64>>, !quantum.bit) -> !quantum.bit
    // CHECK: return [[q]] : !quantum.bit
    %0 = quantum.operator "StatePrep"(%arg1: tensor<2xcomplex<f64>>) qubits(%arg0)
      static_data = {}
      param_map = {state = [0]} qubit_map = {wires = [0]}
    return %0 : !quantum.bit
}

// -----

// CHECK-LABEL: test_multiple_qubits_set_state
func.func @test_multiple_qubits_set_state(%arg0: !quantum.bit, %arg1: !quantum.bit, %arg2: tensor<4xcomplex<f64>>) -> (!quantum.bit, !quantum.bit) {
    // CHECK: [[q:%.+]]:2 = quantum.set_state(%arg2) %arg0, %arg1 : (tensor<4xcomplex<f64>>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
    // CHECK: return [[q]]#0, [[q]]#1 : !quantum.bit, !quantum.bit
    %0:2 = quantum.operator "StatePrep"(%arg2: tensor<4xcomplex<f64>>) qubits(%arg0, %arg1)
      static_data = {}
      param_map = {state = [0]} qubit_map = {wires = [0, 1]}
    return %0#0, %0#1 : !quantum.bit, !quantum.bit
}


// -----

// CHECK-LABEL: test_single_qubit_basis_state
func.func @test_single_qubit_basis_state(%arg0: !quantum.bit, %arg1: tensor<1xi1>) -> !quantum.bit {
    // CHECK: [[q:%.+]] = quantum.set_basis_state(%arg1) %arg0 : (tensor<1xi1>, !quantum.bit) -> !quantum.bit
    // CHECK: return [[q]] : !quantum.bit
    %0 = quantum.operator "BasisState"(%arg1: tensor<1xi1>) qubits(%arg0)
      static_data = {}
      param_map = {state = [0]} qubit_map = {wires = [0]}
    return %0 : !quantum.bit
}

// -----

// CHECK-LABEL: test_multiple_qubits_basis_state
func.func @test_multiple_qubits_basis_state(%arg0: !quantum.bit, %arg1: !quantum.bit, %arg2: tensor<2xi1>) -> (!quantum.bit, !quantum.bit) {
    // CHECK: [[q:%.+]]:2 = quantum.set_basis_state(%arg2) %arg0, %arg1 : (tensor<2xi1>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
    // CHECK: return [[q]]#0, [[q]]#1 : !quantum.bit, !quantum.bit
    %0:2 = quantum.operator "BasisState"(%arg2: tensor<2xi1>) qubits(%arg0, %arg1)
      static_data = {}
      param_map = {state = [0]} qubit_map = {wires = [0, 1]}
    return %0#0, %0#1 : !quantum.bit, !quantum.bit
}


// -----

// CHECK-LABEL: test_non_BasisState_ignored
func.func @test_non_BasisState_ignored(%arg0: !quantum.bit, %arg1: tensor<1xi1>) -> !quantum.bit {
    // CHECK: quantum.operator
    %0 = quantum.operator "gate"(%arg1: tensor<1xi1>) qubits(%arg0)
      static_data = {}
      param_map = {state = [0]} qubit_map = {wires = [0]}
    return %0 : !quantum.bit
}
