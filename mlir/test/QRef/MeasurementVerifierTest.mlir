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

// RUN: quantum-opt --split-input-file --verify-diagnostics %s

func.func @mixed_modes(%q: !qref.bit, %r: !qref.reg<2>, %w: tensor<1xi64>) {
  // expected-error@+1 {{must use either qubits or registers, but not both}}
  %m = "qref.measurement"(%q, %r, %w) <{op_name = "Mixed", operandSegmentSizes = array<i32: 0, 1, 1, 1>}> : (!qref.bit, !qref.reg<2>, tensor<1xi64>) -> i1
  return
}

// -----

func.func @indices_without_register(%w: tensor<1xi64>) {
  // expected-error@+1 {{index arrays require a register}}
  %m = "qref.measurement"(%w) <{op_name = "Indices", operandSegmentSizes = array<i32: 0, 0, 0, 1>}> : (tensor<1xi64>) -> i1
  return
}

// -----

func.func @quantum_param(%q: !qref.bit) {
  // expected-error@+1 {{params must have classical types}}
  %m = qref.measurement "BadParam"(%q: !qref.bit) qubits() : i1
  return
}

// -----

func.func @observable_param(%obs: !quantum.obs) {
  // expected-error@+1 {{params must have classical types}}
  %m = qref.measurement "BadParam"(%obs: !quantum.obs) qubits() : i1
  return
}

// -----

func.func @quantum_classical_result() {
  // expected-error@+1 {{mres must have classical types}}
  %m = qref.measurement "BadResult"() qubits() : !qref.reg<2>
  return
}

// -----

func.func @nested_quantum_result() {
  // expected-error@+1 {{mres must have classical types}}
  %m = qref.measurement "BadResult"() qubits() : tuple<!quantum.bit>
  return
}

// -----

func.func @wrong_index_type(%r: !qref.reg<2>, %w: tensor<1xi32>) {
  // expected-error@+1 {{operand #1 must be variadic of 1D tensor of 64-bit signless integer values}}
  %m = qref.measurement "BadIndices"() quregs(%r: !qref.reg<2>) indices(%w: tensor<1xi32>) : i1
  return
}

// -----

func.func @static_data_type() {
  // expected-error@+1 {{Invalid attribute `static_data` in property conversion}}
  %m = "qref.measurement"() <{op_name = "BadData", static_data = "XY", operandSegmentSizes = array<i32: 0, 0, 0, 0>}> : () -> i1
  return
}

// -----

func.func @adjoint(%q: !qref.bit) {
  // expected-error@+1 {{quantum measurements are not allowed in the adjoint regions}}
  qref.adjoint {
    %m = qref.measurement "Measure"() qubits(%q) : i1
  }
  return
}

// -----

func.func @ctrl(%q: !qref.bit, %c: !qref.bit, %v: i1) {
  // expected-error@+1 {{quantum measurements are not allowed in the ctrl regions}}
  qref.ctrl(%c) ctrlvals(%v) {
    %m = qref.measurement "Measure"() qubits(%q) : i1
  }
  return
}
