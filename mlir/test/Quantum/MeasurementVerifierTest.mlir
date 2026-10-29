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

func.func @mixed_modes(%q: !quantum.bit, %r: !quantum.reg, %w: tensor<1xi64>) {
  // expected-error@+1 {{must use either qubits or registers, but not both}}
  %m, %oq, %or = "quantum.measurement"(%q, %r, %w) <{op_name = "Mixed", operandSegmentSizes = array<i32: 0, 1, 1, 1>, resultSegmentSizes = array<i32: 1, 1, 1>}> : (!quantum.bit, !quantum.reg, tensor<1xi64>) -> (i1, !quantum.bit, !quantum.reg)
  return
}

// -----

func.func @indices_without_register(%w: tensor<1xi64>) {
  // expected-error@+1 {{index arrays require a register}}
  %m = "quantum.measurement"(%w) <{op_name = "Indices", operandSegmentSizes = array<i32: 0, 0, 0, 1>, resultSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<1xi64>) -> i1
  return
}

// -----

func.func @missing_qubit_result(%q: !quantum.bit) {
  // expected-error@+1 {{number of input and output qubits must be the same}}
  %m = quantum.measurement "Missing"() qubits(%q) : i1
  return
}

// -----

func.func @missing_register_result(%r: !quantum.reg, %w: tensor<1xi64>) {
  // expected-error@+1 {{in_qreg and out_qreg must either both be present or absent}}
  %m = quantum.measurement "Missing"() quregs(%r) indices(%w: tensor<1xi64>) : i1
  return
}

// -----

func.func @quantum_param(%q: !quantum.bit) {
  // expected-error@+1 {{params must have classical types}}
  %m = quantum.measurement "BadParam"(%q: !quantum.bit) qubits() : i1
  return
}

// -----

func.func @nested_quantum_param(%q: tuple<!qref.bit>) {
  // expected-error@+1 {{params must have classical types}}
  %m = quantum.measurement "BadParam"(%q: tuple<!qref.bit>) qubits() : i1
  return
}

// -----

func.func @quantum_classical_result() {
  // expected-error@+1 {{mres must have classical types}}
  %m = "quantum.measurement"() <{op_name = "BadResult", operandSegmentSizes = array<i32: 0, 0, 0, 0>, resultSegmentSizes = array<i32: 1, 0, 0>}> : () -> !quantum.bit
  return
}

// -----

func.func @observable_result() {
  // expected-error@+1 {{mres must have classical types}}
  %m = quantum.measurement "BadResult"() qubits() : !quantum.obs
  return
}

// -----

func.func @result_order(%q: !quantum.bit) {
  // expected-error@+1 {{expected classical results followed by qubits or a register}}
  %out, %m = quantum.measurement "Order"() qubits(%q) : !quantum.bit, i1
  return
}

// -----

func.func @static_data_type() {
  // expected-error@+1 {{Invalid attribute `static_data` in property conversion}}
  %m = "quantum.measurement"() <{op_name = "BadData", static_data = "XY", operandSegmentSizes = array<i32: 0, 0, 0, 0>, resultSegmentSizes = array<i32: 1, 0, 0>}> : () -> i1
  return
}

// -----

func.func @adjoint(%q: !quantum.bit) {
  // expected-error@+1 {{quantum measurements are not allowed in the adjoint regions}}
  %out = quantum.adjoint (%q) : !quantum.bit {
  ^bb0(%arg: !quantum.bit):
    %m, %oq = quantum.measurement "Measure"() qubits(%arg) : i1, !quantum.bit
    quantum.yield %oq : !quantum.bit
  }
  return
}

// -----

func.func @ctrl(%q: !quantum.bit, %c: !quantum.bit, %v: i1) {
  // expected-error@+1 {{quantum measurements are not allowed in the ctrl regions}}
  %oc, %oq = quantum.ctrl(%c) ctrlvals(%v) (%q) : !quantum.bit -> !quantum.bit {
  ^bb0(%arg: !quantum.bit):
    %m, %out = quantum.measurement "Measure"() qubits(%arg) : i1, !quantum.bit
    quantum.yield %out : !quantum.bit
  }
  return
}
