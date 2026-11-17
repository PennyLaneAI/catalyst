// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --convert-quantum-to-llvm --split-input-file --verify-diagnostics %s

func.func @other_measurement(%q: !quantum.bit) -> (i1, !quantum.bit) {
  // expected-error @+1 {{failed to legalize operation 'quantum.measurement'}}
  %m, %out = quantum.measurement "OtherMeasurement"() qubits(%q)
      static_data = {pauli_word = "Z"} : i1, !quantum.bit
  return %m, %out : i1, !quantum.bit
}

// -----

func.func @postselection(%q: !quantum.bit) -> (i1, !quantum.bit) {
  // expected-error @+1 {{failed to legalize operation 'quantum.measurement'}}
  %m, %out = quantum.measurement "PauliMeasure"() qubits(%q)
      static_data = {pauli_word = "Z", postselect = 0 : i64} : i1, !quantum.bit
  return %m, %out : i1, !quantum.bit
}

// -----

func.func @register_measurement(%reg: !quantum.reg) -> (i1, !quantum.reg) {
  // expected-error @+1 {{failed to legalize operation 'quantum.measurement'}}
  %m, %out = quantum.measurement "PauliMeasure"() quregs(%reg)
      indices() static_data = {pauli_word = "Z"}
      : i1, !quantum.reg
  return %m, %out : i1, !quantum.reg
}

// -----

func.func @parameters(%q: !quantum.bit, %angle: f64) -> (i1, !quantum.bit) {
  // expected-error @+1 {{failed to legalize operation 'quantum.measurement'}}
  %m, %out = quantum.measurement "PauliMeasure"(%angle: f64) qubits(%q)
      static_data = {pauli_word = "Z"} : i1, !quantum.bit
  return %m, %out : i1, !quantum.bit
}

// -----

func.func @invalid_word(%q: !quantum.bit) -> (i1, !quantum.bit) {
  // expected-error @+1 {{failed to legalize operation 'quantum.measurement'}}
  %m, %out = quantum.measurement "PauliMeasure"() qubits(%q)
      static_data = {pauli_word = "H"} : i1, !quantum.bit
  return %m, %out : i1, !quantum.bit
}

// -----

func.func @word_length_mismatch(%q: !quantum.bit) -> (i1, !quantum.bit) {
  // expected-error @+1 {{failed to legalize operation 'quantum.measurement'}}
  %m, %out = quantum.measurement "PauliMeasure"() qubits(%q)
      static_data = {pauli_word = "ZZ"} : i1, !quantum.bit
  return %m, %out : i1, !quantum.bit
}

// -----

func.func @invalid_result(%q: !quantum.bit) -> (i64, !quantum.bit) {
  // expected-error @+1 {{failed to legalize operation 'quantum.measurement'}}
  %m, %out = quantum.measurement "PauliMeasure"() qubits(%q)
      static_data = {pauli_word = "Z"} : i64, !quantum.bit
  return %m, %out : i64, !quantum.bit
}
