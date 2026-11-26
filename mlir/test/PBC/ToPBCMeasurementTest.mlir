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

// RUN: quantum-opt --to-pbc --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @pauli_measurement
// CHECK-SAME: ([[Q0:%.+]]: !quantum.bit, [[Q1:%.+]]: !quantum.bit)
func.func @pauli_measurement(%q0: !quantum.bit, %q1: !quantum.bit)
    -> (i1, !quantum.bit, !quantum.bit) {
  // CHECK: [[M:%.+]], [[Q:%.+]]:2 = pbc.ppm ["X", "Z"] [[Q0]], [[Q1]]
  %m, %out0, %out1 = quantum.measurement "PauliMeasure"() qubits(%q0, %q1)
      static_data = {pauli_word = "XZ", postselect = none}
      : (i1, !quantum.bit, !quantum.bit)
  // CHECK: return [[M]], [[Q]]#0, [[Q]]#1
  return %m, %out0, %out1 : i1, !quantum.bit, !quantum.bit
}

// CHECK-LABEL: @discarded_measurement_result
// CHECK-SAME: ([[Q0:%.+]]: !quantum.bit)
func.func @discarded_measurement_result(%q0: !quantum.bit) -> !quantum.bit {
  // CHECK: {{%.+}}, [[Q:%.+]] = pbc.ppm ["Y"] [[Q0]]
  %out = quantum.measurement "PauliMeasure"() qubits(%q0)
      static_data = {pauli_word = "Y", meas_uid = "direct"}
      : (!quantum.bit)
  // CHECK: return [[Q]]
  return %out : !quantum.bit
}

// -----

func.func @unsupported_measurement(%q0: !quantum.bit) -> (i1, !quantum.bit) {
  // expected-error @+2 {{failed to legalize operation 'quantum.measurement' that was explicitly marked illegal}}
  // expected-error @+1 {{'quantum.measurement' op measurement "OtherMeasurement" is not supported by PBC conversion, only "PauliMeasure" can be converted}}
  %m, %out = quantum.measurement "OtherMeasurement"() qubits(%q0)
      : (i1, !quantum.bit)
  return %m, %out : i1, !quantum.bit
}
