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

// RUN: quantum-opt --to-ppr %s | FileCheck %s

// CHECK-LABEL: @pauli_measurement
func.func @pauli_measurement(%q0: !quantum.bit, %q1: !quantum.bit)
    -> (i1, !quantum.bit, !quantum.bit) {
  // CHECK: [[M:%.+]], [[Q:%.+]]:2 = pbc.ppm ["X", "Z"]
  %m, %out0, %out1 = quantum.measurement "PauliMeasure"() qubits(%q0, %q1)
      static_data = {pauli_word = "XZ", postselect = none}
      : i1, !quantum.bit, !quantum.bit
  // CHECK: return [[M]], [[Q]]#0, [[Q]]#1
  return %m, %out0, %out1 : i1, !quantum.bit, !quantum.bit
}

// CHECK-LABEL: @discarded_measurement_result
func.func @discarded_measurement_result(%q0: !quantum.bit) -> !quantum.bit {
  // CHECK: {{%.+}}, [[Q:%.+]] = pbc.ppm ["Y"]
  %out = quantum.measurement "PauliMeasure"() qubits(%q0)
      static_data = {pauli_word = "Y", meas_uid = "direct"}
      : !quantum.bit
  // CHECK: return [[Q]]
  return %out : !quantum.bit
}

// CHECK-LABEL: @other_measurements
func.func @other_measurements(%q0: !quantum.bit) -> (i1, !quantum.bit) {
  // CHECK: quantum.measurement "OtherMeasurement"()
  %m, %out = quantum.measurement "OtherMeasurement"() qubits(%q0)
      : i1, !quantum.bit
  return %m, %out : i1, !quantum.bit
}
