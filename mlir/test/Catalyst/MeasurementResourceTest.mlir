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

// RUN: quantum-opt --resource-analysis=output-json %s -o /dev/null | FileCheck %s

// CHECK-LABEL: "reference_measurement"
// CHECK: "measurement_processes": {}
// CHECK: "quantum_operations"
// CHECK: "1"
// CHECK: "MeasureInBasis": 1
func.func @reference_measurement(%q: !qref.bit, %angle: f64) -> i1 {
  %m = qref.measurement "MeasureInBasis"(%angle: f64) qubits(%q) static_data = {plane = "XY"} : i1
  return %m : i1
}

// CHECK-LABEL: "value_measurement"
// CHECK: "measurement_processes": {}
// CHECK: "quantum_operations"
// CHECK: "2"
// CHECK: "MeasurePair": 1
func.func @value_measurement(%q0: !quantum.bit, %q1: !quantum.bit) -> (i1, !quantum.bit, !quantum.bit) {
  %m, %out:2 = quantum.measurement "MeasurePair"() qubits(%q0, %q1) : i1, !quantum.bit, !quantum.bit
  return %m, %out#0, %out#1 : i1, !quantum.bit, !quantum.bit
}
