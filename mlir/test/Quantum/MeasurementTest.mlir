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

// RUN: quantum-opt %s | quantum-opt | FileCheck %s
// RUN: quantum-opt --mlir-print-op-generic %s | quantum-opt | FileCheck %s
// RUN: quantum-opt --canonicalize %s | FileCheck %s
// RUN: quantum-opt --split-to-single-terms %s | FileCheck %s

// CHECK-LABEL: @qubits
func.func @qubits(%angle: f64, %data: tensor<?xf64>, %q0: !quantum.bit, %q1: !quantum.bit)
    -> (i1, tensor<2xf64>, !quantum.bit, !quantum.bit) {
  // CHECK: quantum.measurement "MeasurePair"(%{{.*}}: f64, %{{.*}}: tensor<?xf64>) qubits(%{{.*}}, %{{.*}}) static_data = {plane = "XY", reset = true} param_map = {angle = [0], data = [1]} qubit_map = {wires = [0, 1]} {tag = "preserved"} : (i1, tensor<2xf64>, !quantum.bit, !quantum.bit)
  %m, %data_out, %out:2 = quantum.measurement "MeasurePair"(%angle: f64, %data: tensor<?xf64>)
      qubits(%q0, %q1) static_data = {plane = "XY", reset = true}
      param_map = {angle = [0], data = [1]} qubit_map = {wires = [0, 1]} {tag = "preserved"}
      : (i1, tensor<2xf64>, !quantum.bit, !quantum.bit)
  return %m, %data_out, %out#0, %out#1 : i1, tensor<2xf64>, !quantum.bit, !quantum.bit
}

// CHECK-LABEL: @register
func.func @register(%reg: !quantum.reg, %wires: tensor<?xi64>, %other: tensor<1xi64>)
    -> (tensor<?xi1>, !quantum.reg) {
  // CHECK: quantum.measurement "MeasureMany"() quregs(%{{.*}}) indices(%{{.*}}: tensor<?xi64>, %{{.*}}: tensor<1xi64>) qubit_map = {other = [1], wires = [0]} : (tensor<?xi1>, !quantum.reg)
  %m, %out = quantum.measurement "MeasureMany"()
      quregs(%reg) indices(%wires: tensor<?xi64>, %other: tensor<1xi64>)
      qubit_map = {wires = [0], other = [1]}
      : (tensor<?xi1>, !quantum.reg)
  return %m, %out : tensor<?xi1>, !quantum.reg
}

// CHECK-LABEL: @unused_results
func.func @unused_results(%q: !quantum.bit) {
  // CHECK: quantum.measurement "Unused"() qubits(%{{.*}}) : (i1, !quantum.bit)
  %m, %out = quantum.measurement "Unused"() qubits(%q) : (i1, !quantum.bit)
  // CHECK: quantum.measurement "NoClassicalResults"() qubits(%{{.*}}) : (!quantum.bit)
  %out2 = quantum.measurement "NoClassicalResults"() qubits(%out) : (!quantum.bit)
  return
}

// CHECK-LABEL: @escaped_name
func.func @escaped_name(%q: !quantum.bit) -> (i64, !quantum.bit) {
  // CHECK: quantum.measurement "Quoted \22name\22"() qubits(%{{.*}}) : (i64, !quantum.bit)
  %m, %out = quantum.measurement "Quoted \"name\""() qubits(%q) : (i64, !quantum.bit)
  return %m, %out : i64, !quantum.bit
}

// CHECK-LABEL: @quoted_map_keys
func.func @quoted_map_keys(%angle: f64, %q0: !quantum.bit, %q1: !quantum.bit)
    -> (i1, !quantum.bit, !quantum.bit) {
  // CHECK: quantum.measurement "QuotedKeys"(%{{.*}}: f64) qubits(%{{.*}}, %{{.*}}) param_map = {"my angle" = [0]} qubit_map = {"0" = [0], plain = [1]} : (i1, !quantum.bit, !quantum.bit)
  %m, %out:2 = quantum.measurement "QuotedKeys"(%angle: f64) qubits(%q0, %q1)
      param_map = {"my angle" = [0]} qubit_map = {"0" = [0], "plain" = [1]}
      : (i1, !quantum.bit, !quantum.bit)
  return %m, %out#0, %out#1 : i1, !quantum.bit, !quantum.bit
}
