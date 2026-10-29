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

// CHECK-LABEL: @qubits
func.func @qubits(%angle: f64, %data: tensor<?xf64>, %q0: !qref.bit, %q1: !qref.bit)
    -> (i1, tensor<2xf64>) {
  // CHECK: qref.measurement "MeasurePair"(%{{.*}}: f64, %{{.*}}: tensor<?xf64>) qubits(%{{.*}}, %{{.*}}) static_data = {plane = "XY", reset = true} {tag = "preserved"} : i1, tensor<2xf64>
  %m, %data_out = qref.measurement "MeasurePair"(%angle: f64, %data: tensor<?xf64>)
      qubits(%q0, %q1) static_data = {plane = "XY", reset = true} {tag = "preserved"}
      : i1, tensor<2xf64>
  return %m, %data_out : i1, tensor<2xf64>
}

// CHECK-LABEL: @register
func.func @register(%reg: !qref.reg<4>, %wires: tensor<?xi64>, %other: tensor<1xi64>)
    -> tensor<?xi1> {
  // CHECK: qref.measurement "MeasureMany"() quregs(%{{.*}}: !qref.reg<4>) indices(%{{.*}}: tensor<?xi64>, %{{.*}}: tensor<1xi64>) : tensor<?xi1>
  %m = qref.measurement "MeasureMany"()
      quregs(%reg: !qref.reg<4>) indices(%wires: tensor<?xi64>, %other: tensor<1xi64>)
      : tensor<?xi1>
  return %m : tensor<?xi1>
}

// CHECK-LABEL: @dynamic_register
func.func @dynamic_register(%reg: !qref.reg<?>, %wires: tensor<?xi64>) {
  // CHECK: qref.measurement "Discard"() quregs(%{{.*}}: !qref.reg<?>) indices(%{{.*}}: tensor<?xi64>) : ()
  qref.measurement "Discard"() quregs(%reg: !qref.reg<?>) indices(%wires: tensor<?xi64>) : ()
  return
}

// CHECK-LABEL: @unused_results
func.func @unused_results(%q: !qref.bit) {
  // CHECK: qref.measurement "Unused"() qubits(%{{.*}}) : i1
  %m = qref.measurement "Unused"() qubits(%q) : i1
  // CHECK: qref.measurement "NoResults"() qubits() static_data = {} : ()
  qref.measurement "NoResults"() qubits() static_data = {} : ()
  return
}
