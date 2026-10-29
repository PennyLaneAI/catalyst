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

// RUN: quantum-opt --convert-to-reference-semantics %s | FileCheck %s
// RUN: quantum-opt --convert-to-reference-semantics --mlir-print-op-generic %s | quantum-opt -o /dev/null

// CHECK-LABEL: @pair
func.func @pair(%angle: f64) -> (i1, f64) attributes {quantum.node} {
  %q0 = quantum.alloc_qb : !quantum.bit
  %q1 = quantum.alloc_qb : !quantum.bit
  // CHECK: %[[M:[^ :]+]]:2 = qref.measurement "Pair"(%{{.*}}: f64) qubits(%[[Q0:[^,]+]], %[[Q1:[^)]+]]) static_data = {plane = "XY"} {tag = "keep"} : i1, f64
  %m, %s, %out:2 = quantum.measurement "Pair"(%angle: f64) qubits(%q0, %q1)
      static_data = {plane = "XY"} {tag = "keep"} : i1, f64, !quantum.bit, !quantum.bit
  // CHECK: qref.custom "PauliX"() %[[Q0]]
  %x = quantum.custom "PauliX"() %out#0 : !quantum.bit
  // CHECK: qref.dealloc_qb %[[Q0]]
  quantum.dealloc_qb %x : !quantum.bit
  // CHECK: qref.dealloc_qb %[[Q1]]
  quantum.dealloc_qb %out#1 : !quantum.bit
  // CHECK: return %[[M]]#0, %[[M]]#1 : i1, f64
  return %m, %s : i1, f64
}

// CHECK-LABEL: @register
func.func @register(%wires: tensor<?xi64>) -> tensor<?xi1> attributes {quantum.node} {
  // CHECK: %[[R:[^ ]+]] = qref.alloc
  %reg = quantum.alloc(4) : !quantum.reg
  // CHECK: %[[M:[^ ]+]] = qref.measurement "Many"() quregs(%[[R]]: !qref.reg<4>) indices(%{{.*}}: tensor<?xi64>) static_data = {reset = true} : tensor<?xi1>
  %m, %out = quantum.measurement "Many"() quregs(%reg) indices(%wires: tensor<?xi64>)
      static_data = {reset = true} : tensor<?xi1>, !quantum.reg
  // CHECK: qref.measurement "Discard"() quregs(%[[R]]: !qref.reg<4>) indices(%{{.*}}: tensor<?xi64>) : ()
  %out2 = quantum.measurement "Discard"() quregs(%out) indices(%wires: tensor<?xi64>) : !quantum.reg
  // CHECK: qref.dealloc %[[R]]
  quantum.dealloc %out2 : !quantum.reg
  // CHECK: return %[[M]]
  return %m : tensor<?xi1>
}

// CHECK-LABEL: @branch
func.func @branch(%cond: i1, %wires: tensor<1xi64>) -> i1 attributes {quantum.node} {
  // CHECK: %[[R:[^ ]+]] = qref.alloc
  %reg = quantum.alloc(2) : !quantum.reg
  // CHECK: %[[BR:[^ ]+]] = scf.if %{{.*}} -> (i1)
  %m, %out = scf.if %cond -> (i1, !quantum.reg) {
    // CHECK: %[[A:[^ ]+]] = qref.measurement "A"() quregs(%[[R]]: !qref.reg<2>)
    %a, %ra = quantum.measurement "A"() quregs(%reg) indices(%wires: tensor<1xi64>) : i1, !quantum.reg
    // CHECK: scf.yield %[[A]] : i1
    scf.yield %a, %ra : i1, !quantum.reg
  } else {
    // CHECK: %[[B:[^ ]+]] = qref.measurement "B"() quregs(%[[R]]: !qref.reg<2>)
    %b, %rb = quantum.measurement "B"() quregs(%reg) indices(%wires: tensor<1xi64>) : i1, !quantum.reg
    // CHECK: scf.yield %[[B]] : i1
    scf.yield %b, %rb : i1, !quantum.reg
  }
  // CHECK: qref.dealloc %[[R]]
  quantum.dealloc %out : !quantum.reg
  // CHECK: return %[[BR]]
  return %m : i1
}
