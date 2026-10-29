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

// RUN: quantum-opt --convert-to-value-semantics %s | FileCheck %s
// RUN: quantum-opt --convert-to-value-semantics --mlir-print-op-generic %s | quantum-opt -o /dev/null

// CHECK-LABEL: @pair
func.func @pair(%angle: f64) -> (i1, f64) attributes {quantum.node} {
  %q0 = qref.alloc_qb : !qref.bit
  %q1 = qref.alloc_qb : !qref.bit
  // CHECK: %[[M:[^ :]+]]:2, %[[Q:[^ :]+]]:2 = quantum.measurement "Pair"(%{{.*}}: f64) qubits(%{{.*}}, %{{.*}}) static_data = {plane = "XY"} {tag = "keep"} : i1, f64, !quantum.bit, !quantum.bit
  %m, %s = qref.measurement "Pair"(%angle: f64) qubits(%q0, %q1)
      static_data = {plane = "XY"} {tag = "keep"} : i1, f64
  // CHECK: %[[X:[^ ]+]] = quantum.custom "PauliX"() %[[Q]]#0
  qref.custom "PauliX"() %q0 : !qref.bit
  // CHECK: quantum.dealloc_qb %[[X]]
  qref.dealloc_qb %q0 : !qref.bit
  // CHECK: quantum.dealloc_qb %[[Q]]#1
  qref.dealloc_qb %q1 : !qref.bit
  // CHECK: return %[[M]]#0, %[[M]]#1 : i1, f64
  return %m, %s : i1, f64
}

// CHECK-LABEL: @extracted_qubits
func.func @extracted_qubits() -> i1 attributes {quantum.node} {
  %reg = qref.alloc(2) : !qref.reg<2>
  %q0 = qref.get %reg[0] : !qref.reg<2> -> !qref.bit
  %q1 = qref.get %reg[1] : !qref.reg<2> -> !qref.bit
  // CHECK: %[[Q0:[^ ]+]] = quantum.extract
  // CHECK: %[[Q1:[^ ]+]] = quantum.extract
  // CHECK: %[[M:[^ ,]+]], %[[Q:[^ :]+]]:2 = quantum.measurement "Parity"() qubits(%[[Q0]], %[[Q1]]) : i1, !quantum.bit, !quantum.bit
  %m = qref.measurement "Parity"() qubits(%q0, %q1) : i1
  // CHECK: quantum.insert %{{.*}}[{{ *}}0], %[[Q]]#0
  // CHECK: %[[R:[^ ]+]] = quantum.insert %{{.*}}[{{ *}}1], %[[Q]]#1
  // CHECK: quantum.dealloc %[[R]]
  qref.dealloc %reg : !qref.reg<2>
  // CHECK: return %[[M]]
  return %m : i1
}

// CHECK-LABEL: @register
func.func @register(%size: i64, %wires: tensor<?xi64>) -> tensor<?xi1> attributes {quantum.node} {
  // CHECK: %[[R:[^ ]+]] = quantum.alloc
  %reg = qref.alloc(%size) : !qref.reg<?>
  // CHECK: %[[M:[^ ,]+]], %[[R1:[^ ]+]] = quantum.measurement "Many"() quregs(%[[R]]) indices(%{{.*}}: tensor<?xi64>) static_data = {reset = true} : tensor<?xi1>, !quantum.reg
  %m = qref.measurement "Many"() quregs(%reg: !qref.reg<?>) indices(%wires: tensor<?xi64>)
      static_data = {reset = true} : tensor<?xi1>
  // CHECK: %[[R2:[^ ]+]] = quantum.measurement "Discard"() quregs(%[[R1]]) indices(%{{.*}}: tensor<?xi64>) : !quantum.reg
  qref.measurement "Discard"() quregs(%reg: !qref.reg<?>) indices(%wires: tensor<?xi64>) : ()
  // CHECK: quantum.dealloc %[[R2]]
  qref.dealloc %reg : !qref.reg<?>
  // CHECK: return %[[M]]
  return %m : tensor<?xi1>
}

// CHECK-LABEL: @branch
func.func @branch(%cond: i1) -> i1 attributes {quantum.node} {
  %q = qref.alloc_qb : !qref.bit
  // CHECK: %[[BR:[^ :]+]]:2 = scf.if %{{.*}} -> (i1, !quantum.bit)
  %m = scf.if %cond -> (i1) {
    // CHECK: %[[A:[^ ,]+]], %[[QA:[^ ]+]] = quantum.measurement "A"
    %a = qref.measurement "A"() qubits(%q) : i1
    // CHECK: scf.yield %[[A]], %[[QA]] : i1, !quantum.bit
    scf.yield %a : i1
  } else {
    // CHECK: %[[B:[^ ,]+]], %[[QB:[^ ]+]] = quantum.measurement "B"
    %b = qref.measurement "B"() qubits(%q) : i1
    // CHECK: scf.yield %[[B]], %[[QB]] : i1, !quantum.bit
    scf.yield %b : i1
  }
  // CHECK: quantum.dealloc_qb %[[BR]]#1
  qref.dealloc_qb %q : !qref.bit
  // CHECK: return %[[BR]]#0
  return %m : i1
}
