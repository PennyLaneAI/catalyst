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

// RUN: quantum-opt --convert-to-value-semantics --split-input-file %s | FileCheck %s

// CHECK-LABEL: @dynamic_and_static_data
func.func @dynamic_and_static_data(%angle: f64) -> (i1, f64) attributes {quantum.node} {
  // CHECK: [[Q0:%.+]] = quantum.alloc_qb : !quantum.bit
  // CHECK: [[Q1:%.+]] = quantum.alloc_qb : !quantum.bit
  %q0 = qref.alloc_qb : !qref.bit
  %q1 = qref.alloc_qb : !qref.bit
  // CHECK: [[M:%.+]]:2, [[Q:%.+]]:2 = quantum.measurement "pair"(%arg0: f64) qubits([[Q0]], [[Q1]]) static_data = {plane = "XY"} {tag = "keep"} : i1, f64, !quantum.bit, !quantum.bit
  %m, %s = qref.measurement "pair"(%angle: f64) qubits(%q0, %q1)
      static_data = {plane = "XY"} {tag = "keep"} : i1, f64
  // CHECK: [[X:%.+]] = quantum.custom "PauliX"() [[Q]]#0 : !quantum.bit
  qref.custom "PauliX"() %q0 : !qref.bit
  // CHECK: quantum.dealloc_qb [[X]] : !quantum.bit
  // CHECK: quantum.dealloc_qb [[Q]]#1 : !quantum.bit
  qref.dealloc_qb %q0 : !qref.bit
  qref.dealloc_qb %q1 : !qref.bit
  // CHECK: return [[M]]#0, [[M]]#1 : i1, f64
  return %m, %s : i1, f64
}

// -----

// CHECK-LABEL: @extracted_qubits
func.func @extracted_qubits() -> i1 attributes {quantum.node} {
  // CHECK: [[R:%.+]] = quantum.alloc( 2) : !quantum.reg
  %reg = qref.alloc(2) : !qref.reg<2>
  // CHECK: [[Q0:%.+]] = quantum.extract [[R]][ 0] : !quantum.reg -> !quantum.bit
  // CHECK: [[Q1:%.+]] = quantum.extract [[R]][ 1] : !quantum.reg -> !quantum.bit
  %q0 = qref.get %reg[0] : !qref.reg<2> -> !qref.bit
  %q1 = qref.get %reg[1] : !qref.reg<2> -> !qref.bit
  // CHECK: [[M:%.+]], [[Q:%.+]]:2 = quantum.measurement "parity"() qubits([[Q0]], [[Q1]]) : i1, !quantum.bit, !quantum.bit
  %m = qref.measurement "parity"() qubits(%q0, %q1) : i1
  // CHECK: [[R1:%.+]] = quantum.insert [[R]][ 0], [[Q]]#0 : !quantum.reg, !quantum.bit
  // CHECK: [[R2:%.+]] = quantum.insert [[R1]][ 1], [[Q]]#1 : !quantum.reg, !quantum.bit
  // CHECK: quantum.dealloc [[R2]] : !quantum.reg
  qref.dealloc %reg : !qref.reg<2>
  // CHECK: return [[M]]
  return %m : i1
}

// -----

// CHECK-LABEL: @register
func.func @register(%size: i64, %wires: tensor<?xi64>) -> tensor<?xi1> attributes {quantum.node} {
  // CHECK: [[R:%.+]] = quantum.alloc(%arg0) : !quantum.reg
  %reg = qref.alloc(%size) : !qref.reg<?>
  // CHECK: [[M:%.+]], [[R1:%.+]] = quantum.measurement "many"() quregs([[R]]) indices(%arg1: tensor<?xi64>) static_data = {reset = true} : tensor<?xi1>, !quantum.reg
  %m = qref.measurement "many"() quregs(%reg: !qref.reg<?>) indices(%wires: tensor<?xi64>)
      static_data = {reset = true} : tensor<?xi1>
  // CHECK: [[R2:%.+]] = quantum.measurement "discard"() quregs([[R1]]) indices(%arg1: tensor<?xi64>) : !quantum.reg
  qref.measurement "discard"() quregs(%reg: !qref.reg<?>) indices(%wires: tensor<?xi64>) : ()
  // CHECK: quantum.dealloc [[R2]] : !quantum.reg
  qref.dealloc %reg : !qref.reg<?>
  // CHECK: return [[M]]
  return %m : tensor<?xi1>
}

// -----

// CHECK-LABEL: @branch
func.func @branch(%cond: i1) -> i1 attributes {quantum.node} {
  // CHECK: [[Q:%.+]] = quantum.alloc_qb : !quantum.bit
  %q = qref.alloc_qb : !qref.bit
  // CHECK: [[BR:%.+]]:2 = scf.if %arg0 -> (i1, !quantum.bit)
  %m = scf.if %cond -> (i1) {
    // CHECK: [[A:%.+]], [[QA:%.+]] = quantum.measurement "A"() qubits([[Q]]) : i1, !quantum.bit
    %a = qref.measurement "A"() qubits(%q) : i1
    // CHECK: scf.yield [[A]], [[QA]] : i1, !quantum.bit
    scf.yield %a : i1
  } else {
    // CHECK: [[B:%.+]], [[QB:%.+]] = quantum.measurement "B"() qubits([[Q]]) : i1, !quantum.bit
    %b = qref.measurement "B"() qubits(%q) : i1
    // CHECK: scf.yield [[B]], [[QB]] : i1, !quantum.bit
    scf.yield %b : i1
  }
  // CHECK: quantum.dealloc_qb [[BR]]#1
  qref.dealloc_qb %q : !qref.bit
  // CHECK: return [[BR]]#0
  return %m : i1
}
