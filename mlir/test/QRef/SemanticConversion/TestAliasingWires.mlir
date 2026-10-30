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

// RUN: quantum-opt --convert-to-value-semantics --canonicalize --split-input-file --verify-diagnostics %s | FileCheck %s

// Aliasing wires that are known at compile time are rejected instead of being extracted twice.

// qp.ctrl(qp.ctrl(qp.RX(0.1, 0), 1), 1): wire 1 controls a region that also uses wire 1.
func.func @nested_ctrl_same_wire() -> f64 attributes {quantum.node} {
  %false = arith.constant false
  %cst = arith.constant 1.000000e-01 : f64
  %0 = qref.alloc(10) : !qref.reg<10>
  // expected-note@+1 {{first use of the qubit}}
  %1 = qref.get %0[1] : !qref.reg<10> -> !qref.bit
  // expected-error@+1 {{ctrl qubits, work qubits and the qubits used inside the ctrl region must refer to distinct qubits}}
  qref.ctrl(%1) ctrlvals(%false) {
    // expected-note@+1 {{aliasing use of the same qubit}}
    %5 = qref.get %0[1] : !qref.reg<10> -> !qref.bit
    qref.ctrl(%5) ctrlvals(%false) {
      %6 = qref.get %0[0] : !qref.reg<10> -> !qref.bit
      qref.custom "RX"(%cst) %6 : !qref.bit
    }
  }
  %2 = qref.get %0[0] : !qref.reg<10> -> !qref.bit
  %3 = qref.namedobs %2[ PauliZ] : !quantum.obs
  %4 = quantum.expval %3 : f64
  qref.dealloc %0 : !qref.reg<10>
  return %4 : f64
}

// -----

// qp.ctrl(qp.X(1), control=1)
func.func @ctrl_on_target() attributes {quantum.node} {
  %true = arith.constant true
  %r = qref.alloc(2) : !qref.reg<2>
  // expected-note@+1 {{first use of the qubit}}
  %c = qref.get %r[1] : !qref.reg<2> -> !qref.bit
  // expected-error@+1 {{ctrl qubits, work qubits and the qubits used inside the ctrl region must refer to distinct qubits}}
  qref.ctrl(%c) ctrlvals(%true) {
    // expected-note@+1 {{aliasing use of the same qubit}}
    %t = qref.get %r[1] : !qref.reg<2> -> !qref.bit
    qref.custom "PauliX"() %t : !qref.bit
  }
  qref.dealloc %r : !qref.reg<2>
  return
}

// -----

// qp.ctrl(qp.X(1), control=0, work_wires=[1]): work wire overlapping with target
func.func @ctrl_work_wire_on_target() attributes {quantum.node} {
  %true = arith.constant true
  %r = qref.alloc(2) : !qref.reg<2>
  %c = qref.get %r[0] : !qref.reg<2> -> !qref.bit
  // expected-note@+1 {{first use of the qubit}}
  %w = qref.get %r[1] : !qref.reg<2> -> !qref.bit
  // expected-error@+1 {{ctrl qubits, work qubits and the qubits used inside the ctrl region must refer to distinct qubits}}
  qref.ctrl(%c) ctrlvals(%true) work_wires(%w) {
    // expected-note@+1 {{aliasing use of the same qubit}}
    %t = qref.get %r[1] : !qref.reg<2> -> !qref.bit
    qref.custom "PauliX"() %t : !qref.bit
  }
  qref.dealloc %r : !qref.reg<2>
  return
}

// -----

// qp.ctrl(qp.X(0), control=1, work_wires=[1]): work wire overlapping with control
func.func @ctrl_work_wire_on_target() attributes {quantum.node} {
  %true = arith.constant true
  %r = qref.alloc(2) : !qref.reg<2>
  // expected-note@+1 {{first use of the qubit}}
  %c = qref.get %r[1] : !qref.reg<2> -> !qref.bit
  // expected-note@+1 {{aliasing use of the same qubit}}
  %w = qref.get %r[1] : !qref.reg<2> -> !qref.bit
  // expected-error@+1 {{ctrl qubits, work qubits and the qubits used inside the ctrl region must refer to distinct qubits}}
  qref.ctrl(%c) ctrlvals(%true) work_wires(%w) {
    %t = qref.get %r[0] : !qref.reg<2> -> !qref.bit
    qref.custom "PauliX"() %t : !qref.bit
  }
  qref.dealloc %r : !qref.reg<2>
  return
}

// -----

// qp.CNOT([1, 1]), one qref.get per wire use.
func.func @gate_static_alias() attributes {quantum.node} {
  %r = qref.alloc(2) : !qref.reg<2>
  // expected-note@+1 {{first use of the qubit}}
  %a = qref.get %r[1] : !qref.reg<2> -> !qref.bit
  // expected-note@+1 {{aliasing use of the same qubit}}
  %b = qref.get %r[1] : !qref.reg<2> -> !qref.bit
  // expected-error@+1 {{qubit operands must refer to distinct qubits}}
  qref.custom "CNOT"() %a, %b : !qref.bit, !qref.bit
  qref.dealloc %r : !qref.reg<2>
  return
}

// -----

// The same dynamic index Value twice.
func.func @gate_dynamic_alias(%i: i64) attributes {quantum.node} {
  %r = qref.alloc(2) : !qref.reg<2>
  // expected-note@+1 {{first use of the qubit}}
  %a = qref.get %r[%i] : !qref.reg<2>, i64 -> !qref.bit
  // expected-note@+1 {{aliasing use of the same qubit}}
  %b = qref.get %r[%i] : !qref.reg<2>, i64 -> !qref.bit
  // expected-error@+1 {{qubit operands must refer to distinct qubits}}
  qref.custom "CNOT"() %a, %b : !qref.bit, !qref.bit
  qref.dealloc %r : !qref.reg<2>
  return
}

// -----

// The subroutine needs r[1] through its register argument, and the call also passes r[1] as %q.
func.func private @uses_q_and_r1(%r: !qref.reg<3>, %q: !qref.bit) {
  qref.custom "PauliX"() %q : !qref.bit
  %a = qref.get %r[1] : !qref.reg<3> -> !qref.bit
  qref.custom "PauliX"() %a : !qref.bit
  return
}
func.func @call_alias() attributes {quantum.node} {
  %r = qref.alloc(3) : !qref.reg<3>
  // expected-note@+1 {{first use of the qubit}}
  %q = qref.get %r[1] : !qref.reg<3> -> !qref.bit
  // expected-error@+2 {{qubit arguments of a subroutine call must refer to distinct qubits}}
  // expected-note@+1 {{aliasing use of the same qubit}}
  func.call @uses_q_and_r1(%r, %q) : (!qref.reg<3>, !qref.bit) -> ()
  qref.dealloc %r : !qref.reg<3>
  return
}

// -----

// A region that uses r[1] statically and r[%i] dynamically must take the register whole,
// with no separate extract of wire 1 outside the region. Independent of branch order.

// CHECK-LABEL: func.func @if_dynamic_then_static
// CHECK:         [[R:%.+]] = quantum.alloc( 3)
// CHECK-NOT:     quantum.extract
// CHECK:         [[OUT:%.+]] = scf.if {{%.+}} -> (!quantum.reg) {
// CHECK:           quantum.extract [[R]][%arg0]
// CHECK:         } else {
// CHECK:           quantum.extract [[R]][ 1]
// CHECK:         }
// CHECK-NOT:     quantum.insert
// CHECK:         quantum.dealloc [[OUT]]
func.func @if_dynamic_then_static(%i: i64, %cond: i1) attributes {quantum.node} {
  %r = qref.alloc(3) : !qref.reg<3>
  scf.if %cond {
    %a = qref.get %r[%i] : !qref.reg<3>, i64 -> !qref.bit
    qref.custom "PauliX"() %a : !qref.bit
  } else {
    %b = qref.get %r[1] : !qref.reg<3> -> !qref.bit
    qref.custom "PauliX"() %b : !qref.bit
  }
  qref.dealloc %r : !qref.reg<3>
  return
}

// -----

// CHECK-LABEL: func.func @switch_mixed
// CHECK:         [[R:%.+]] = quantum.alloc( 3)
// CHECK-NOT:     quantum.extract
// CHECK:         [[OUT:%.+]] = scf.index_switch {{%.+}} -> !quantum.reg
// CHECK:           quantum.extract [[R]][%arg0]
// CHECK:         default {
// CHECK:           quantum.extract [[R]][ 1]
// CHECK:         }
// CHECK-NOT:     quantum.insert
// CHECK:         quantum.dealloc [[OUT]]
func.func @switch_mixed(%i: i64, %c: index) attributes {quantum.node} {
  %r = qref.alloc(3) : !qref.reg<3>
  scf.index_switch %c
  case 0 {
    %a = qref.get %r[%i] : !qref.reg<3>, i64 -> !qref.bit
    qref.custom "PauliX"() %a : !qref.bit
    scf.yield
  }
  default {
    %b = qref.get %r[1] : !qref.reg<3> -> !qref.bit
    qref.custom "PauliX"() %b : !qref.bit
  }
  qref.dealloc %r : !qref.reg<3>
  return
}

// -----

// CHECK-LABEL: func.func @while_mixed
// CHECK:         [[R:%.+]] = quantum.alloc( 3)
// CHECK-NOT:     quantum.extract
// CHECK:         [[OUT:%.+]] = scf.while ([[ARG:%.+]] = [[R]]) : (!quantum.reg) -> !quantum.reg {
// CHECK:           quantum.extract [[ARG]][%arg0]
// CHECK:         } do {
// CHECK:         ^bb0([[ARG2:%.+]]: !quantum.reg):
// CHECK:           quantum.extract [[ARG2]][ 1]
// CHECK:         }
// CHECK-NOT:     quantum.insert
// CHECK:         quantum.dealloc [[OUT]]
func.func @while_mixed(%i: i64, %flag: i1) attributes {quantum.node} {
  %r = qref.alloc(3) : !qref.reg<3>
  scf.while : () -> () {
    %a = qref.get %r[%i] : !qref.reg<3>, i64 -> !qref.bit
    qref.custom "PauliX"() %a : !qref.bit
    scf.condition(%flag)
  } do {
    %b = qref.get %r[1] : !qref.reg<3> -> !qref.bit
    qref.custom "PauliX"() %b : !qref.bit
    scf.yield
  }
  qref.dealloc %r : !qref.reg<3>
  return
}
