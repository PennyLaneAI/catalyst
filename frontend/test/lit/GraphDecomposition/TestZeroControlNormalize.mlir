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

// Constant-false control values must be conjugated with PauliX and rewritten to all-ones
// ctrlvals before the decomposition graph is built, since `<n>C(...)` GraphOpIds ignore values.
// Keep C(TestRZ) in the gate set so this checks normalization alone (no rule application).
// The private funcs below only mark GraphOpIds as handled so on-demand Python loading is skipped.

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=C(TestRZ)=1.0,PauliX=1.0})' %s | FileCheck %s

// CHECK-LABEL: func.func @zero_control(
// CHECK-SAME:  %[[C:.*]]: !quantum.bit, %[[Q:.*]]: !quantum.bit, %[[T:.*]]: f64
func.func @zero_control(%ctrl: !quantum.bit, %q: !quantum.bit, %theta: f64) -> (!quantum.bit, !quantum.bit) {
  %false = arith.constant false

  // CHECK-DAG: %[[TRUE:.*]] = arith.constant true
  // CHECK: %[[CX0:.*]] = quantum.custom "PauliX"() %[[C]] : !quantum.bit
  // CHECK: %[[O:.*]], %[[OC:.*]] = quantum.custom "TestRZ"(%[[T]]) %[[Q]] ctrls(%[[CX0]]) ctrlvals(%[[TRUE]]) : !quantum.bit ctrls !quantum.bit
  // CHECK: %[[CX1:.*]] = quantum.custom "PauliX"() %[[OC]] : !quantum.bit
  // CHECK: return %[[O]], %[[CX1]]
  %out, %outc = quantum.custom "TestRZ"(%theta) %q ctrls(%ctrl) ctrlvals(%false) : !quantum.bit ctrls !quantum.bit
  return %out, %outc : !quantum.bit, !quantum.bit
}

// A dynamic control value gets an X guarded by `scf.if !value` before and after the gate.
// CHECK-LABEL: func.func @dynamic_control(
// CHECK-SAME:  %[[C:.*]]: !quantum.bit, %[[Q:.*]]: !quantum.bit, %[[T:.*]]: f64, %[[V:.*]]: i1
func.func @dynamic_control(%ctrl: !quantum.bit, %q: !quantum.bit, %theta: f64, %v: i1) -> (!quantum.bit, !quantum.bit) {
  // CHECK-DAG: %[[TRUE:.*]] = arith.constant true
  // CHECK: %[[NOT:.*]] = arith.xori %[[V]], %[[TRUE]] : i1
  // CHECK: %[[PRE:.*]] = scf.if %[[NOT]] -> (!quantum.bit) {
  // CHECK:   %[[X0:.*]] = quantum.custom "PauliX"() %[[C]] : !quantum.bit
  // CHECK:   scf.yield %[[X0]] : !quantum.bit
  // CHECK: } else {
  // CHECK:   scf.yield %[[C]] : !quantum.bit
  // CHECK: }
  // CHECK: %[[O:.*]], %[[OC:.*]] = quantum.custom "TestRZ"(%[[T]]) %[[Q]] ctrls(%[[PRE]]) ctrlvals(%[[TRUE]]) : !quantum.bit ctrls !quantum.bit
  // CHECK: %[[POST:.*]] = scf.if %[[NOT]] -> (!quantum.bit) {
  // CHECK:   %[[X1:.*]] = quantum.custom "PauliX"() %[[OC]] : !quantum.bit
  // CHECK:   scf.yield %[[X1]] : !quantum.bit
  // CHECK: } else {
  // CHECK:   scf.yield %[[OC]] : !quantum.bit
  // CHECK: }
  // CHECK: return %[[O]], %[[POST]]
  %out, %outc = quantum.custom "TestRZ"(%theta) %q ctrls(%ctrl) ctrlvals(%v) : !quantum.bit ctrls !quantum.bit
  return %out, %outc : !quantum.bit, !quantum.bit
}

func.func private @ctrl_rz_handled(%theta: f64, %q: !quantum.bit, %ctrl: !quantum.bit) -> (!quantum.bit, !quantum.bit) attributes {
    target_gate = "C(TestRZ){0:[f64]}{wires:1}{}",
    resources = {operations = {"C(TestRZ){0:[f64]}{wires:1}{}" = 1 : i64}} } {
  %true = arith.constant true
  %o, %oc = quantum.custom "TestRZ"(%theta) %q ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %o, %oc : !quantum.bit, !quantum.bit
}

func.func private @pauli_x_basis(%q: !quantum.bit) -> !quantum.bit attributes {
    target_gate = "PauliX{}{wires:1}{}",
    resources = {operations = {"PauliX{}{wires:1}{}" = 1 : i64}} } {
  %o = quantum.custom "PauliX"() %q : !quantum.bit
  return %o : !quantum.bit
}
