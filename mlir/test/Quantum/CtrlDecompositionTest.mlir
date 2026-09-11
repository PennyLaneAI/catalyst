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

// RUN: quantum-opt --decompose-lowering --split-input-file -verify-diagnostics %s | FileCheck %s


// CHECK-LABEL: func.func @controlled_id_match(
// CHECK-SAME:  %[[C:.*]]: !quantum.bit, %[[Q:.*]]: !quantum.bit
func.func @controlled_id_match(%ctrl: !quantum.bit, %q: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK: %[[O:.*]], %[[OC:.*]] = quantum.custom "Hadamard"() %[[Q]] ctrls(%[[C]]) ctrlvals(%{{.*}}) : !quantum.bit ctrls !quantum.bit
  // CHECK: return %[[O]], %[[OC]]
  %out, %outc = quantum.custom "U"() %q ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %out, %outc : !quantum.bit, !quantum.bit
}

func.func private @ctrl_u(%q: !quantum.bit, %ctrl: !quantum.bit, %cv: i1) -> (!quantum.bit, !quantum.bit)
    attributes {target_gate = "C(U){}{wires:1}{}", llvm.linkage = #llvm.linkage<internal>} {
  %o, %oc = quantum.custom "Hadamard"() %q ctrls(%ctrl) ctrlvals(%cv) : !quantum.bit ctrls !quantum.bit
  return %o, %oc : !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: func.func @no_base_rule_fallback(
// CHECK-SAME:  %[[C:.*]]: !quantum.bit, %[[Q:.*]]: !quantum.bit, %[[T:.*]]: f64
func.func @no_base_rule_fallback(%ctrl: !quantum.bit, %q: !quantum.bit, %theta: f64) -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK: %[[O:.*]], %[[OC:.*]] = quantum.custom "RX"(%[[T]]) %[[Q]] ctrls(%[[C]]) ctrlvals(%{{.*}}) : !quantum.bit ctrls !quantum.bit
  // CHECK-NOT: PauliX
  // CHECK: return %[[O]], %[[OC]]
  %out, %outc = quantum.custom "RX"(%theta) %q ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %out, %outc : !quantum.bit, !quantum.bit
}

func.func private @plain_rx(%theta: f64, %q: !quantum.bit) -> !quantum.bit
    attributes {target_gate = "RX", llvm.linkage = #llvm.linkage<internal>} {
  %o = quantum.custom "PauliX"() %q : !quantum.bit
  return %o : !quantum.bit
}

// -----

// CHECK-LABEL: func.func @distinct_from_base(
// CHECK-SAME:  %[[C:.*]]: !quantum.bit, %[[Q0:.*]]: !quantum.bit, %[[Q1:.*]]: !quantum.bit
func.func @distinct_from_base(%ctrl: !quantum.bit, %q0: !quantum.bit, %q1: !quantum.bit)
    -> (!quantum.bit, !quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // plain U takes the base rule -> PauliX
  // CHECK: %[[A:.*]] = quantum.custom "PauliX"() %[[Q0]] : !quantum.bit
  %a = quantum.custom "U"() %q0 : !quantum.bit
  // C(U) takes the controlled rule -> C(PauliZ)
  // CHECK: %[[B:.*]], %[[BC:.*]] = quantum.custom "PauliZ"() %[[Q1]] ctrls(%[[C]]) ctrlvals(%{{.*}}) : !quantum.bit ctrls !quantum.bit
  %b, %bc = quantum.custom "U"() %q1 ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  // CHECK: return %[[A]], %[[B]], %[[BC]]
  return %a, %b, %bc : !quantum.bit, !quantum.bit, !quantum.bit
}

func.func private @base_u(%q: !quantum.bit) -> !quantum.bit
    attributes {target_gate = "U{}{wires:1}{}", llvm.linkage = #llvm.linkage<internal>} {
  %o = quantum.custom "PauliX"() %q : !quantum.bit
  return %o : !quantum.bit
}

func.func private @ctrl_u2(%q: !quantum.bit, %ctrl: !quantum.bit, %cv: i1) -> (!quantum.bit, !quantum.bit)
    attributes {target_gate = "C(U){}{wires:1}{}", llvm.linkage = #llvm.linkage<internal>} {
  %o, %oc = quantum.custom "PauliZ"() %q ctrls(%ctrl) ctrlvals(%cv) : !quantum.bit ctrls !quantum.bit
  return %o, %oc : !quantum.bit, !quantum.bit
}
