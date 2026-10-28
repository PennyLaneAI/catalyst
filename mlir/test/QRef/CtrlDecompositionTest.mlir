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

// RUN: quantum-opt --decompose-lowering --split-input-file -verify-diagnostics %s | FileCheck %s --check-prefixes=ALL,CALL
// RUN: quantum-opt --decompose-lowering=inline-rule-body --split-input-file -verify-diagnostics %s | FileCheck %s --check-prefixes=ALL,INLINE

func.func  @main_circuit() attributes {quantum.node} {
  %0 = quantum.alloc( 2) : !quantum.reg

  // ALL: [[q:%.+]] = quantum.extract {{%.+}}[ 1] : !quantum.reg -> !quantum.bit
  // ALL: [[c:%.+]] = quantum.extract {{%.+}}[ 0] : !quantum.reg -> !quantum.bit
  // ALL: {{%.+}}:2 = call @controlled_id_match([[q]], [[c]])
  %c = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
  %q = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
  %1:2 = func.call @controlled_id_match(%c, %q) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)

  %2 = quantum.insert %0[ 0], %1#1 : !quantum.reg, !quantum.bit
  %3 = quantum.insert %2[ 1], %1#0 : !quantum.reg, !quantum.bit
  quantum.dealloc %3 : !quantum.reg
  return
}

// ALL-LABEL: func.func @controlled_id_match(
// ALL-SAME:  %[[Q:.*]]: !quantum.bit, %[[C:.*]]: !quantum.bit
func.func @controlled_id_match(%ctrl: !quantum.bit, %q: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CALL: %[[oq:.*]]:2 = call @ctrl_u_0(%{{.*}}, %[[Q]], %[[C]]) : (i1, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
  // CALL: return %[[oq]]#0, %[[oq]]#1 : !quantum.bit, !quantum.bit

  // INLINE: %[[O:.*]], %[[OC:.*]] = quantum.custom "Hadamard"() %[[Q]] ctrls(%[[C]]) ctrlvals(%{{.*}}) : !quantum.bit ctrls !quantum.bit
  // INLINE: return %[[O]], %[[OC]]
  %out, %outc = quantum.custom "U"() %q ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %out, %outc : !quantum.bit, !quantum.bit
}

// CALL: func.func private @ctrl_u_0(%arg0: i1, %[[Q:.*]]: !quantum.bit, %[[C:.*]]: !quantum.bit) -> (!quantum.bit, !quantum.bit)
// CALL:   %[[O:.*]], %[[OC:.*]] = quantum.custom "Hadamard"() %[[Q]] ctrls(%[[C]]) ctrlvals(%arg0) : !quantum.bit ctrls !quantum.bit
// CALL:   return %[[O]], %[[OC]]

func.func private @ctrl_u(%q: !quantum.bit, %ctrl: !quantum.bit, %cv: i1) -> (!quantum.bit, !quantum.bit)
    attributes {target_gate = "C(U){}{wires:1}{}", llvm.linkage = #llvm.linkage<internal>} {
  %o, %oc = quantum.custom "Hadamard"() %q ctrls(%ctrl) ctrlvals(%cv) : !quantum.bit ctrls !quantum.bit
  return %o, %oc : !quantum.bit, !quantum.bit
}

// -----

// ALL-LABEL: func.func @no_base_rule_fallback(
// ALL-SAME:  %[[T:.*]]: f64, %[[Q:.*]]: !quantum.bit, %[[C:.*]]: !quantum.bit
func.func @no_base_rule_fallback(%ctrl: !quantum.bit, %q: !quantum.bit, %theta: f64) -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // ALL: %[[O:.*]], %[[OC:.*]] = quantum.custom "RX"(%[[T]]) %[[Q]] ctrls(%[[C]]) ctrlvals(%{{.*}}) : !quantum.bit ctrls !quantum.bit
  // ALL-NOT: PauliX
  // ALL: return %[[O]], %[[OC]]
  %out, %outc = quantum.custom "RX"(%theta) %q ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %out, %outc : !quantum.bit, !quantum.bit
}

func.func private @plain_rx(%theta: f64, %q: !quantum.bit) -> !quantum.bit
    attributes {target_gate = "RX", llvm.linkage = #llvm.linkage<internal>} {
  %o = quantum.custom "PauliX"() %q : !quantum.bit
  return %o : !quantum.bit
}

// -----

// ALL-LABEL: func.func @distinct_from_base(
// ALL-SAME:  %[[Q0:.*]]: !quantum.bit, %[[Q1:.*]]: !quantum.bit, %[[C:.*]]: !quantum.bit
func.func @distinct_from_base(%ctrl: !quantum.bit, %q0: !quantum.bit, %q1: !quantum.bit)
    -> (!quantum.bit, !quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // plain U takes the base rule -> PauliX
  // CALL: %[[A:.*]] = call @base_u_1(%[[Q0]]) : (!quantum.bit) -> !quantum.bit
  // INLINE: %[[A:.*]] = quantum.custom "PauliX"() %[[Q0]] : !quantum.bit
  %a = quantum.custom "U"() %q0 : !quantum.bit

  // C(U) takes the controlled rule -> C(PauliZ)
  // CALL: %[[OQ:.*]]:2 = call @ctrl_u2_0(%{{.*}}, %[[Q1]], %[[C]]) : (i1, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
  // INLINE: %[[B:.*]], %[[BC:.*]] = quantum.custom "PauliZ"() %[[Q1]] ctrls(%[[C]]) ctrlvals(%{{.*}}) : !quantum.bit ctrls !quantum.bit
  %b, %bc = quantum.custom "U"() %q1 ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit

  // CALL: return %[[A]], %[[OQ]]#0, %[[OQ]]#1
  // INLINE: return %[[A]], %[[B]], %[[BC]]
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
