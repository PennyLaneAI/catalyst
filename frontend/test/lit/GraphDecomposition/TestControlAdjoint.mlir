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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=C(Adjoint(H))=1.0,C(H)=1.0 alt-decomps=C(Adjoint(U)){}{wires:1}{}=ctrl_adj_u,C(U){}{wires:1}{}=ctrl_u})' %s | FileCheck %s

// CHECK-LABEL: func.func @controlled_adjoint(
// CHECK-SAME:  %[[C:.*]]: !quantum.bit, %[[Q:.*]]: !quantum.bit
func.func @controlled_adjoint(%ctrl: !quantum.bit, %q: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // C(Adjoint(U)) takes the ctrl_adj_u rule -> two C(Adjoint(H)) (adj + ctrls)
  // CHECK: %[[A:.*]], %[[AC:.*]] = quantum.custom "H"() %[[Q]] adj ctrls(%[[C]]) ctrlvals(%{{.*}}) : !quantum.bit ctrls !quantum.bit
  // CHECK: %[[B:.*]], %[[BC:.*]] = quantum.custom "H"() %[[A]] adj ctrls(%[[AC]]) ctrlvals(%{{.*}}) : !quantum.bit ctrls !quantum.bit
  // CHECK: return %[[B]], %[[BC]]
  %out, %outc = quantum.custom "U"() %q adj ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %out, %outc : !quantum.bit, !quantum.bit
}

// CHECK-LABEL: func.func @plain_controlled(
// CHECK-SAME:  %[[C:.*]]: !quantum.bit, %[[Q:.*]]: !quantum.bit
func.func @plain_controlled(%ctrl: !quantum.bit, %q: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // C(U) (no adjoint) is a distinct node: it takes ctrl_u -> a single non-adjoint C(H)
  // CHECK: %[[O:.*]], %[[OC:.*]] = quantum.custom "H"() %[[Q]] ctrls(%[[C]]) ctrlvals(%{{.*}}) : !quantum.bit ctrls !quantum.bit
  // CHECK-NOT: adj
  // CHECK: return %[[O]], %[[OC]]
  %out, %outc = quantum.custom "U"() %q ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %out, %outc : !quantum.bit, !quantum.bit
}

// C(Adjoint(U)) -> two C(Adjoint(H)).
func.func private @ctrl_adj_u(%q: !quantum.bit, %ctrl: !quantum.bit) -> (!quantum.bit, !quantum.bit) attributes {
    target_gate = "{op = \"U\", traits = {adj = true, controls = 1 : i64}, wires = [1]}",
    resources = {operations = {"{op = \"H\", traits = {adj = true, controls = 1 : i64}, wires = [1]}" = 2 : i64}} } {
  %true = arith.constant true
  %a, %ac = quantum.custom "H"() %q adj ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  %b, %bc = quantum.custom "H"() %a adj ctrls(%ac) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %b, %bc : !quantum.bit, !quantum.bit
}

// C(U) -> a single C(H).
func.func private @ctrl_u(%q: !quantum.bit, %ctrl: !quantum.bit) -> (!quantum.bit, !quantum.bit) attributes {
    target_gate = "{op = \"U\", traits = {controls = 1 : i64}, wires = [1]}",
    resources = {operations = {"{op = \"H\", traits = {controls = 1 : i64}, wires = [1]}" = 1 : i64}} } {
  %true = arith.constant true
  %o, %oc = quantum.custom "H"() %q ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %o, %oc : !quantum.bit, !quantum.bit
}
