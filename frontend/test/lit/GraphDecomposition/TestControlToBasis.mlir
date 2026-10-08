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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=C(TestHadamard)=1.0 alt-decomps=C(U){}{wires:1}{}=ctrl_u})' %s | FileCheck %s --check-prefixes=ALL,CALL

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{inline-rule-body gate-set=C(TestHadamard)=1.0 alt-decomps=C(U){}{wires:1}{}=ctrl_u})' %s | FileCheck %s --check-prefixes=ALL,INLINE

// ALL-LABEL: func.func @controlled_basis(
// ALL-SAME:  %[[Q:.*]]: !quantum.bit, %[[C:.*]]: !quantum.bit
func.func @controlled_basis(%ctrl: !quantum.bit, %q: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // ALL: %[[O:.*]], %[[OC:.*]] = quantum.custom "TestHadamard"() %[[Q]] ctrls(%[[C]]) ctrlvals(%{{.*}}) : !quantum.bit ctrls !quantum.bit
  // ALL: return %[[O]], %[[OC]]
  %out, %outc = quantum.custom "TestHadamard"() %q ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %out, %outc : !quantum.bit, !quantum.bit
}

// ALL-LABEL: func.func @distribution(
// ALL-SAME:  %[[Q:.*]]: !quantum.bit, %[[C:.*]]: !quantum.bit
func.func @distribution(%ctrl: !quantum.bit, %q: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // ALL-NOT: "U"

  // CALL: %[[OQ:.*]]:2 = call @ctrl_u_0(%[[Q]], %[[C]])
  // CALL: return %[[OQ]]#0, %[[OQ]]#1

  // INLINE: %[[A:.*]], %[[AC:.*]] = quantum.custom "TestHadamard"() %[[Q]] ctrls(%[[C]]) ctrlvals(%{{.*}}) : !quantum.bit ctrls !quantum.bit
  // INLINE: %[[B:.*]], %[[BC:.*]] = quantum.custom "TestHadamard"() %[[A]] ctrls(%[[AC]]) ctrlvals(%{{.*}}) : !quantum.bit ctrls !quantum.bit
  // INLINE: return %[[B]], %[[BC]]
  %out, %outc = quantum.custom "U"() %q ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %out, %outc : !quantum.bit, !quantum.bit
}

// C(U) distributed to two controlled Hadamards (value-agnostic: controls on all-ones).
func.func private @ctrl_u(%q: !quantum.bit, %ctrl: !quantum.bit) -> (!quantum.bit, !quantum.bit) attributes {
    target_gate = "C(U){}{wires:1}{}",
    resources = {operations = {"C(TestHadamard){}{wires:1}{}" = 2 : i64}} } {
  %true = arith.constant true
  %a, %ac = quantum.custom "TestHadamard"() %q ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  %b, %bc = quantum.custom "TestHadamard"() %a ctrls(%ac) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %b, %bc : !quantum.bit, !quantum.bit
}

// CALL: func.func private @ctrl_u_0
