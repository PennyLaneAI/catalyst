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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=C(TestT)=1.0 alt-decomps=C(V){}{wires:1}{}=v_to_ctrl_t})' %s | FileCheck %s --check-prefixes=ALL,CALL

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{inline-rule-body gate-set=C(TestT)=1.0 alt-decomps=C(V){}{wires:1}{}=v_to_ctrl_t})' %s | FileCheck %s --check-prefixes=ALL,INLINE

// ALL-LABEL: func.func @controlled_in_gateset(
// ALL-SAME:  %[[Q:.*]]: !quantum.bit, %[[C:.*]]: !quantum.bit
func.func @controlled_in_gateset(%ctrl: !quantum.bit, %q: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // ALL: %[[O:.*]], %[[OC:.*]] = quantum.custom "TestT"() %[[Q]] ctrls(%[[C]]) ctrlvals(%{{.*}}) : !quantum.bit ctrls !quantum.bit
  // ALL: return %[[O]], %[[OC]]
  %out, %outc = quantum.custom "TestT"() %q ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %out, %outc : !quantum.bit, !quantum.bit
}

// ALL-LABEL: func.func @decompose_to_controlled(
// ALL-SAME:  %[[Q:.*]]: !quantum.bit, %[[C:.*]]: !quantum.bit
func.func @decompose_to_controlled(%ctrl: !quantum.bit, %q: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // ALL-NOT: "V"

  // CALL: %[[OQ:.*]]:2 = call @v_to_ctrl_t_0(%[[Q]], %[[C]])
  // CALL: return %[[OQ]]#0, %[[OQ]]#1

  // INLINE: %[[O:.*]], %[[OC:.*]] = quantum.custom "TestT"() %[[Q]] ctrls(%[[C]]) ctrlvals(%{{.*}}) : !quantum.bit ctrls !quantum.bit
  // INLINE: return %[[O]], %[[OC]]
  %out, %outc = quantum.custom "V"() %q ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %out, %outc : !quantum.bit, !quantum.bit
}

func.func private @v_to_ctrl_t(%q: !quantum.bit, %ctrl: !quantum.bit) -> (!quantum.bit, !quantum.bit) attributes {
    target_gate = "C(V){}{wires:1}{}",
    resources = {operations = {"C(TestT){}{wires:1}{}" = 1 : i64}} } {
  %true = arith.constant true
  %o, %oc = quantum.custom "TestT"() %q ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %o, %oc : !quantum.bit, !quantum.bit
}

// CALL: func.func private @v_to_ctrl_t_0
