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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=C(T)=1.0 alt-decomps=C(V){}{wires:1}{}=v_to_ctrl_t})' %s | FileCheck %s

// XFAIL: *
// Pre-existing decompose-lowering bug (not ctrl-specific): applying a rule to an op-level
// controlled op crashes on a `replaceOp` result-count assertion, and underneath it a
// `ComplexType` cast assertion (Casting.h) that also breaks plain non-ctrl E2E. Independent of
// the graphOpId/rule-synthesis ctrl support; tracked separately.

// CHECK-LABEL: func.func @controlled_in_gateset(
// CHECK-SAME:  %[[C:.*]]: !quantum.bit, %[[Q:.*]]: !quantum.bit
func.func @controlled_in_gateset(%ctrl: !quantum.bit, %q: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK: %[[O:.*]], %[[OC:.*]] = quantum.custom "T"() %[[Q]] ctrls(%[[C]]) ctrlvals(%{{.*}}) : !quantum.bit ctrls !quantum.bit
  // CHECK: return %[[O]], %[[OC]]
  %out, %outc = quantum.custom "T"() %q ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %out, %outc : !quantum.bit, !quantum.bit
}

// CHECK-LABEL: func.func @decompose_to_controlled(
// CHECK-SAME:  %[[C:.*]]: !quantum.bit, %[[Q:.*]]: !quantum.bit
func.func @decompose_to_controlled(%ctrl: !quantum.bit, %q: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK-NOT: "V"
  // CHECK: %[[O:.*]], %[[OC:.*]] = quantum.custom "T"() %[[Q]] ctrls(%[[C]]) ctrlvals(%{{.*}}) : !quantum.bit ctrls !quantum.bit
  // CHECK: return %[[O]], %[[OC]]
  %out, %outc = quantum.custom "V"() %q ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %out, %outc : !quantum.bit, !quantum.bit
}

func.func private @v_to_ctrl_t(%q: !quantum.bit, %ctrl: !quantum.bit) -> (!quantum.bit, !quantum.bit) attributes {
    target_gate = "C(V){}{wires:1}{}",
    resources = {operations = {"C(T){}{wires:1}{}" = 1 : i64}} } {
  %true = arith.constant true
  %o, %oc = quantum.custom "T"() %q ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %o, %oc : !quantum.bit, !quantum.bit
}
