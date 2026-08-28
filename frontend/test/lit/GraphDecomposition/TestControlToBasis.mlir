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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=C(Hadamard)=1.0 alt-decomps=C(U){}{wires:1}{}=ctrl_u})' %s | FileCheck %s

// XFAIL: *
// Pre-existing decompose-lowering bug (not ctrl-specific): applying a rule to an op-level
// controlled op crashes on a `replaceOp` result-count assertion, and underneath it a
// `ComplexType` cast assertion (Casting.h) that also breaks plain non-ctrl E2E. Independent of
// the graphOpId/rule-synthesis ctrl support; tracked separately.

// CHECK-LABEL: func.func @controlled_basis(
// CHECK-SAME:  %[[C:.*]]: !quantum.bit, %[[Q:.*]]: !quantum.bit
func.func @controlled_basis(%ctrl: !quantum.bit, %q: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK: %[[O:.*]], %[[OC:.*]] = quantum.custom "Hadamard"() %[[Q]] ctrls(%[[C]]) ctrlvals(%{{.*}}) : !quantum.bit ctrls !quantum.bit
  // CHECK: return %[[O]], %[[OC]]
  %out, %outc = quantum.custom "Hadamard"() %q ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %out, %outc : !quantum.bit, !quantum.bit
}

// CHECK-LABEL: func.func @distribution(
// CHECK-SAME:  %[[C:.*]]: !quantum.bit, %[[Q:.*]]: !quantum.bit
func.func @distribution(%ctrl: !quantum.bit, %q: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK-NOT: "U"
  // CHECK: %[[A:.*]], %[[AC:.*]] = quantum.custom "Hadamard"() %[[Q]] ctrls(%[[C]]) ctrlvals(%{{.*}}) : !quantum.bit ctrls !quantum.bit
  // CHECK: %[[B:.*]], %[[BC:.*]] = quantum.custom "Hadamard"() %[[A]] ctrls(%[[AC]]) ctrlvals(%{{.*}}) : !quantum.bit ctrls !quantum.bit
  // CHECK: return %[[B]], %[[BC]]
  %out, %outc = quantum.custom "U"() %q ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %out, %outc : !quantum.bit, !quantum.bit
}

// C(U) distributed to two controlled Hadamards (value-agnostic: controls on all-ones).
func.func private @ctrl_u(%q: !quantum.bit, %ctrl: !quantum.bit) -> (!quantum.bit, !quantum.bit) attributes {
    target_gate = "C(U){}{wires:1}{}",
    resources = {operations = {"C(Hadamard){}{wires:1}{}" = 2 : i64}} } {
  %true = arith.constant true
  %a, %ac = quantum.custom "Hadamard"() %q ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  %b, %bc = quantum.custom "Hadamard"() %a ctrls(%ac) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %b, %bc : !quantum.bit, !quantum.bit
}
