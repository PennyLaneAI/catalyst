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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=C(Adjoint(H))=1.0 alt-decomps=C(Adjoint(U)){}{wires:1}{}=ctrl_adj_u})' %s | FileCheck %s

// CHECK-LABEL: func.func @composite_region(
// CHECK-SAME:  %[[C:.*]]: !quantum.bit, %[[Q:.*]]: !quantum.bit
func.func @composite_region(%ctrl: !quantum.bit, %q: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK-NOT: quantum.ctrl
  // CHECK: %[[A:.*]], %[[AC:.*]] = quantum.custom "H"() %[[Q]] adj ctrls(%{{.*}}) ctrlvals(%{{.*}}) : !quantum.bit ctrls !quantum.bit
  // CHECK: %[[B:.*]], %[[BC:.*]] = quantum.custom "H"() %[[A]] adj ctrls(%[[AC]]) ctrlvals(%{{.*}}) : !quantum.bit ctrls !quantum.bit
  // CHECK: return %[[B]], %[[BC]]
  %out, %outc = quantum.custom "U"() %q adj ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %out, %outc : !quantum.bit, !quantum.bit
}

// C(Adjoint(U)) rule: base decomposition (H; H) as adjoint gates inside a quantum.ctrl region.
func.func private @ctrl_adj_u(%q: !quantum.bit, %ctrl: !quantum.bit) -> (!quantum.bit, !quantum.bit) attributes {
    target_gate = "C(Adjoint(U)){}{wires:1}{}",
    resources = {operations = {"C(Adjoint(H)){}{wires:1}{}" = 2 : i64}} } {
  %true = arith.constant true
  %oc, %oq = quantum.ctrl(%ctrl) ctrlvals(%true) (%q) : !quantum.bit -> !quantum.bit {
  ^bb0(%arg0: !quantum.bit):
    %a = quantum.custom "H"() %arg0 adj : !quantum.bit
    %b = quantum.custom "H"() %a adj : !quantum.bit
    quantum.yield %b : !quantum.bit
  }
  return %oq, %oc : !quantum.bit, !quantum.bit
}
