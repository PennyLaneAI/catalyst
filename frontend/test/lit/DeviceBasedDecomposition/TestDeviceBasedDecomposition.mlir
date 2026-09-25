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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(device-based-decomposition{gate-set=C(Adjoint(H))=1.0,C(H)=1.0 alt-decomps=C(Adjoint(U)){}{wires:1}{}=ctrl_adj_u,C(U){}{wires:1}{}=ctrl_u})' %s | FileCheck %s

// CHECK-LABEL: func.func @controlled_adjoint_region(
// CHECK-SAME:  %[[C:.*]]: !quantum.bit, %[[Q:.*]]: !quantum.bit
func.func @controlled_adjoint_region(%c: !quantum.bit, %q: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  %a:2 = quantum.adjoint(%c, %q) : !quantum.bit, !quantum.bit {
    ^bb0(%ac: !quantum.bit, %aq: !quantum.bit):
      %oc, %or = quantum.ctrl(%ac) ctrlvals(%true) (%aq) : !quantum.bit -> !quantum.bit {
        ^bb1(%iq: !quantum.bit):
          %u = quantum.custom "U"() %iq : !quantum.bit
          quantum.yield %u : !quantum.bit
      }
      quantum.yield %oc, %or : !quantum.bit, !quantum.bit
  }
  // CHECK: %[[A:.*]], %[[AC:.*]] = quantum.custom "H"() %[[Q]] adj ctrls(%[[C]]) ctrlvals(%{{.*}}) : !quantum.bit ctrls !quantum.bit
  // CHECK: %[[B:.*]], %[[BC:.*]] = quantum.custom "H"() %[[A]] adj ctrls(%[[AC]]) ctrlvals(%{{.*}}) : !quantum.bit ctrls !quantum.bit
  // CHECK: return %[[BC]], %[[B]]
  return %a#0, %a#1 : !quantum.bit, !quantum.bit
}

// C(Adjoint(U)) -> two C(Adjoint(H)).
func.func private @ctrl_adj_u(%q: !quantum.bit, %ctrl: !quantum.bit) -> (!quantum.bit, !quantum.bit) attributes {
    target_gate = "C(Adjoint(U)){}{wires:1}{}",
    resources = {operations = {"C(Adjoint(H)){}{wires:1}{}" = 2 : i64}} } {
  %true = arith.constant true
  %a, %ac = quantum.custom "H"() %q adj ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  %b, %bc = quantum.custom "H"() %a adj ctrls(%ac) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %b, %bc : !quantum.bit, !quantum.bit
}
