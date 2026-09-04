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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=2C(H)=1.0 alt-decomps=2C(U){}{wires:1}{}=cc_u})' %s | FileCheck %s

// CHECK-LABEL: func.func @multi_region(
func.func @multi_region(%c1: !quantum.bit, %c2: !quantum.bit, %q: !quantum.bit)
    -> (!quantum.bit, !quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK-NOT: quantum.ctrl
  // CHECK: %[[A:.*]], %[[AC:.*]]:2 = quantum.custom "H"() %{{.*}} ctrls(%{{.*}}, %{{.*}}) ctrlvals(%{{.*}}, %{{.*}}) : !quantum.bit ctrls !quantum.bit, !quantum.bit
  // CHECK: %[[B:.*]], %[[BC:.*]]:2 = quantum.custom "H"() %[[A]] ctrls(%[[AC]]#0, %[[AC]]#1) ctrlvals(%{{.*}}, %{{.*}}) : !quantum.bit ctrls !quantum.bit, !quantum.bit
  // CHECK: return %[[B]], %[[BC]]#0, %[[BC]]#1
  %out, %outc:2 = quantum.custom "U"() %q ctrls(%c1, %c2) ctrlvals(%true, %true) : !quantum.bit ctrls !quantum.bit, !quantum.bit
  return %out, %outc#0, %outc#1 : !quantum.bit, !quantum.bit, !quantum.bit
}

// 2C(U): base decomposition (H; H) inside a two-control quantum.ctrl region.
func.func private @cc_u(%q: !quantum.bit, %c1: !quantum.bit, %c2: !quantum.bit)
    -> (!quantum.bit, !quantum.bit, !quantum.bit) attributes {
    target_gate = "{op = \"U\", traits = {controls = 2 : i64}, wires = [1]}",
    resources = {operations = {"{op = \"H\", traits = {controls = 2 : i64}, wires = [1]}" = 2 : i64}} } {
  %true = arith.constant true
  %oc:2, %oq = quantum.ctrl(%c1, %c2) ctrlvals(%true, %true) (%q) : !quantum.bit, !quantum.bit -> !quantum.bit {
  ^bb0(%arg0: !quantum.bit):
    %a = quantum.custom "H"() %arg0 : !quantum.bit
    %b = quantum.custom "H"() %a : !quantum.bit
    quantum.yield %b : !quantum.bit
  }
  return %oq, %oc#0, %oc#1 : !quantum.bit, !quantum.bit, !quantum.bit
}
