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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=2C(H)=1.0,3C(H)=1.0 alt-decomps=2C(U){}{wires:1}{}=cc_u,3C(U){}{wires:1}{}=ccc_u})' %s | FileCheck %s

// CHECK-LABEL: func.func @two_controls(
func.func @two_controls(%c1: !quantum.bit, %c2: !quantum.bit, %q: !quantum.bit)
    -> (!quantum.bit, !quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK-NOT: "U"
  // CHECK: %[[A:.*]], %[[AC:.*]]:2 = quantum.custom "H"() %{{.*}} ctrls(%{{.*}}, %{{.*}}) ctrlvals(%{{.*}}, %{{.*}}) : !quantum.bit ctrls !quantum.bit, !quantum.bit
  // CHECK: %[[B:.*]], %[[BC:.*]]:2 = quantum.custom "H"() %[[A]] ctrls(%[[AC]]#0, %[[AC]]#1) ctrlvals(%{{.*}}, %{{.*}}) : !quantum.bit ctrls !quantum.bit, !quantum.bit
  %out, %outc:2 = quantum.custom "U"() %q ctrls(%c1, %c2) ctrlvals(%true, %true) : !quantum.bit ctrls !quantum.bit, !quantum.bit
  return %out, %outc#0, %outc#1 : !quantum.bit, !quantum.bit, !quantum.bit
}

// CHECK-LABEL: func.func @three_controls(
func.func @three_controls(%c1: !quantum.bit, %c2: !quantum.bit, %c3: !quantum.bit, %q: !quantum.bit)
    -> (!quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK-NOT: "U"
  // CHECK: %{{.*}}, %{{.*}}:3 = quantum.custom "H"() %{{.*}} ctrls(%{{.*}}, %{{.*}}, %{{.*}}) ctrlvals(%{{.*}}, %{{.*}}, %{{.*}}) : !quantum.bit ctrls !quantum.bit, !quantum.bit, !quantum.bit
  %out, %outc:3 = quantum.custom "U"() %q ctrls(%c1, %c2, %c3) ctrlvals(%true, %true, %true) : !quantum.bit ctrls !quantum.bit, !quantum.bit, !quantum.bit
  return %out, %outc#0, %outc#1, %outc#2 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
}

// 2C(U) -> two 2C(H).
func.func private @cc_u(%q: !quantum.bit, %c1: !quantum.bit, %c2: !quantum.bit)
    -> (!quantum.bit, !quantum.bit, !quantum.bit) attributes {
    target_gate = "2C(U){}{wires:1}{}",
    resources = {operations = {"2C(H){}{wires:1}{}" = 2 : i64}} } {
  %true = arith.constant true
  %a, %ac:2 = quantum.custom "H"() %q ctrls(%c1, %c2) ctrlvals(%true, %true) : !quantum.bit ctrls !quantum.bit, !quantum.bit
  %b, %bc:2 = quantum.custom "H"() %a ctrls(%ac#0, %ac#1) ctrlvals(%true, %true) : !quantum.bit ctrls !quantum.bit, !quantum.bit
  return %b, %bc#0, %bc#1 : !quantum.bit, !quantum.bit, !quantum.bit
}

// 3C(U) -> one 3C(H).
func.func private @ccc_u(%q: !quantum.bit, %c1: !quantum.bit, %c2: !quantum.bit, %c3: !quantum.bit)
    -> (!quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit) attributes {
    target_gate = "3C(U){}{wires:1}{}",
    resources = {operations = {"3C(H){}{wires:1}{}" = 1 : i64}} } {
  %true = arith.constant true
  %o, %oc:3 = quantum.custom "H"() %q ctrls(%c1, %c2, %c3) ctrlvals(%true, %true, %true) : !quantum.bit ctrls !quantum.bit, !quantum.bit, !quantum.bit
  return %o, %oc#0, %oc#1, %oc#2 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
}
