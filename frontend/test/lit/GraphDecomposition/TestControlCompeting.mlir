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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=C(RZ)=1.0 alt-decomps=C(Rot){}{wires:1}{}=dedicated,C(Rot){}{wires:1}{}=distribute})' %s | FileCheck %s

// CHECK-LABEL: func.func @competing(
// CHECK-SAME:  %[[C:.*]]: !quantum.bit, %[[Q:.*]]: !quantum.bit
func.func @competing(%ctrl: !quantum.bit, %q: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK: %[[O:.*]], %[[OC:.*]] = quantum.custom "RZ"() %[[Q]] ctrls(%[[C]]) ctrlvals(%{{.*}}) : !quantum.bit ctrls !quantum.bit
  // CHECK-NOT: quantum.custom "RZ"
  // CHECK: return %[[O]], %[[OC]]
  %out, %outc = quantum.custom "Rot"() %q ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %out, %outc : !quantum.bit, !quantum.bit
}

// Pathway 1: C(Rot) -> C(RZ) (cost 1).
func.func private @dedicated(%q: !quantum.bit, %ctrl: !quantum.bit) -> (!quantum.bit, !quantum.bit) attributes {
    target_gate = "C(Rot){}{wires:1}{}",
    resources = {operations = {"C(RZ){}{wires:1}{}" = 1 : i64}} } {
  %true = arith.constant true
  %o, %oc = quantum.custom "RZ"() %q ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %o, %oc : !quantum.bit, !quantum.bit
}

// Pathway 2: C(Rot) -> C(RZ) C(RZ) (cost 2).
func.func private @distribute(%q: !quantum.bit, %ctrl: !quantum.bit) -> (!quantum.bit, !quantum.bit) attributes {
    target_gate = "C(Rot){}{wires:1}{}",
    resources = {operations = {"C(RZ){}{wires:1}{}" = 2 : i64}} } {
  %true = arith.constant true
  %a, %ac = quantum.custom "RZ"() %q ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  %b, %bc = quantum.custom "RZ"() %a ctrls(%ac) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %b, %bc : !quantum.bit, !quantum.bit
}
