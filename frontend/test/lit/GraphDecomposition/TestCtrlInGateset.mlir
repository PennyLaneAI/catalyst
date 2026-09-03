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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=C(testT)=1.0 alt-decomps=myCZ{}{wires:2}{}=cz_to_ct})' %s | FileCheck %s

// A controlled op that is already in the gate set stays untouched: the controlled ID
// matches the controlled gate-set entry and is not stripped to its base operator.
// CHECK-LABEL: func.func @ctrl_in_gateset(
// CHECK-SAME:  [[C:%.+]]: !quantum.bit, [[Q:%.+]]: !quantum.bit
func.func @ctrl_in_gateset(%c: !quantum.bit, %q: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
  // CHECK: quantum.custom "testT"() [[Q]] ctrls([[C]]) ctrlvals({{%.+}}) : !quantum.bit ctrls !quantum.bit
  %true = arith.constant true
  %outq, %outc = quantum.custom "testT"() %q ctrls(%c) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %outq, %outc : !quantum.bit, !quantum.bit
}

// A plain op decomposes to a controlled op that is in the gate set: the rule's `C(testT)` resource
// is parsed as a distinct node and reached.
// CHECK-LABEL: func.func @decompose_to_ctrl(
// CHECK-SAME:  [[Q0:%.+]]: !quantum.bit, [[Q1:%.+]]: !quantum.bit
func.func @decompose_to_ctrl(%q0: !quantum.bit, %q1: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
  // CHECK-NOT: quantum.custom "myCZ"
  // CHECK: quantum.custom "testT"() [[Q1]] ctrls([[Q0]]) ctrlvals({{%.+}}) : !quantum.bit ctrls !quantum.bit
  %o:2 = quantum.custom "myCZ"() %q0, %q1 : !quantum.bit, !quantum.bit
  return %o#0, %o#1 : !quantum.bit, !quantum.bit
}

func.func private @cz_to_ct(%q0: !quantum.bit, %q1: !quantum.bit) -> (!quantum.bit, !quantum.bit) attributes {
    target_gate = "{op = \"myCZ\", wires = [2]}",
    resources = {operations = {"{op = \"testT\", traits = {controls = 1 : i64}, wires = [1]}" = 1 : i64}} } {
  %true = arith.constant true
  %oq, %oc = quantum.custom "testT"() %q1 ctrls(%q0) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %oc, %oq : !quantum.bit, !quantum.bit
}
