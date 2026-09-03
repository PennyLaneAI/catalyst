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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=C(testT)=1.0 alt-decomps=myCZ{}{wires:2}{}=cz_region})' %s | FileCheck %s

// CHECK-LABEL: func.func @distribute_ctrl_region(
// CHECK-SAME:  [[Q0:%.+]]: !quantum.bit, [[Q1:%.+]]: !quantum.bit
func.func @distribute_ctrl_region(%q0: !quantum.bit, %q1: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
  // CHECK-NOT: quantum.ctrl(
  // CHECK: quantum.custom "testT"() [[Q1]] ctrls([[Q0]]) ctrlvals({{%.+}}) : !quantum.bit ctrls !quantum.bit
  %o:2 = quantum.custom "myCZ"() %q0, %q1 : !quantum.bit, !quantum.bit
  return %o#0, %o#1 : !quantum.bit, !quantum.bit
}

// myCZ decomposes to a single controlled testT, expressed as a `quantum.ctrl` region over testT.
// CHECK-LABEL: func.func private @cz_region
func.func private @cz_region(%q0: !quantum.bit, %q1: !quantum.bit) -> (!quantum.bit, !quantum.bit) attributes {
    target_gate = "{op = \"myCZ\", wires = [2]}",
    resources = {operations = {"{op = \"testT\", traits = {controls = 1 : i64}, wires = [1]}" = 1 : i64}} } {
  %true = arith.constant true
  %oc, %oq = quantum.ctrl(%q0) ctrlvals(%true) (%q1) : !quantum.bit -> !quantum.bit {
  ^bb0(%arg0: !quantum.bit):
    %t = quantum.custom "testT"() %arg0 : !quantum.bit
    quantum.yield %t : !quantum.bit
  }
  return %oc, %oq : !quantum.bit, !quantum.bit
}
