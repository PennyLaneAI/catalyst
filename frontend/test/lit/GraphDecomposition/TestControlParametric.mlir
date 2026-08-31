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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=C(PhaseShift)=1.0 alt-decomps=C(RZ){0:[f64]}{wires:1}{}=ctrl_rz})' %s | FileCheck %s

// CHECK-LABEL: func.func @parametric(
// CHECK-SAME:  %[[C:.*]]: !quantum.bit, %[[Q:.*]]: !quantum.bit, %[[T:.*]]: f64
func.func @parametric(%ctrl: !quantum.bit, %q: !quantum.bit, %theta: f64) -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK-NOT: "RZ"
  // CHECK: %[[O:.*]], %[[OC:.*]] = quantum.custom "PhaseShift"(%[[T]]) %[[Q]] ctrls(%[[C]]) ctrlvals(%{{.*}}) : !quantum.bit ctrls !quantum.bit
  // CHECK: return %[[O]], %[[OC]]
  %out, %outc = quantum.custom "RZ"(%theta) %q ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %out, %outc : !quantum.bit, !quantum.bit
}

// C(RZ)(theta) -> C(PhaseShift)(theta): the parameter passes through, the control is value-agnostic.
func.func private @ctrl_rz(%theta: f64, %q: !quantum.bit, %ctrl: !quantum.bit) -> (!quantum.bit, !quantum.bit) attributes {
    target_gate = "C(RZ){0:[f64]}{wires:1}{}",
    resources = {operations = {"C(PhaseShift){0:[f64]}{wires:1}{}" = 1 : i64}} } {
  %true = arith.constant true
  %o, %oc = quantum.custom "PhaseShift"(%theta) %q ctrls(%ctrl) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
  return %o, %oc : !quantum.bit, !quantum.bit
}
