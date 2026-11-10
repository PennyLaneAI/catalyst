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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=C(Test_B)=1.0})' %s | FileCheck %s --check-prefixes=ALL,CALL

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{inline-rule-body gate-set=C(Test_B)=1.0})' %s | FileCheck %s --check-prefixes=ALL,INLINE

// ALL-LABEL: func.func @circuit(
// ALL-SAME:  %[[Q:.*]]: !quantum.bit, %[[W:.*]]: !quantum.bit, %[[C:.*]]: !quantum.bit
func.func @circuit(%q: !quantum.bit, %c: !quantum.bit, %w: !quantum.bit)
    -> (!quantum.bit, !quantum.bit, !quantum.bit) {
  // ALL: %[[TRUE:.*]] = arith.constant true
  %true = arith.constant true

  // CALL: %[[OQ:.*]]:3 = call @test_a_to_b_0(%[[TRUE]], %[[Q]], %[[W]], %[[C]])
  // CALL: return %[[OQ]]#0, %[[OQ]]#1, %[[OQ]]#2

  // INLINE-NOT: quantum.ctrl
  // INLINE: %[[O0:.*]]:2, %[[OC0:.*]] = quantum.operator "Test_B"() qubits(%[[Q]], %[[W]])
  // INLINE-NEXT: ctrls(%[[C]]) ctrl_vals(%[[TRUE]])
  // INLINE: %[[O1:.*]]:2, %[[OC1:.*]] = quantum.operator "Test_B"() qubits(%[[O0]]#0, %[[O0]]#1)
  // INLINE-NEXT: ctrls(%[[OC0]]) ctrl_vals(%[[TRUE]])
  // INLINE: return %[[O1]]#0, %[[O1]]#1, %[[OC1]]
  %out_ctrl, %out_work, %out_q = quantum.ctrl(%c) ctrlvals(%true)
      work_wires(%w) (%q) {work_wire_type = "zeroed"}
      : !quantum.bit work_wires !quantum.bit -> !quantum.bit {
  ^bb0(%arg0: !quantum.bit):
    %a = quantum.custom "Test_A"() %arg0 : !quantum.bit
    quantum.yield %a : !quantum.bit
  }
  return %out_q, %out_ctrl, %out_work : !quantum.bit, !quantum.bit, !quantum.bit
}

// C(Test_A) with one work wire -> two C(Test_B) with the same work-wire layout.
func.func private @test_a_to_b(%q: !quantum.bit, %work: !quantum.bit, %ctrl: !quantum.bit, %cv: i1)
    -> (!quantum.bit, !quantum.bit, !quantum.bit) attributes {
  target_gate = "C(Test_A){}{ctrl_work_wires:1,wires:1}{ctrl_work_wire_type = \22zeroed\22}",
  resources = {operations = {
      "C(Test_B){}{ctrl_work_wires:1,wires:1}{ctrl_work_wire_type = \22zeroed\22}" = 2 : i64
  }}
} {
  %oc, %ow, %oq = quantum.ctrl(%ctrl) ctrlvals(%cv)
      work_wires(%work) (%q) {work_wire_type = "zeroed"}
      : !quantum.bit work_wires !quantum.bit -> !quantum.bit {
  ^bb0(%arg0: !quantum.bit):
    %b0 = quantum.custom "Test_B"() %arg0 : !quantum.bit
    %b1 = quantum.custom "Test_B"() %b0 : !quantum.bit
    quantum.yield %b1 : !quantum.bit
  }
  return %oq, %oc, %ow : !quantum.bit, !quantum.bit, !quantum.bit
}

// CALL: func.func private @test_a_to_b_0
// CALL-NOT: quantum.ctrl
