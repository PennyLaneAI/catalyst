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

// RUN: quantum-opt --ctrl-lowering --split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @ctrl_single_gate
func.func @ctrl_single_gate(%ctrl: !quantum.bit, %q: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
  // CHECK-NOT: quantum.ctrl
  // CHECK: %[[TRUE:.*]] = arith.constant true
  %true = arith.constant true
  // CHECK: %[[OUT:.*]], %[[OUTC:.*]] = quantum.custom "Hadamard"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%[[TRUE]]) : !quantum.bit ctrls !quantum.bit
  %outc, %outq = quantum.ctrl(%ctrl) ctrlvals(%true) (%q) : !quantum.bit -> !quantum.bit {
  ^bb0(%arg0: !quantum.bit):
    %h = quantum.custom "Hadamard"() %arg0 : !quantum.bit
    quantum.yield %h : !quantum.bit
  }
  // CHECK: return %[[OUT]], %[[OUTC]]
  return %outq, %outc : !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: @ctrl_two_gates
func.func @ctrl_two_gates(%ctrl: !quantum.bit, %q: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK: %[[O1:.*]], %[[C1:.*]] = quantum.custom "Hadamard"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}})
  // CHECK: %[[O2:.*]], %[[C2:.*]] = quantum.custom "PauliX"() %[[O1]] ctrls(%[[C1]]) ctrlvals(%{{.*}})
  %outc, %outq = quantum.ctrl(%ctrl) ctrlvals(%true) (%q) : !quantum.bit -> !quantum.bit {
  ^bb0(%arg0: !quantum.bit):
    %h = quantum.custom "Hadamard"() %arg0 : !quantum.bit
    %x = quantum.custom "PauliX"() %h : !quantum.bit
    quantum.yield %x : !quantum.bit
  }
  return %outq, %outc : !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: @ctrl_merge_existing(
// CHECK-SAME:  %[[CTRL:[^:]+]]: !quantum.bit, %[[INNER:[^:]+]]: !quantum.bit, %[[Q:[^:]+]]: !quantum.bit
func.func @ctrl_merge_existing(%ctrl: !quantum.bit, %inner: !quantum.bit, %q: !quantum.bit)
    -> (!quantum.bit, !quantum.bit, !quantum.bit) {
  // CHECK-DAG: %[[TRUE:.*]] = arith.constant true
  // CHECK-DAG: %[[FALSE:.*]] = arith.constant false
  %true = arith.constant true
  %false = arith.constant false
  // CHECK: %[[TO:.*]], %[[TCC:.*]]:2 = quantum.custom "PauliX"() %[[Q]] ctrls(%[[INNER]], %[[CTRL]]) ctrlvals(%[[TRUE]], %[[FALSE]]) : !quantum.bit ctrls !quantum.bit, !quantum.bit
  // CHECK: return %[[TCC]]#0, %[[TO]], %[[TCC]]#1
  %outc, %outq:2 = quantum.ctrl(%ctrl) ctrlvals(%false) (%inner, %q) : !quantum.bit -> !quantum.bit, !quantum.bit {
  ^bb0(%argc: !quantum.bit, %argq: !quantum.bit):
    %xo, %xc = quantum.custom "PauliX"() %argq ctrls(%argc) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    quantum.yield %xc, %xo : !quantum.bit, !quantum.bit
  }
  return %outq#0, %outq#1, %outc : !quantum.bit, !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: @ctrl_duplicate
func.func @ctrl_duplicate(%ctrl: !quantum.bit, %inner: !quantum.bit, %q: !quantum.bit)
    -> (!quantum.bit, !quantum.bit, !quantum.bit) {
  // CHECK: %[[TRUE:.*]] = arith.constant true
  %true = arith.constant true
  // CHECK: quantum.custom "PauliX"() %{{.*}} ctrls(%{{.*}}, %{{.*}}) ctrlvals(%[[TRUE]], %[[TRUE]]) : !quantum.bit ctrls !quantum.bit, !quantum.bit
  %outc, %outq:2 = quantum.ctrl(%ctrl) ctrlvals(%true) (%inner, %q) : !quantum.bit -> !quantum.bit, !quantum.bit {
  ^bb0(%argc: !quantum.bit, %argq: !quantum.bit):
    %xo, %xc = quantum.custom "PauliX"() %argq ctrls(%argc) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    quantum.yield %xc, %xo : !quantum.bit, !quantum.bit
  }
  return %outq#0, %outq#1, %outc : !quantum.bit, !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: @ctrl_zero_value_not_expanded
func.func @ctrl_zero_value_not_expanded(%ctrl: !quantum.bit, %q: !quantum.bit)
    -> (!quantum.bit, !quantum.bit) {
  // CHECK: %[[FALSE:.*]] = arith.constant false
  %false = arith.constant false
  // CHECK-NOT: quantum.custom "PauliX"
  // CHECK: quantum.custom "Hadamard"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%[[FALSE]])
  // CHECK-NOT: quantum.custom "PauliX"
  %outc, %outq = quantum.ctrl(%ctrl) ctrlvals(%false) (%q) : !quantum.bit -> !quantum.bit {
  ^bb0(%arg0: !quantum.bit):
    %h = quantum.custom "Hadamard"() %arg0 : !quantum.bit
    quantum.yield %h : !quantum.bit
  }
  return %outq, %outc : !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: @ctrl_adjoint_gate
func.func @ctrl_adjoint_gate(%ctrl: !quantum.bit, %q: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK: quantum.custom "S"() %{{.*}} adj ctrls(%{{.*}}) ctrlvals(%{{.*}})
  %outc, %outq = quantum.ctrl(%ctrl) ctrlvals(%true) (%q) : !quantum.bit -> !quantum.bit {
  ^bb0(%arg0: !quantum.bit):
    %s = quantum.custom "S"() %arg0 adj : !quantum.bit
    quantum.yield %s : !quantum.bit
  }
  return %outq, %outc : !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: @ctrl_over_register
func.func @ctrl_over_register(%ctrl: !quantum.bit, %reg: !quantum.reg) -> (!quantum.bit, !quantum.reg) {
  %true = arith.constant true
  // CHECK: %[[Q:.*]] = quantum.extract %{{.*}}[ 0]
  // CHECK: %[[HO:.*]], %[[HC:.*]] = quantum.custom "Hadamard"() %[[Q]] ctrls(%{{.*}}) ctrlvals(%{{.*}})
  // CHECK: quantum.insert %{{.*}}[ 0], %[[HO]]
  %outc, %outr = quantum.ctrl(%ctrl) ctrlvals(%true) (%reg) : !quantum.bit -> !quantum.reg {
  ^bb0(%arg0: !quantum.reg):
    %q = quantum.extract %arg0[ 0] : !quantum.reg -> !quantum.bit
    %h = quantum.custom "Hadamard"() %q : !quantum.bit
    %r = quantum.insert %arg0[ 0], %h : !quantum.reg, !quantum.bit
    quantum.yield %r : !quantum.reg
  }
  return %outc, %outr : !quantum.bit, !quantum.reg
}

// -----

// CHECK-LABEL: @ctrl_nested
func.func @ctrl_nested(%outer: !quantum.bit, %inner: !quantum.bit, %q: !quantum.bit)
    -> (!quantum.bit, !quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK-NOT: quantum.ctrl
  // CHECK: quantum.custom "PauliZ"() %{{.*}} ctrls(%{{.*}}, %{{.*}}) ctrlvals(%{{.*}}, %{{.*}})
  %oc, %or:2 = quantum.ctrl(%outer) ctrlvals(%true) (%inner, %q) : !quantum.bit -> !quantum.bit, !quantum.bit {
  ^bb0(%argi: !quantum.bit, %argq: !quantum.bit):
    %ic, %iq = quantum.ctrl(%argi) ctrlvals(%true) (%argq) : !quantum.bit -> !quantum.bit {
    ^bb1(%argq2: !quantum.bit):
      %z = quantum.custom "PauliZ"() %argq2 : !quantum.bit
      quantum.yield %z : !quantum.bit
    }
    quantum.yield %ic, %iq : !quantum.bit, !quantum.bit
  }
  return %or#0, %or#1, %oc : !quantum.bit, !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: @ctrl_nested_gate_before_after
func.func @ctrl_nested_gate_before_after(%outer: !quantum.bit, %inner: !quantum.bit, %q: !quantum.bit)
    -> (!quantum.bit, !quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK-NOT: quantum.ctrl
  // CHECK: quantum.custom "Hadamard"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}})
  // CHECK: quantum.custom "PauliZ"() %{{.*}} ctrls(%{{.*}}, %{{.*}}) ctrlvals(%{{.*}}, %{{.*}})
  // CHECK: quantum.custom "T"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}})
  %oc, %or:2 = quantum.ctrl(%outer) ctrlvals(%true) (%inner, %q) : !quantum.bit -> !quantum.bit, !quantum.bit {
  ^bb0(%argi: !quantum.bit, %argq: !quantum.bit):
    // gates before the inner ctrl
    %pre = quantum.custom "Hadamard"() %argq : !quantum.bit
    %ic, %iq = quantum.ctrl(%argi) ctrlvals(%true) (%pre) : !quantum.bit -> !quantum.bit {
    ^bb1(%iiq: !quantum.bit):
      %z = quantum.custom "PauliZ"() %iiq : !quantum.bit
      quantum.yield %z : !quantum.bit
    }
    // gate after the inner ctrl
    %post = quantum.custom "T"() %iq : !quantum.bit
    quantum.yield %ic, %post : !quantum.bit, !quantum.bit
  }
  return %or#0, %or#1, %oc : !quantum.bit, !quantum.bit, !quantum.bit
}

// -----

// A measurement inside a ctrl region is rejected by the verifier.
func.func @ctrl_measure(%ctrl: !quantum.bit, %q: !quantum.bit) -> !quantum.bit {
  %true = arith.constant true
  // expected-error @+1 {{quantum measurements are not allowed in the ctrl regions}}
  %outc, %outq = quantum.ctrl(%ctrl) ctrlvals(%true) (%q) : !quantum.bit -> !quantum.bit {
  ^bb0(%arg0: !quantum.bit):
    %m, %new = quantum.measure %arg0 : i1, !quantum.bit
    quantum.yield %new : !quantum.bit
  }
  return %outc : !quantum.bit
}

// -----

// Region-bearing control flow outside the supported set (scf.if / scf.for / scf.while /
// scf.index_switch) has no rule for controlling its body, so it is rejected rather than silently
// left uncontrolled. scf.execute_region stands in for any such op here.
func.func @ctrl_unsupported_scf(%ctrl: !quantum.bit, %q: !quantum.bit) -> !quantum.bit {
  %true = arith.constant true
  %outc, %outq = quantum.ctrl(%ctrl) ctrlvals(%true) (%q) : !quantum.bit -> !quantum.bit {
  ^bb0(%arg0: !quantum.bit):
    // expected-error @+1 {{unsupported scf operation inside a quantum.ctrl region}}
    %r = scf.execute_region -> !quantum.bit {
      %h = quantum.custom "Hadamard"() %arg0 : !quantum.bit
      scf.yield %h : !quantum.bit
    }
    quantum.yield %r : !quantum.bit
  }
  return %outc : !quantum.bit
}
