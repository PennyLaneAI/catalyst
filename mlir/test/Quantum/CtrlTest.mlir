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

// -----

// Region-bearing ops that are not from scf must be accepted

// CHECK-LABEL: @ctrl_with_region_op_non_scf
func.func @ctrl_with_region_op_non_scf(%ctrl: !quantum.bit, %q: !quantum.bit, %arg2: tensor<f64>, %arg3: tensor<i32>) -> !quantum.bit {
  %true = arith.constant true
  %outc, %outq = quantum.ctrl(%ctrl) ctrlvals(%true) (%q) : !quantum.bit -> !quantum.bit {
  ^bb0(%arg0: !quantum.bit):

    // CHECK: "stablehlo.case"
    %case = "stablehlo.case"(%arg3) ({
      %neg = stablehlo.negate %arg2 : tensor<f64>
      stablehlo.return %neg : tensor<f64>
    }, {
      stablehlo.return %arg2 : tensor<f64>
    }) : (tensor<i32>) -> (tensor<f64>)

    %0 = tensor.extract %case[] : tensor<f64>

    // CHECK: RX
    // CHECK-SAME: ctrls
    %rx = quantum.custom "RX"(%0) %arg0 : !quantum.bit
    quantum.yield %rx : !quantum.bit
  }
  return %outc : !quantum.bit
}

// -----

// Work wires turn the controlled gate into quantum.operator. The reserved ctrl_work_wires group
// is distinct from the target wires and the work-wire type becomes static identity data.

// CHECK-LABEL: @ctrl_with_work_wire
func.func @ctrl_with_work_wire(%ctrl: !quantum.bit, %work: !quantum.bit, %q: !quantum.bit)
    -> (!quantum.bit, !quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK: %[[Q:.*]]:2, %[[C:.*]] = quantum.operator "PauliX"() qubits(%{{.*}}, %{{.*}})
  // CHECK-NEXT: ctrls(%{{.*}}) ctrl_vals(%[[TRUE:.*]])
  // CHECK: static_data = {ctrl_work_wire_type = "zeroed"}
  // CHECK: qubit_map = {ctrl_work_wires = [1], wires = [0]}
  // CHECK: return %[[C]], %[[Q]]#1, %[[Q]]#0
  %out_ctrl, %out_work, %out_q = quantum.ctrl(%ctrl) ctrlvals(%true)
      work_wires(%work) (%q) {work_wire_type = "zeroed"}
      : !quantum.bit work_wires !quantum.bit -> !quantum.bit {
  ^bb0(%arg: !quantum.bit):
    %x = quantum.custom "PauliX"() %arg : !quantum.bit
    quantum.yield %x : !quantum.bit
  }
  return %out_ctrl, %out_work, %out_q : !quantum.bit, !quantum.bit, !quantum.bit
}

// -----

// An operator's generic work_wires register remains separate when control work wires are added.

// CHECK-LABEL: @ctrl_preserves_generic_work_register
func.func @ctrl_preserves_generic_work_register(
    %ctrl: !quantum.bit, %ctrl_work: !quantum.bit, %q: !quantum.bit,
    %generic_work: !quantum.bit) -> (!quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK: %[[Q:.*]]:3, %[[C:.*]] = quantum.operator "MultiControlledX"()
  // CHECK-SAME: qubits(%{{.*}}, %{{.*}}, %{{.*}})
  // CHECK-NEXT: ctrls(%{{.*}}) ctrl_vals(%{{.*}})
  // CHECK: static_data = {ctrl_work_wire_type = "borrowed", work_wire_type = "zeroed"}
  // CHECK: qubit_map = {ctrl_work_wires = [2], wires = [0], work_wires = [1]}
  %out_ctrl, %out_ctrl_work, %out_q:2 = quantum.ctrl(%ctrl) ctrlvals(%true)
      work_wires(%ctrl_work) (%q, %generic_work)
      : !quantum.bit work_wires !quantum.bit -> !quantum.bit, !quantum.bit {
  ^bb0(%arg_q: !quantum.bit, %arg_work: !quantum.bit):
    %r:2 = quantum.operator "MultiControlledX"() qubits(%arg_q, %arg_work)
        static_data = {work_wire_type = "zeroed"}
        param_map = {} qubit_map = {wires = [0], work_wires = [1]}
    quantum.yield %r#0, %r#1 : !quantum.bit, !quantum.bit
  }
  return %out_ctrl, %out_ctrl_work, %out_q#0, %out_q#1
      : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
}

// -----

// Register-mode operators cannot be controlled: index arrays into a shared register cannot absorb
// the individually-valued control/work qubits a quantum.ctrl region threads through.

func.func @ctrl_register_mode_operator_with_work_wire(
    %ctrl: !quantum.bit, %work: !quantum.bit, %reg: !quantum.reg, %idx: tensor<1xi64>)
    -> (!quantum.bit, !quantum.bit, !quantum.reg) {
  %true = arith.constant true
  %out_ctrl, %out_work, %out_reg = quantum.ctrl(%ctrl) ctrlvals(%true)
      work_wires(%work) (%reg) {work_wire_type = "zeroed"}
      : !quantum.bit work_wires !quantum.bit -> !quantum.reg {
  ^bb0(%arg: !quantum.reg):
    // expected-error @+1 {{cannot control a register-mode quantum operation inside a quantum.ctrl region}}
    %r = quantum.operator "MyOp"() quregs(%arg) indices(%idx : tensor<1xi64>)
        qubit_map = {wires = [0]}
    quantum.yield %r : !quantum.reg
  }
  return %out_ctrl, %out_work, %out_reg : !quantum.bit, !quantum.bit, !quantum.reg
}

// -----

// The same register-mode restriction applies when the region has no work wires.

func.func @ctrl_register_mode_operator_without_work_wire(
    %ctrl: !quantum.bit, %reg: !quantum.reg, %idx: tensor<1xi64>)
    -> (!quantum.bit, !quantum.reg) {
  %true = arith.constant true
  %out_ctrl, %out_reg = quantum.ctrl(%ctrl) ctrlvals(%true) (%reg)
      : !quantum.bit -> !quantum.reg {
  ^bb0(%arg: !quantum.reg):
    // expected-error @+1 {{cannot control a register-mode quantum operation inside a quantum.ctrl region}}
    %r = quantum.operator "MyOp"() quregs(%arg) indices(%idx : tensor<1xi64>)
        qubit_map = {wires = [0]}
    quantum.yield %r : !quantum.reg
  }
  return %out_ctrl, %out_reg : !quantum.bit, !quantum.reg
}

// -----

// Nested control regions merge both work registers. Borrowed wins when the types differ.

// CHECK-LABEL: @nested_ctrl_work_wires
func.func @nested_ctrl_work_wires(
    %outer: !quantum.bit, %outer_work: !quantum.bit, %inner: !quantum.bit,
    %inner_work: !quantum.bit, %q: !quantum.bit)
    -> (!quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK: quantum.operator "PauliZ"() qubits(%{{.*}}, %{{.*}}, %{{.*}})
  // CHECK-NEXT: ctrls(%{{.*}}, %{{.*}}) ctrl_vals(%{{.*}}, %{{.*}})
  // CHECK: static_data = {ctrl_work_wire_type = "borrowed"}
  // CHECK: qubit_map = {ctrl_work_wires = [1, 2], wires = [0]}
  %oc, %ow, %or:3 = quantum.ctrl(%outer) ctrlvals(%true) work_wires(%outer_work)
      (%inner, %inner_work, %q) {work_wire_type = "zeroed"}
      : !quantum.bit work_wires !quantum.bit -> !quantum.bit, !quantum.bit, !quantum.bit {
  ^bb0(%ic: !quantum.bit, %iw: !quantum.bit, %iq: !quantum.bit):
    %nic, %niw, %nq = quantum.ctrl(%ic) ctrlvals(%true) work_wires(%iw) (%iq)
        : !quantum.bit work_wires !quantum.bit -> !quantum.bit {
    ^bb1(%arg: !quantum.bit):
      %z = quantum.custom "PauliZ"() %arg : !quantum.bit
      quantum.yield %z : !quantum.bit
    }
    quantum.yield %nic, %niw, %nq : !quantum.bit, !quantum.bit, !quantum.bit
  }
  return %oc, %ow, %or#0, %or#1, %or#2
      : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
}

// -----

// Forwarded arguments must stay separate from params when adding work wires.

// CHECK-LABEL: @ctrl_work_wires_forward_args(
// CHECK-SAME: %[[CTRL:[^:]+]]: !quantum.bit, %[[WORK:[^:]+]]: !quantum.bit, %[[Q:[^:]+]]: !quantum.bit, %[[P:[^:]+]]: f64, %[[F:[^:]+]]: f64
func.func @ctrl_work_wires_forward_args(
    %ctrl: !quantum.bit, %work: !quantum.bit, %q: !quantum.bit, %p: f64, %f: f64)
    -> (!quantum.bit, !quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK: %[[OQ:.*]]:2, %[[OC:.*]] = quantum.operator "MyOp"(%[[P]]: f64) qubits(%[[Q]], %[[WORK]])
  // CHECK-NEXT: UID(123) forward(%[[F]]: f64)
  // CHECK-NEXT: ctrls(%[[CTRL]]) ctrl_vals(%{{.*}})
  // CHECK: param_map = {theta = [0]} qubit_map = {ctrl_work_wires = [1], wires = [0]}
  // CHECK: return %[[OC]], %[[OQ]]#1, %[[OQ]]#0
  %oc, %ow, %oq = quantum.ctrl(%ctrl) ctrlvals(%true) work_wires(%work) (%q)
      : !quantum.bit work_wires !quantum.bit -> !quantum.bit {
  ^bb0(%arg: !quantum.bit):
    %r = quantum.operator "MyOp"(%p: f64) qubits(%arg) UID(123) forward(%f: f64)
        param_map = {theta = [0]} qubit_map = {wires = [0]}
    quantum.yield %r : !quantum.bit
  }
  return %oc, %ow, %oq : !quantum.bit, !quantum.bit, !quantum.bit
}

// -----

// Omitted maps are valid; adding work wires must also account for the original targets.

// CHECK-LABEL: @ctrl_work_wires_without_maps
func.func @ctrl_work_wires_without_maps(
    %ctrl: !quantum.bit, %work: !quantum.bit, %q: !quantum.bit)
    -> (!quantum.bit, !quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK: quantum.operator "MyOp"() qubits(%{{.*}}, %{{.*}})
  // CHECK-NOT: param_map
  // CHECK: qubit_map = {ctrl_work_wires = [1], wires = [0]}
  %oc, %ow, %oq = quantum.ctrl(%ctrl) ctrlvals(%true) work_wires(%work) (%q)
      : !quantum.bit work_wires !quantum.bit -> !quantum.bit {
  ^bb0(%arg: !quantum.bit):
    %r = quantum.operator "MyOp"() qubits(%arg)
    quantum.yield %r : !quantum.bit
  }
  return %oc, %ow, %oq : !quantum.bit, !quantum.bit, !quantum.bit
}

// -----

// An absent param_map must not become an explicitly empty map for a parameterized operator.

// CHECK-LABEL: @ctrl_work_wires_without_param_map
func.func @ctrl_work_wires_without_param_map(
    %ctrl: !quantum.bit, %work: !quantum.bit, %q: !quantum.bit, %p: f64)
    -> (!quantum.bit, !quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK: quantum.operator "MyOp"(%{{.*}}: f64) qubits(%{{.*}}, %{{.*}})
  // CHECK-NOT: param_map
  // CHECK: qubit_map = {ctrl_work_wires = [1], wires = [0]}
  %oc, %ow, %oq = quantum.ctrl(%ctrl) ctrlvals(%true) work_wires(%work) (%q)
      : !quantum.bit work_wires !quantum.bit -> !quantum.bit {
  ^bb0(%arg: !quantum.bit):
    %r = quantum.operator "MyOp"(%p: f64) qubits(%arg) qubit_map = {wires = [0]}
    quantum.yield %r : !quantum.bit
  }
  return %oc, %ow, %oq : !quantum.bit, !quantum.bit, !quantum.bit
}
