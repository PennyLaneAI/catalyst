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

// RUN: quantum-opt --ctrl-lowering --split-input-file -verify-diagnostics %s | FileCheck %s --implicit-check-not="!quantum.bit" --implicit-check-not="!quantum.reg"

// CHECK-LABEL: @ctrl_single_gate
// CHECK-NOT: qref.ctrl
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: qref.custom "Hadamard"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%[[TRUE]]) : !qref.bit ctrls !qref.bit

module {
  func.func @ctrl_single_gate(%arg0: !qref.bit, %arg1: !qref.bit) {
    %true = arith.constant true
    qref.ctrl(%arg0) ctrlvals(%true) {
      qref.custom "Hadamard"() %arg1 : !qref.bit
    }
    return
  }
}

// -----

// CHECK-LABEL: @ctrl_two_gates
// CHECK: qref.custom "Hadamard"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}})
// CHECK: qref.custom "PauliX"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}})

module {
  func.func @ctrl_two_gates(%arg0: !qref.bit, %arg1: !qref.bit) {
    %true = arith.constant true
    qref.ctrl(%arg0) ctrlvals(%true) {
      qref.custom "Hadamard"() %arg1 : !qref.bit
      qref.custom "PauliX"() %arg1 : !qref.bit
    }
    return
  }
}

// -----

// CHECK-LABEL: @ctrl_merge_existing(
// CHECK-SAME:  %[[CTRL:[^:]+]]: !qref.bit, %[[INNER:[^:]+]]: !qref.bit, %[[Q:[^:]+]]: !qref.bit
// CHECK-DAG: %[[TRUE:.*]] = arith.constant true
// CHECK-DAG: %[[FALSE:.*]] = arith.constant false
// CHECK: qref.custom "PauliX"() %[[Q]] ctrls(%[[INNER]], %[[CTRL]]) ctrlvals(%[[TRUE]], %[[FALSE]]) : !qref.bit ctrls !qref.bit, !qref.bit

module {
  func.func @ctrl_merge_existing(%arg0: !qref.bit, %arg1: !qref.bit, %arg2: !qref.bit) {
    %true = arith.constant true
    %false = arith.constant false
    qref.ctrl(%arg0) ctrlvals(%false) {
      qref.custom "PauliX"() %arg2 ctrls(%arg1) ctrlvals(%true) : !qref.bit ctrls !qref.bit
    }
    return
  }
}

// -----

// CHECK-LABEL: @ctrl_duplicate
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: qref.custom "PauliX"() %{{.*}} ctrls(%{{.*}}, %{{.*}}) ctrlvals(%[[TRUE]], %[[TRUE]]) : !qref.bit ctrls !qref.bit, !qref.bit

module {
  func.func @ctrl_duplicate(%arg0: !qref.bit, %arg1: !qref.bit, %arg2: !qref.bit) {
    %true = arith.constant true
    qref.ctrl(%arg0) ctrlvals(%true) {
      qref.custom "PauliX"() %arg2 ctrls(%arg1) ctrlvals(%true) : !qref.bit ctrls !qref.bit
    }
    return
  }
}

// -----

// CHECK-LABEL: @ctrl_zero_value_not_expanded
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK-NOT: qref.custom "PauliX"
// CHECK: qref.custom "Hadamard"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%[[FALSE]])
// CHECK-NOT: qref.custom "PauliX"

module {
  func.func @ctrl_zero_value_not_expanded(%arg0: !qref.bit, %arg1: !qref.bit) {
    %false = arith.constant false
    qref.ctrl(%arg0) ctrlvals(%false) {
      qref.custom "Hadamard"() %arg1 : !qref.bit
    }
    return
  }
}

// -----

// CHECK-LABEL: @ctrl_adjoint_gate
// CHECK: qref.custom "S"() %{{.*}} adj ctrls(%{{.*}}) ctrlvals(%{{.*}})

module {
  func.func @ctrl_adjoint_gate(%arg0: !qref.bit, %arg1: !qref.bit) {
    %true = arith.constant true
    qref.ctrl(%arg0) ctrlvals(%true) {
      qref.custom "S"() %arg1 adj : !qref.bit
    }
    return
  }
}

// -----

// CHECK-LABEL: @ctrl_over_register
// CHECK: %[[Q:.*]] = qref.get %{{.*}}[ 0]
// CHECK: qref.custom "Hadamard"() %[[Q]] ctrls(%{{.*}}) ctrlvals(%{{.*}})

module {
  func.func @ctrl_over_register(%arg0: !qref.bit, %arg1: !qref.reg<?>) {
    %true = arith.constant true
    qref.ctrl(%arg0) ctrlvals(%true) {
      %0 = qref.get %arg1[ 0] : !qref.reg<?> -> !qref.bit
      qref.custom "Hadamard"() %0 : !qref.bit
    }
    return
  }
}

// -----

// CHECK-LABEL: @ctrl_nested
// CHECK-NOT: qref.ctrl
// CHECK: qref.custom "PauliZ"() %{{.*}} ctrls(%{{.*}}, %{{.*}}) ctrlvals(%{{.*}}, %{{.*}})

module {
  func.func @ctrl_nested(%arg0: !qref.bit, %arg1: !qref.bit, %arg2: !qref.bit) {
    %true = arith.constant true
    qref.ctrl(%arg0) ctrlvals(%true) {
      qref.ctrl(%arg1) ctrlvals(%true) {
        qref.custom "PauliZ"() %arg2 : !qref.bit
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @ctrl_nested_gate_before_after
// CHECK-NOT: qref.ctrl
// CHECK: qref.custom "Hadamard"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}})
// CHECK: qref.custom "PauliZ"() %{{.*}} ctrls(%{{.*}}, %{{.*}}) ctrlvals(%{{.*}}, %{{.*}})
// CHECK: qref.custom "T"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}})
// gates before the inner ctrl
// gate after the inner ctrl

module {
  func.func @ctrl_nested_gate_before_after(%arg0: !qref.bit, %arg1: !qref.bit, %arg2: !qref.bit) {
    %true = arith.constant true
    qref.ctrl(%arg0) ctrlvals(%true) {
      qref.custom "Hadamard"() %arg2 : !qref.bit
      qref.ctrl(%arg1) ctrlvals(%true) {
        qref.custom "PauliZ"() %arg2 : !qref.bit
      }
      qref.custom "T"() %arg2 : !qref.bit
    }
    return
  }
}

// -----

// Measurements cannot be controlled.
func.func @ctrl_measure(%ctrl: !qref.bit, %q: !qref.bit) {
  %true = arith.constant true
  // expected-error @+1 {{quantum measurements are not allowed in the ctrl regions}}
  qref.ctrl(%ctrl) ctrlvals(%true) {
    %m = qref.measure %q : i1
  }
  return
}

// -----

// A gate must not repeat a control already supplied by its region.
func.func @ctrl_repeated_control(%ctrl: !qref.bit, %q: !qref.bit) {
  %true = arith.constant true
  // expected-error @+1 {{Ctrl op ctrl qubits must not reappear as the individual gates' ctrl qubits inside the ctrl region}}
  qref.ctrl(%ctrl) ctrlvals(%true) {
    qref.custom "PauliX"() %q ctrls(%ctrl) ctrlvals(%true) : !qref.bit ctrls !qref.bit
  }
  return
}

// -----

// Reference-semantics controls can be distributed into an scf.execute_region directly.
// CHECK-LABEL: @ctrl_execute_region
// CHECK: scf.execute_region
// CHECK: qref.custom "Hadamard"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}})
func.func @ctrl_execute_region(%ctrl: !qref.bit, %q: !qref.bit) {
  %true = arith.constant true
  qref.ctrl(%ctrl) ctrlvals(%true) {
    scf.execute_region {
      qref.custom "Hadamard"() %q : !qref.bit
      scf.yield
    }
  }
  return
}

// -----

// A call acting on qubits has no rule for controlling the callee, so it is rejected rather than
// silently left uncontrolled.

module {
  func.func private @rotate(%arg0: !qref.bit, %arg1: f64) {
    qref.custom "RZ"(%arg1) %arg0 : !qref.bit
    return
  }
  func.func @ctrl_call(%arg0: !qref.bit, %arg1: !qref.bit, %arg2: f64) {
    %true = arith.constant true
    qref.ctrl(%arg0) ctrlvals(%true) {
    // expected-error @+1 {{cannot control a function call inside a qref.ctrl region}}
      func.call @rotate(%arg1, %arg2) : (!qref.bit, f64) -> ()
    }
    return
  }
}

// -----

// Classical calls inside a ctrl region are cloned.
// CHECK-LABEL: @ctrl_classical_call
// CHECK-NOT: qref.ctrl
// CHECK: [[Y:%.+]] = call @double
// CHECK: qref.custom "RX"([[Y]]) {{.*}} ctrls

module {
  func.func private @double(f64) -> f64
  func.func @ctrl_classical_call(%arg0: !qref.bit, %arg1: !qref.bit, %arg2: f64) {
    %true = arith.constant true
    qref.ctrl(%arg0) ctrlvals(%true) {
      %0 = func.call @double(%arg2) : (f64) -> f64
      qref.custom "RX"(%0) %arg1 : !qref.bit
    }
    return
  }
}

// -----

// Region-bearing ops that are not from scf must be accepted
// CHECK-LABEL: @ctrl_with_region_op_non_scf
// CHECK: "stablehlo.case"
// CHECK: RX
// CHECK-SAME: ctrls

module {
  func.func @ctrl_with_region_op_non_scf(%arg0: !qref.bit, %arg1: !qref.bit, %arg2: tensor<f64>, %arg3: tensor<i32>) {
    %true = arith.constant true
    qref.ctrl(%arg0) ctrlvals(%true) {
      %0 = "stablehlo.case"(%arg3) ({
        %1 = stablehlo.negate %arg2 : tensor<f64>
        stablehlo.return %1 : tensor<f64>
      }, {
        stablehlo.return %arg2 : tensor<f64>
      }) : (tensor<i32>) -> tensor<f64>
      %extracted = tensor.extract %0[] : tensor<f64>
      qref.custom "RX"(%extracted) %arg1 : !qref.bit
    }
    return
  }
}

// -----

// A global-phase callee is quantum even without reference arguments or results.
func.func private @phase(%theta: f64) {
  qref.gphase(%theta)
  return
}

func.func @ctrl_phase_call(%ctrl: !qref.bit, %theta: f64) {
  %true = arith.constant true
  qref.ctrl(%ctrl) ctrlvals(%true) {
    // expected-error @+1 {{cannot control a function call inside a qref.ctrl region}}
    func.call @phase(%theta) : (f64) -> ()
  }
  return
}
