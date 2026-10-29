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

// RUN: quantum-opt --ctrl-lowering --split-input-file %s | FileCheck %s --implicit-check-not="!quantum.bit" --implicit-check-not="!quantum.reg"

// CHECK-LABEL: @ctrl_scf_if_then_only
// CHECK-NOT: qref.ctrl
// CHECK: scf.if %{{.*}} {
// CHECK:   qref.custom "Hadamard"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}}) : !qref.bit ctrls !qref.bit
// CHECK: }

module {
  func.func @ctrl_scf_if_then_only(%arg0: !qref.bit, %arg1: !qref.bit, %arg2: i1) {
    %true = arith.constant true
    qref.ctrl(%arg0) ctrlvals(%true) {
      scf.if %arg2 {
        qref.custom "Hadamard"() %arg1 : !qref.bit
      }
    }
    return
  }
}


// -----

// CHECK-LABEL: @ctrl_scf_if_both_branches
// CHECK: scf.if %{{.*}} {
// CHECK:   qref.custom "Hadamard"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}}) : !qref.bit ctrls !qref.bit
// CHECK: } else {
// CHECK:   qref.custom "PauliX"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}}) : !qref.bit ctrls !qref.bit
// CHECK: }

module {
  func.func @ctrl_scf_if_both_branches(%arg0: !qref.bit, %arg1: !qref.bit, %arg2: i1) {
    %true = arith.constant true
    qref.ctrl(%arg0) ctrlvals(%true) {
      scf.if %arg2 {
        qref.custom "Hadamard"() %arg1 : !qref.bit
      } else {
        qref.custom "PauliX"() %arg1 : !qref.bit
      }
    }
    return
  }
}


// -----

// CHECK-LABEL: @ctrl_scf_if_threaded
// CHECK: qref.custom "S"() %[[PRE:.*]] ctrls(%[[PREC:.*]]) ctrlvals(%{{.*}}) : !qref.bit ctrls !qref.bit
// CHECK: scf.if %{{.*}} {
// CHECK:   qref.custom "Hadamard"() %[[PRE]] ctrls(%[[PREC]]) ctrlvals(%{{.*}}) : !qref.bit ctrls !qref.bit
// CHECK: }
// CHECK: qref.custom "T"() %[[PRE]] ctrls(%[[PREC]]) ctrlvals(%{{.*}}) : !qref.bit ctrls !qref.bit

module {
  func.func @ctrl_scf_if_threaded(%arg0: !qref.bit, %arg1: !qref.bit, %arg2: i1) {
    %true = arith.constant true
    qref.ctrl(%arg0) ctrlvals(%true) {
      qref.custom "S"() %arg1 : !qref.bit
      scf.if %arg2 {
        qref.custom "Hadamard"() %arg1 : !qref.bit
      }
      qref.custom "T"() %arg1 : !qref.bit
    }
    return
  }
}


// -----

// CHECK-LABEL: @ctrl_scf_for_basic
// CHECK-NOT: qref.ctrl
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:   qref.custom "Hadamard"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}}) : !qref.bit ctrls !qref.bit

module {
  func.func @ctrl_scf_for_basic(%arg0: !qref.bit, %arg1: !qref.bit, %arg2: index, %arg3: index, %arg4: index) {
    %true = arith.constant true
    qref.ctrl(%arg0) ctrlvals(%true) {
      scf.for %arg5 = %arg2 to %arg3 step %arg4 {
        qref.custom "Hadamard"() %arg1 : !qref.bit
      }
    }
    return
  }
}


// -----

// CHECK-LABEL: @ctrl_scf_for_threaded
// The pre-gate's target/control seed the loop's iter_args; the loop's results feed the post-gate.
// CHECK: qref.custom "S"() %[[SO:.*]] ctrls(%[[SC:.*]]) ctrlvals(%[[C:.*]]) : !qref.bit ctrls !qref.bit
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{[^ ]*}} step %{{.*}} {
// CHECK:   qref.custom "Hadamard"() %[[SO]] ctrls(%[[SC]]) ctrlvals(%[[C]]) : !qref.bit ctrls !qref.bit
// CHECK: }
// CHECK: qref.custom "T"() %[[SO]] ctrls(%[[SC]]) ctrlvals(%[[C]])

module {
  func.func @ctrl_scf_for_threaded(%arg0: !qref.bit, %arg1: !qref.bit, %arg2: index, %arg3: index, %arg4: index) {
    %true = arith.constant true
    qref.ctrl(%arg0) ctrlvals(%true) {
      qref.custom "S"() %arg1 : !qref.bit
      scf.for %arg5 = %arg2 to %arg3 step %arg4 {
        qref.custom "Hadamard"() %arg1 : !qref.bit
      }
      qref.custom "T"() %arg1 : !qref.bit
    }
    return
  }
}


// -----

// CHECK-LABEL: @ctrl_scf_for_with_if
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:   scf.if %{{.*}} {
// CHECK:     qref.custom "Hadamard"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}}) : !qref.bit ctrls !qref.bit

module {
  func.func @ctrl_scf_for_with_if(%arg0: !qref.bit, %arg1: !qref.bit, %arg2: index, %arg3: index, %arg4: index, %arg5: i1) {
    %true = arith.constant true
    qref.ctrl(%arg0) ctrlvals(%true) {
      scf.for %arg6 = %arg2 to %arg3 step %arg4 {
        scf.if %arg5 {
          qref.custom "Hadamard"() %arg1 : !qref.bit
        }
      }
    }
    return
  }
}


// -----

// CHECK-LABEL: @ctrl_scf_while
// CHECK-NOT: qref.ctrl
// CHECK: scf.while ({{.*}}) : (i64) -> i64 {
// CHECK:   scf.condition(%{{.*}}) %{{.*}} : i64
// CHECK: } do {
// CHECK:   qref.custom "Hadamard"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}}) : !qref.bit ctrls !qref.bit

module {
  func.func @ctrl_scf_while(%arg0: !qref.bit, %arg1: !qref.bit, %arg2: i64) {
    %true = arith.constant true
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    qref.ctrl(%arg0) ctrlvals(%true) {
      %0 = scf.while (%arg3 = %c0_i64) : (i64) -> i64 {
        %1 = arith.cmpi slt, %arg3, %arg2 : i64
        scf.condition(%1) %arg3 : i64
      } do {
      ^bb0(%arg3: i64):
        qref.custom "Hadamard"() %arg1 : !qref.bit
        %1 = arith.addi %arg3, %c1_i64 : i64
        scf.yield %1 : i64
      }
    }
    return
  }
}


// -----

// CHECK-LABEL: @ctrl_scf_index_switch
// CHECK-NOT: qref.ctrl
// CHECK: scf.index_switch %{{.*}}
// CHECK: case 0 {
// CHECK:   qref.custom "Hadamard"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}}) : !qref.bit ctrls !qref.bit
// CHECK:   scf.yield
// CHECK: }
// CHECK: case 1 {
// CHECK:   qref.custom "PauliX"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}}) : !qref.bit ctrls !qref.bit
// CHECK: }
// CHECK: default {
// CHECK: }

module {
  func.func @ctrl_scf_index_switch(%arg0: !qref.bit, %arg1: !qref.bit, %arg2: index) {
    %true = arith.constant true
    qref.ctrl(%arg0) ctrlvals(%true) {
      scf.index_switch %arg2
      case 0 {
        qref.custom "Hadamard"() %arg1 : !qref.bit
        scf.yield
      }
      case 1 {
        qref.custom "PauliX"() %arg1 : !qref.bit
        scf.yield
      }
      default {
      }
    }
    return
  }
}


// -----

// Resource hints on SCF ops must survive ctrl-lowering recreation.
// CHECK-LABEL: @ctrl_scf_preserves_hints
// CHECK-NOT: qref.ctrl
// CHECK: scf.for
// CHECK: } {catalyst.estimated_iterations = 1.000000e+01 : f64}
// CHECK: scf.while
// CHECK: } attributes {catalyst.estimated_iterations = 2.500000e+00 : f64}
// CHECK: scf.if
// CHECK: } {catalyst.estimated_probability = 7.500000e-01 : f64}
// CHECK: scf.index_switch {{.+}} {catalyst.estimated_probabilities = [2.000000e-01, 3.000000e-01]}

module {
  func.func @ctrl_scf_preserves_hints(%arg0: !qref.bit, %arg1: !qref.bit, %arg2: index, %arg3: index, %arg4: index, %arg5: i1, %arg6: i64, %arg7: index) {
    %true = arith.constant true
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    qref.ctrl(%arg0) ctrlvals(%true) {
      scf.for %arg8 = %arg2 to %arg3 step %arg4 {
        qref.custom "Hadamard"() %arg1 : !qref.bit
      } {catalyst.estimated_iterations = 1.000000e+01 : f64}
      %0 = scf.while (%arg8 = %c0_i64) : (i64) -> i64 {
        %1 = arith.cmpi slt, %arg8, %arg6 : i64
        scf.condition(%1) %arg8 : i64
      } do {
      ^bb0(%arg8: i64):
        qref.custom "PauliX"() %arg1 : !qref.bit
        %1 = arith.addi %arg8, %c1_i64 : i64
        scf.yield %1 : i64
      } attributes {catalyst.estimated_iterations = 2.500000e+00 : f64}
      scf.if %arg5 {
        qref.custom "PauliY"() %arg1 : !qref.bit
      } {catalyst.estimated_probability = 7.500000e-01 : f64}
      scf.index_switch %arg7 {catalyst.estimated_probabilities = [2.000000e-01, 3.000000e-01]}
      case 0 {
        qref.custom "PauliZ"() %arg1 : !qref.bit
        scf.yield
      }
      case 1 {
        scf.yield
      }
      default {
      }
    }
    return
  }
}

