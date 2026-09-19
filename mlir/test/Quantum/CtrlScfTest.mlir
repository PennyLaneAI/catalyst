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

// RUN: quantum-opt --ctrl-lowering --split-input-file %s | FileCheck %s

// CHECK-LABEL: @ctrl_scf_if_then_only
func.func @ctrl_scf_if_then_only(%ctrl: !quantum.bit, %q: !quantum.bit, %cond: i1)
    -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK-NOT: qref.ctrl
  // CHECK: scf.if %{{.*}} {
  // CHECK:   qref.custom "Hadamard"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}}) : !qref.bit ctrls !qref.bit
  // CHECK: }
  %outc, %outq = quantum.ctrl(%ctrl) ctrlvals(%true) (%q) : !quantum.bit -> !quantum.bit {
  ^bb0(%arg0: !quantum.bit):
    %r = scf.if %cond -> !quantum.bit {
      %h = quantum.custom "Hadamard"() %arg0 : !quantum.bit
      scf.yield %h : !quantum.bit
    } else {
      scf.yield %arg0 : !quantum.bit
    }
    quantum.yield %r : !quantum.bit
  }
  return %outc, %outq : !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: @ctrl_scf_if_both_branches
func.func @ctrl_scf_if_both_branches(%ctrl: !quantum.bit, %q: !quantum.bit, %cond: i1)
    -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK: scf.if %{{.*}} {
  // CHECK:   qref.custom "Hadamard"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}}) : !qref.bit ctrls !qref.bit
  // CHECK: } else {
  // CHECK:   qref.custom "PauliX"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}}) : !qref.bit ctrls !qref.bit
  // CHECK: }
  %outc, %outq = quantum.ctrl(%ctrl) ctrlvals(%true) (%q) : !quantum.bit -> !quantum.bit {
  ^bb0(%arg0: !quantum.bit):
    %r = scf.if %cond -> !quantum.bit {
      %h = quantum.custom "Hadamard"() %arg0 : !quantum.bit
      scf.yield %h : !quantum.bit
    } else {
      %x = quantum.custom "PauliX"() %arg0 : !quantum.bit
      scf.yield %x : !quantum.bit
    }
    quantum.yield %r : !quantum.bit
  }
  return %outc, %outq : !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: @ctrl_scf_if_threaded
func.func @ctrl_scf_if_threaded(%ctrl: !quantum.bit, %q: !quantum.bit, %cond: i1)
    -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK: qref.custom "S"() %[[PRE:.*]] ctrls(%[[PREC:.*]]) ctrlvals(%{{.*}}) : !qref.bit ctrls !qref.bit
  // CHECK: scf.if %{{.*}} {
  // CHECK:   qref.custom "Hadamard"() %[[PRE]] ctrls(%[[PREC]]) ctrlvals(%{{.*}}) : !qref.bit ctrls !qref.bit
  // CHECK: }
  // CHECK: qref.custom "T"() %[[PRE]] ctrls(%[[PREC]]) ctrlvals(%{{.*}}) : !qref.bit ctrls !qref.bit
  %outc, %outq = quantum.ctrl(%ctrl) ctrlvals(%true) (%q) : !quantum.bit -> !quantum.bit {
  ^bb0(%arg0: !quantum.bit):
    %s = quantum.custom "S"() %arg0 : !quantum.bit
    %r = scf.if %cond -> !quantum.bit {
      %h = quantum.custom "Hadamard"() %s : !quantum.bit
      scf.yield %h : !quantum.bit
    } else {
      scf.yield %s : !quantum.bit
    }
    %t = quantum.custom "T"() %r : !quantum.bit
    quantum.yield %t : !quantum.bit
  }
  return %outc, %outq : !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: @ctrl_scf_for_basic
func.func @ctrl_scf_for_basic(%ctrl: !quantum.bit, %q: !quantum.bit, %lb: index, %ub: index,
                              %step: index) -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK-NOT: qref.ctrl
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
  // CHECK:   qref.custom "Hadamard"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}}) : !qref.bit ctrls !qref.bit
  %outc, %outq = quantum.ctrl(%ctrl) ctrlvals(%true) (%q) : !quantum.bit -> !quantum.bit {
  ^bb0(%arg0: !quantum.bit):
    %r = scf.for %i = %lb to %ub step %step iter_args(%qi = %arg0) -> !quantum.bit {
      %h = quantum.custom "Hadamard"() %qi : !quantum.bit
      scf.yield %h : !quantum.bit
    }
    quantum.yield %r : !quantum.bit
  }
  return %outc, %outq : !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: @ctrl_scf_for_threaded
func.func @ctrl_scf_for_threaded(%ctrl: !quantum.bit, %q: !quantum.bit, %lb: index, %ub: index,
                                 %step: index) -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // The pre-gate's target/control seed the loop's iter_args; the loop's results feed the post-gate.
  // CHECK: qref.custom "S"() %[[SO:.*]] ctrls(%[[SC:.*]]) ctrlvals(%[[C:.*]]) : !qref.bit ctrls !qref.bit
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{[^ ]*}} step %{{.*}} {
  // CHECK:   qref.custom "Hadamard"() %[[SO]] ctrls(%[[SC]]) ctrlvals(%[[C]]) : !qref.bit ctrls !qref.bit
  // CHECK: }
  // CHECK: qref.custom "T"() %[[SO]] ctrls(%[[SC]]) ctrlvals(%[[C]])
  %outc, %outq = quantum.ctrl(%ctrl) ctrlvals(%true) (%q) : !quantum.bit -> !quantum.bit {
  ^bb0(%arg0: !quantum.bit):
    %s = quantum.custom "S"() %arg0 : !quantum.bit
    %r = scf.for %i = %lb to %ub step %step iter_args(%qi = %s) -> !quantum.bit {
      %h = quantum.custom "Hadamard"() %qi : !quantum.bit
      scf.yield %h : !quantum.bit
    }
    %t = quantum.custom "T"() %r : !quantum.bit
    quantum.yield %t : !quantum.bit
  }
  return %outc, %outq : !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: @ctrl_scf_for_with_if
func.func @ctrl_scf_for_with_if(%ctrl: !quantum.bit, %q: !quantum.bit, %lb: index, %ub: index,
                                %step: index, %cond: i1) -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
  // CHECK:   scf.if %{{.*}} {
  // CHECK:     qref.custom "Hadamard"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}}) : !qref.bit ctrls !qref.bit
  %outc, %outq = quantum.ctrl(%ctrl) ctrlvals(%true) (%q) : !quantum.bit -> !quantum.bit {
  ^bb0(%arg0: !quantum.bit):
    %r = scf.for %i = %lb to %ub step %step iter_args(%qi = %arg0) -> !quantum.bit {
      %ri = scf.if %cond -> !quantum.bit {
        %h = quantum.custom "Hadamard"() %qi : !quantum.bit
        scf.yield %h : !quantum.bit
      } else {
        scf.yield %qi : !quantum.bit
      }
      scf.yield %ri : !quantum.bit
    }
    quantum.yield %r : !quantum.bit
  }
  return %outc, %outq : !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: @ctrl_scf_while
func.func @ctrl_scf_while(%ctrl: !quantum.bit, %q: !quantum.bit, %n: i64) -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  // CHECK-NOT: qref.ctrl
  // CHECK: scf.while ({{.*}}) : (i64) -> i64 {
  // CHECK:   scf.condition(%{{.*}}) %{{.*}} : i64
  // CHECK: } do {
  // CHECK:   qref.custom "Hadamard"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}}) : !qref.bit ctrls !qref.bit
  %outc, %outq = quantum.ctrl(%ctrl) ctrlvals(%true) (%q) : !quantum.bit -> !quantum.bit {
  ^bb0(%arg0: !quantum.bit):
    %r:2 = scf.while (%i = %c0, %qi = %arg0) : (i64, !quantum.bit) -> (i64, !quantum.bit) {
      %cond = arith.cmpi slt, %i, %n : i64
      scf.condition(%cond) %i, %qi : i64, !quantum.bit
    } do {
    ^bb0(%i2: i64, %q2: !quantum.bit):
      %h = quantum.custom "Hadamard"() %q2 : !quantum.bit
      %inext = arith.addi %i2, %c1 : i64
      scf.yield %inext, %h : i64, !quantum.bit
    }
    quantum.yield %r#1 : !quantum.bit
  }
  return %outc, %outq : !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: @ctrl_scf_index_switch
func.func @ctrl_scf_index_switch(%ctrl: !quantum.bit, %q: !quantum.bit, %idx: index)
    -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK-NOT: quantum.ctrl
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
  %outc, %outq = quantum.ctrl(%ctrl) ctrlvals(%true) (%q) : !quantum.bit -> !quantum.bit {
  ^bb0(%arg0: !quantum.bit):
    %r = scf.index_switch %idx -> !quantum.bit
    case 0 {
      %h = quantum.custom "Hadamard"() %arg0 : !quantum.bit
      scf.yield %h : !quantum.bit
    }
    case 1 {
      %x = quantum.custom "PauliX"() %arg0 : !quantum.bit
      scf.yield %x : !quantum.bit
    }
    default {
      scf.yield %arg0 : !quantum.bit
    }
    quantum.yield %r : !quantum.bit
  }
  return %outc, %outq : !quantum.bit, !quantum.bit
}
