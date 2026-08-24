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
  // CHECK-NOT: quantum.ctrl
  // CHECK: scf.if %{{.*}} -> (!quantum.bit, !quantum.bit) {
  // CHECK:   quantum.custom "Hadamard"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}})
  // CHECK:   scf.yield %{{.*}}, %{{.*}} : !quantum.bit, !quantum.bit
  // CHECK: } else {
  // CHECK:   scf.yield %{{.*}}, %{{.*}} : !quantum.bit, !quantum.bit
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
  // CHECK: scf.if %{{.*}} -> (!quantum.bit, !quantum.bit) {
  // CHECK:   quantum.custom "Hadamard"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}})
  // CHECK: } else {
  // CHECK:   quantum.custom "PauliX"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}})
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
  // CHECK: %[[PRE:.*]], %[[PREC:.*]] = quantum.custom "S"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}})
  // CHECK: %[[IF:.*]]:2 = scf.if %{{.*}} -> (!quantum.bit, !quantum.bit) {
  // CHECK:   quantum.custom "Hadamard"() %[[PRE]] ctrls(%[[PREC]]) ctrlvals(%{{.*}})
  // CHECK: } else {
  // CHECK:   scf.yield %[[PRE]], %[[PREC]] : !quantum.bit, !quantum.bit
  // CHECK: }
  // CHECK: quantum.custom "T"() %[[IF]]#0 ctrls(%[[IF]]#1) ctrlvals(%{{.*}})
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
  // CHECK-NOT: quantum.ctrl
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (!quantum.bit, !quantum.bit)
  // CHECK:   quantum.custom "Hadamard"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}})
  // CHECK:   scf.yield %{{.*}}, %{{.*}} : !quantum.bit, !quantum.bit
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
  // CHECK: %[[SO:.*]], %[[SC:.*]] = quantum.custom "S"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}})
  // CHECK: %[[F:.*]]:2 = scf.for %{{.*}} iter_args(%{{[^ ]*}} = %[[SO]], %{{[^ ]*}} = %[[SC]]) -> (!quantum.bit, !quantum.bit)
  // CHECK: quantum.custom "T"() %[[F]]#0 ctrls(%[[F]]#1) ctrlvals(%{{.*}})
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
  // CHECK: scf.for %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (!quantum.bit, !quantum.bit)
  // CHECK:   scf.if %{{.*}} -> (!quantum.bit, !quantum.bit)
  // CHECK:     quantum.custom "Hadamard"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}})
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
  // CHECK-NOT: quantum.ctrl
  // CHECK: scf.while ({{.*}}) : (i64, !quantum.bit, !quantum.bit) -> (i64, !quantum.bit, !quantum.bit)
  // CHECK:   scf.condition(%{{.*}}) %{{.*}}, %{{.*}}, %{{.*}} : i64, !quantum.bit, !quantum.bit
  // CHECK: } do {
  // CHECK:   quantum.custom "Hadamard"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}})
  // CHECK:   scf.yield %{{.*}}, %{{.*}}, %{{.*}} : i64, !quantum.bit, !quantum.bit
  // CHECK: }
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
  // CHECK: scf.index_switch %{{.*}} -> !quantum.bit, !quantum.bit
  // CHECK: case 0 {
  // CHECK:   quantum.custom "Hadamard"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}})
  // CHECK:   scf.yield %{{.*}}, %{{.*}} : !quantum.bit, !quantum.bit
  // CHECK: }
  // CHECK: case 1 {
  // CHECK:   quantum.custom "PauliX"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}})
  // CHECK: }
  // CHECK: default {
  // CHECK:   scf.yield %{{.*}}, %{{.*}} : !quantum.bit, !quantum.bit
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
