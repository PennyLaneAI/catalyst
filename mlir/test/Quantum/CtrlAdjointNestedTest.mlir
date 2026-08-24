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

// Nested quantum.ctrl / quantum.adjoint regions are reduced innermost-out by running the two
// lowering passes as a fixpoint. Two rounds of (ctrl-lowering, adjoint-lowering) are enough for the
// nesting depths exercised here.
// RUN: quantum-opt --ctrl-lowering --adjoint-lowering --ctrl-lowering --adjoint-lowering \
// RUN:   --split-input-file %s | FileCheck %s

// CHECK-LABEL: @nested_ctrl_of_ctrl
func.func @nested_ctrl_of_ctrl(%c1: !quantum.bit, %c2: !quantum.bit, %q: !quantum.bit)
    -> (!quantum.bit, !quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK-NOT: quantum.ctrl
  // CHECK: quantum.custom "Hadamard"() %{{.*}} ctrls(%{{.*}}, %{{.*}}) ctrlvals(%{{.*}}, %{{.*}})
  %oc, %or:2 = quantum.ctrl(%c1) ctrlvals(%true) (%c2, %q) : !quantum.bit -> !quantum.bit, !quantum.bit {
  ^bb0(%ac2: !quantum.bit, %aq: !quantum.bit):
    %ic, %iq = quantum.ctrl(%ac2) ctrlvals(%true) (%aq) : !quantum.bit -> !quantum.bit {
    ^bb1(%iiq: !quantum.bit):
      %h = quantum.custom "Hadamard"() %iiq : !quantum.bit
      quantum.yield %h : !quantum.bit
    }
    quantum.yield %ic, %iq : !quantum.bit, !quantum.bit
  }
  return %oc, %or#0, %or#1 : !quantum.bit, !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: @nested_ctrl_of_adj
func.func @nested_ctrl_of_adj(%c: !quantum.bit, %q: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK-NOT: quantum.ctrl
  // CHECK-NOT: quantum.adjoint
  // CHECK: quantum.custom "S"() %{{.*}} adj ctrls(%{{.*}}) ctrlvals(%{{.*}})
  %oc, %or = quantum.ctrl(%c) ctrlvals(%true) (%q) : !quantum.bit -> !quantum.bit {
  ^bb0(%aq: !quantum.bit):
    %a = quantum.adjoint(%aq) : !quantum.bit {
    ^bb1(%iq: !quantum.bit):
      %s = quantum.custom "S"() %iq : !quantum.bit
      quantum.yield %s : !quantum.bit
    }
    quantum.yield %a : !quantum.bit
  }
  return %oc, %or : !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: @nested_adj_of_ctrl
func.func @nested_adj_of_ctrl(%c: !quantum.bit, %q: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK-NOT: quantum.ctrl
  // CHECK-NOT: quantum.adjoint
  // CHECK: quantum.custom "S"() %{{.*}} adj ctrls(%{{.*}}) ctrlvals(%{{.*}})
  %a:2 = quantum.adjoint(%c, %q) : !quantum.bit, !quantum.bit {
  ^bb0(%ac: !quantum.bit, %aq: !quantum.bit):
    %oc, %or = quantum.ctrl(%ac) ctrlvals(%true) (%aq) : !quantum.bit -> !quantum.bit {
    ^bb1(%iq: !quantum.bit):
      %s = quantum.custom "S"() %iq : !quantum.bit
      quantum.yield %s : !quantum.bit
    }
    quantum.yield %oc, %or : !quantum.bit, !quantum.bit
  }
  return %a#0, %a#1 : !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: @nested_ctrl_of_adj_of_ctrl
func.func @nested_ctrl_of_adj_of_ctrl(%c1: !quantum.bit, %c2: !quantum.bit, %q: !quantum.bit)
    -> (!quantum.bit, !quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK-NOT: quantum.ctrl
  // CHECK-NOT: quantum.adjoint
  // CHECK: quantum.custom "S"() %{{.*}} adj ctrls(%{{.*}}, %{{.*}}) ctrlvals(%{{.*}}, %{{.*}})
  %oc, %or:2 = quantum.ctrl(%c1) ctrlvals(%true) (%c2, %q) : !quantum.bit -> !quantum.bit, !quantum.bit {
  ^bb0(%ac2: !quantum.bit, %aq: !quantum.bit):
    %adj:2 = quantum.adjoint(%ac2, %aq) : !quantum.bit, !quantum.bit {
    ^bb1(%bc2: !quantum.bit, %bq: !quantum.bit):
      %ic, %iq = quantum.ctrl(%bc2) ctrlvals(%true) (%bq) : !quantum.bit -> !quantum.bit {
      ^bb2(%iiq: !quantum.bit):
        %s = quantum.custom "S"() %iiq : !quantum.bit
        quantum.yield %s : !quantum.bit
      }
      quantum.yield %ic, %iq : !quantum.bit, !quantum.bit
    }
    quantum.yield %adj#0, %adj#1 : !quantum.bit, !quantum.bit
  }
  return %oc, %or#0, %or#1 : !quantum.bit, !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: @nested_adj_of_ctrl_of_adj
func.func @nested_adj_of_ctrl_of_adj(%c: !quantum.bit, %q: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
  %true = arith.constant true
  // CHECK-NOT: quantum.ctrl
  // CHECK-NOT: quantum.adjoint
  // CHECK: quantum.custom "S"() %{{[^ ]+}} ctrls(%{{[^ ]+}}) ctrlvals(%{{[^ ]+}})
  %oa:2 = quantum.adjoint(%c, %q) : !quantum.bit, !quantum.bit {
  ^bb0(%ac: !quantum.bit, %aq: !quantum.bit):
    %mc, %mr = quantum.ctrl(%ac) ctrlvals(%true) (%aq) : !quantum.bit -> !quantum.bit {
    ^bb1(%iq: !quantum.bit):
      %ia = quantum.adjoint(%iq) : !quantum.bit {
      ^bb2(%iiq: !quantum.bit):
        %s = quantum.custom "S"() %iiq : !quantum.bit
        quantum.yield %s : !quantum.bit
      }
      quantum.yield %ia : !quantum.bit
    }
    quantum.yield %mc, %mr : !quantum.bit, !quantum.bit
  }
  return %oa#0, %oa#1 : !quantum.bit, !quantum.bit
}
