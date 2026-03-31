// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --compact-conditional-ppr %s | FileCheck %s

// CHECK-LABEL: func.func @fold_ppr_if
func.func @fold_ppr_if(%q : !quantum.bit, %if_cond : i1) -> !quantum.bit {
    // CHECK-NOT: scf.if
    // CHECK: %[[OUT:.*]] = pbc.ppr ["X"](4) %{{.*}} cond(%[[IFCOND:.*]]) : !quantum.bit
    // CHECK: return %[[OUT]] : !quantum.bit
    %q_out = scf.if %if_cond -> (!quantum.bit) {
        %q_then = pbc.ppr ["X"](4) %q : !quantum.bit
        scf.yield %q_then : !quantum.bit
    } else {
        scf.yield %q : !quantum.bit
    }

    func.return %q_out : !quantum.bit
}

// -----

// CHECK-LABEL: func.func @fold_ppr_arbitrary_if
func.func @fold_ppr_arbitrary_if(%q : !quantum.bit, %if_cond : i1, %theta : f64) -> !quantum.bit {
    // CHECK-NOT: scf.if
    // CHECK: %[[OUT:.*]] = pbc.ppr.arbitrary ["Y"](%[[THETA:.*]]) %{{.*}} cond(%[[IFCOND:.*]]) : !quantum.bit
    // CHECK: return %[[OUT]] : !quantum.bit
    %q_out = scf.if %if_cond -> (!quantum.bit) {
        %q_then = pbc.ppr.arbitrary ["Y"](%theta) %q : !quantum.bit
        scf.yield %q_then : !quantum.bit
    } else {
        scf.yield %q : !quantum.bit
    }

    func.return %q_out : !quantum.bit
}

// -----

// CHECK-LABEL: func.func @combine_with_existing_cond
func.func @combine_with_existing_cond(%q : !quantum.bit, %if_cond : i1, %op_cond : i1) -> !quantum.bit {
    // CHECK: %[[COND:.*]] = arith.andi %[[IFCOND:.*]], %[[OPCOND:.*]] : i1
    // CHECK: %[[OUT:.*]] = pbc.ppr ["Z"](8) %{{.*}} cond(%[[COND]]) : !quantum.bit
    // CHECK-NOT: scf.if
    %q_out = scf.if %if_cond -> (!quantum.bit) {
        %q_then = pbc.ppr ["Z"](8) %q cond(%op_cond) : !quantum.bit
        scf.yield %q_then : !quantum.bit
    } else {
        scf.yield %q : !quantum.bit
    }

    func.return %q_out : !quantum.bit
}

// -----

// CHECK-LABEL: func.func @combine_arbitrary_with_existing_cond
func.func @combine_arbitrary_with_existing_cond(%q : !quantum.bit, %if_cond : i1, %op_cond : i1,
                                                %theta : f64) -> !quantum.bit {
    // CHECK: %[[COND:.*]] = arith.andi %[[IFCOND:.*]], %[[OPCOND:.*]] : i1
    // CHECK: %[[OUT:.*]] = pbc.ppr.arbitrary ["X"](%[[THETA:.*]]) %{{.*}} cond(%[[COND]]) : !quantum.bit
    // CHECK-NOT: scf.if
    %q_out = scf.if %if_cond -> (!quantum.bit) {
        %q_then = pbc.ppr.arbitrary ["X"](%theta) %q cond(%op_cond) : !quantum.bit
        scf.yield %q_then : !quantum.bit
    } else {
        scf.yield %q : !quantum.bit
    }

    func.return %q_out : !quantum.bit
}
