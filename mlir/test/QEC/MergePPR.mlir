// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --merge-ppr --split-input-file -verify-diagnostics %s | FileCheck %s
// RUN: quantum-opt --merge-ppr="max-pauli-size=3" --split-input-file -verify-diagnostics %s | FileCheck %s

////////////////////////////////////// Single gate merge tests /////////////////////////////////////

func.func public @merge_ppr_X_pi_2(%q1: !quantum.bit) -> tensor<i1> {
    // CHECK-NOT: qec.ppr ["X"](2)
    // CHECK: qec.ppr ["X"]
    %0 = qec.ppr ["X"](2) %q1: !quantum.bit
    %0 = qec.ppr ["X"](2) %q1: !quantum.bit
}

func.func public @merge_ppr_X_pi_4(%q1: !quantum.bit) -> tensor<i1> {
    // CHECK-NOT: qec.ppr ["X"](4)
    // CHECK: qec.ppr ["X"](2)
    %0 = qec.ppr ["X"](4) %q1: !quantum.bit
    %0 = qec.ppr ["X"](4) %q1: !quantum.bit
}

func.func public @merge_ppr_X_pi_8(%q1: !quantum.bit) -> tensor<i1> {
    // CHECK-NOT: qec.ppr ["X"](8)
    // CHECK: qec.ppr ["X"](4)
    %0 = qec.ppr ["X"](8) %q1: !quantum.bit
    %0 = qec.ppr ["X"](8) %q1: !quantum.bit
}


func.func public @merge_ppr_Y_pi_2(%q1: !quantum.bit) -> tensor<i1> {
    // CHECK-NOT: qec.ppr ["Y"](2)
    // CHECK: qec.ppr ["Y"]
    %0 = qec.ppr ["Y"](2) %q1: !quantum.bit
    %0 = qec.ppr ["Y"](2) %q1: !quantum.bit
}

func.func public @merge_ppr_Y_pi_4(%q1: !quantum.bit) -> tensor<i1> {
    // CHECK-NOT: qec.ppr ["Y"](4)
    // CHECK: qec.ppr ["Y"](2)
    %0 = qec.ppr ["Y"](4) %q1: !quantum.bit
    %0 = qec.ppr ["Y"](4) %q1: !quantum.bit
}

func.func public @merge_ppr_Y_pi_8(%q1: !quantum.bit) -> tensor<i1> {
    // CHECK-NOT: qec.ppr ["Y"](8)
    // CHECK: qec.ppr ["Y"](4)
    %0 = qec.ppr ["Y"](8) %q1: !quantum.bit
    %0 = qec.ppr ["Y"](8) %q1: !quantum.bit
}

func.func public @merge_ppr_Z_pi_2(%q1: !quantum.bit) -> tensor<i1> {
    // CHECK-NOT: qec.ppr ["Z"](2)
    // CHECK: qec.ppr ["Z"]
    %0 = qec.ppr ["Z"](2) %q1: !quantum.bit
    %0 = qec.ppr ["Z"](2) %q1: !quantum.bit
}

func.func public @merge_ppr_Z_pi_4(%q1: !quantum.bit) -> tensor<i1> {
    // CHECK-NOT: qec.ppr ["Z"](4)
    // CHECK: qec.ppr ["Z"](2)
    %0 = qec.ppr ["Z"](4) %q1: !quantum.bit
    %0 = qec.ppr ["Z"](4) %q1: !quantum.bit
}

func.func public @merge_ppr_Z_pi_8(%q1: !quantum.bit) -> tensor<i1> {
    // CHECK-NOT: qec.ppr ["Z"](8)
    // CHECK: qec.ppr ["Z"](4)
    %0 = qec.ppr ["Z"](8) %q1: !quantum.bit
    %0 = qec.ppr ["Z"](8) %q1: !quantum.bit
}


//////////////////////////////////////// Merge inverse tests ///////////////////////////////////////


func.func public @merge_ppr_X_pi_2(%q1: !quantum.bit) -> tensor<i1> {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["X"](2) %q1: !quantum.bit
    %0 = qec.ppr ["X"](-2) %q1: !quantum.bit
}

func.func public @merge_ppr_X_pi_2(%q1: !quantum.bit) -> tensor<i1> {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["X"](-2) %q1: !quantum.bit
    %0 = qec.ppr ["X"](2) %q1: !quantum.bit
}

func.func public @merge_ppr_X_pi_4(%q1: !quantum.bit) -> tensor<i1> {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["X"](4) %q1: !quantum.bit
    %0 = qec.ppr ["X"](-4) %q1: !quantum.bit
}

func.func public @merge_ppr_X_pi_4(%q1: !quantum.bit) -> tensor<i1> {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["X"](-4) %q1: !quantum.bit
    %0 = qec.ppr ["X"](4) %q1: !quantum.bit
}

func.func public @merge_ppr_X_pi_8(%q1: !quantum.bit) -> tensor<i1> {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["X"](8) %q1: !quantum.bit
    %0 = qec.ppr ["X"](-8) %q1: !quantum.bit
}

func.func public @merge_ppr_X_pi_8(%q1: !quantum.bit) -> tensor<i1> {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["X"](-8) %q1: !quantum.bit
    %0 = qec.ppr ["X"](8) %q1: !quantum.bit
}

func.func public @merge_ppr_Y_pi_2(%q1: !quantum.bit) -> tensor<i1> {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["Y"](2) %q1: !quantum.bit
    %0 = qec.ppr ["Y"](-2) %q1: !quantum.bit
}

func.func public @merge_ppr_Y_pi_2(%q1: !quantum.bit) -> tensor<i1> {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["Y"](-2) %q1: !quantum.bit
    %0 = qec.ppr ["Y"](2) %q1: !quantum.bit
}

func.func public @merge_ppr_Y_pi_4(%q1: !quantum.bit) -> tensor<i1> {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["Y"](4) %q1: !quantum.bit
    %0 = qec.ppr ["Y"](-4) %q1: !quantum.bit
}

func.func public @merge_ppr_Y_pi_4(%q1: !quantum.bit) -> tensor<i1> {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["Y"](-4) %q1: !quantum.bit
    %0 = qec.ppr ["Y"](4) %q1: !quantum.bit
}

func.func public @merge_ppr_Y_pi_8(%q1: !quantum.bit) -> tensor<i1> {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["Y"](8) %q1: !quantum.bit
    %0 = qec.ppr ["Y"](-8) %q1: !quantum.bit
}

func.func public @merge_ppr_Y_pi_8(%q1: !quantum.bit) -> tensor<i1> {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["Y"](-8) %q1: !quantum.bit
    %0 = qec.ppr ["Y"](8) %q1: !quantum.bit
}

func.func public @merge_ppr_Z_pi_2(%q1: !quantum.bit) -> tensor<i1> {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["Z"](2) %q1: !quantum.bit
    %0 = qec.ppr ["Z"](-2) %q1: !quantum.bit
}

func.func public @merge_ppr_Z_pi_2(%q1: !quantum.bit) -> tensor<i1> {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["Z"](-2) %q1: !quantum.bit
    %0 = qec.ppr ["Z"](2) %q1: !quantum.bit
}

func.func public @merge_ppr_Z_pi_4(%q1: !quantum.bit) -> tensor<i1> {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["Z"](4) %q1: !quantum.bit
    %0 = qec.ppr ["Z"](-4) %q1: !quantum.bit
}

func.func public @merge_ppr_Z_pi_4(%q1: !quantum.bit) -> tensor<i1> {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["Z"](-4) %q1: !quantum.bit
    %0 = qec.ppr ["Z"](4) %q1: !quantum.bit
}

func.func public @merge_ppr_Z_pi_8(%q1: !quantum.bit) -> tensor<i1> {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["Z"](8) %q1: !quantum.bit
    %0 = qec.ppr ["Z"](-8) %q1: !quantum.bit
}

func.func public @merge_ppr_Z_pi_8(%q1: !quantum.bit) -> tensor<i1> {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["Z"](-8) %q1: !quantum.bit
    %0 = qec.ppr ["Z"](8) %q1: !quantum.bit
}
