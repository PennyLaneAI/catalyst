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

// RUN: quantum-opt --remove-global-phases --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @remove_simple_global_phases
func.func @remove_simple_global_phases(%arg0: f64) {

    // CHECK-NOT: quantum.gphase
    quantum.gphase(%arg0)

    return
}

// -----

// CHECK-LABEL: func.func @remove_multiple_global_phases
func.func @remove_multiple_global_phases(%arg0: f64, %arg1: f64) {

    // CHECK-NOT: quantum.gphase
    quantum.gphase(%arg0)
    quantum.gphase(%arg1)

    return
}

// -----

// CHECK-LABEL: func.func @keep_only_controlled_global_phases(
// CHECK-SAME: [[CTRL:%.+]]: !quantum.bit, [[Q:%.+]]: !quantum.bit, [[THETA:%.+]]: f64) -> (!quantum.bit, !quantum.bit)
func.func @keep_only_controlled_global_phases(%ctrl: !quantum.bit, %q: !quantum.bit, %theta: f64) -> (!quantum.bit, !quantum.bit){

    // CHECK: [[TRUE:%.+]] = arith.constant true
    %true = arith.constant true

    // CHECK: [[OUTC:%.+]], [[OUTQ:%.+]] = quantum.ctrl([[CTRL]]) ctrlvals([[TRUE]]) ([[Q]]) : !quantum.bit -> !quantum.bit
    %outc, %outq = quantum.ctrl(%ctrl) ctrlvals(%true) (%q) : !quantum.bit -> !quantum.bit {
        ^bb0(%arg0: !quantum.bit):
            %h = quantum.custom "Hadamard"() %arg0 : !quantum.bit
            // CHECK: quantum.gphase([[THETA]])
            quantum.gphase(%theta)
            quantum.yield %h : !quantum.bit
    }

    // CHECK-NOT: quantum.gphase
    quantum.gphase(%theta)

    // CHECK: quantum.gphase([[THETA]]) ctrls([[OUTQ]]) ctrlvals([[TRUE]]) : ctrls !quantum.bit
    %gphase = quantum.gphase(%theta) ctrls(%outq) ctrlvals(%true) : ctrls !quantum.bit

    return %outq, %outc : !quantum.bit, !quantum.bit
}

// -----
