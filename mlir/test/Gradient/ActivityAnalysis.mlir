// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt %s --lower-gradients=print-activity --split-input-file -o /dev/null 2>&1 | FileCheck %s

// CHECK-LABEL: Activity for '@noControlFlow' [0]:
// CHECK-DAG: "x": Active
func.func @noControlFlow(%arg0: f64 {activity.id = "x"}) -> f64 attributes {qnode, diff_method = "parameter-shift"} {
    %c0_i64 = arith.constant 0 : i64
    quantum.device ["backend", "lightning.qubit"]
    %qreg = quantum.alloc(1) : !quantum.reg
    %qbit = quantum.extract %qreg[%c0_i64] : !quantum.reg -> !quantum.bit
    // CHECK-DAG: "x^2": Active
    %mul = arith.mulf %arg0, %arg0 {activity.id = "x^2"} : f64
    // CHECK-DAG: "rx^2": Active
    %rx = quantum.custom "RX"(%mul) %qbit {activity.id = "rx^2"} : !quantum.bit
    // CHECK-DAG: "qreg0": Active
    %qreg0 = quantum.insert %qreg[%c0_i64], %rx {activity.id = "qreg0"} : !quantum.reg, !quantum.bit
    // CHECK-DAG: "qbit0": Active
    %qbit0 = quantum.extract %qreg0[%c0_i64] {activity.id = "qbit0"} : !quantum.reg -> !quantum.bit
    // CHECK-DAG: "obs": Active
    %obs = quantum.namedobs %qbit0[PauliZ] {activity.id = "obs"} : !quantum.obs
    // CHECK-DAG: "expval": Active
    %expval = quantum.expval %obs {activity.id = "expval"} : f64
    quantum.dealloc %qreg : !quantum.reg
    return %expval : f64
}

// TODO: test cases
// - if statements: both branches, only one branch

func.func @gradCallNoControlFlow(%arg0: f64) -> f64 {
    %0 = gradient.grad "defer" @noControlFlow(%arg0) : (f64) -> f64
    func.return %0 : f64
}

// -----

// CHECK-LABEL: Activity for '@tensorTypes' [0]:
// CHECK-DAG: "x": Active
func.func @tensorTypes(%arg0: tensor<3xf64> {activity.id = "x"}) -> tensor<3xf64> attributes {qnode, diff_method = "parameter-shift"} {
    // CHECK-DAG: "res": Active
    %res = math.sin %arg0 {activity.id = "res"} : tensor<3xf64>
    return %res : tensor<3xf64>
}

func.func @gradCallTensorTypes(%arg0: tensor<3xf64>) -> tensor<3x3xf64> {
    %0 = gradient.grad "defer" @tensorTypes(%arg0) : (tensor<3xf64>) -> tensor<3x3xf64>
    return %0 : tensor<3x3xf64>
}

// -----

// CHECK-LABEL: Activity for '@secondArg' [1]:
// CHECK-DAG: "x": Constant
// CHECK-DAG: "y": Active
func.func @secondArg(%arg0: f64 {activity.id = "x"}, %arg1: f64 {activity.id = "y"}) -> f64 attributes {qnode, diff_method = "parameter-shift"} {
    return %arg1 : f64
}

func.func @gradCallSecondArg(%arg0: f64, %arg1: f64) -> f64 {
    %0 = gradient.grad "defer" @secondArg(%arg0, %arg1) {diffArgIndices = dense<[1]> : tensor<1xindex>} : (f64, f64) -> f64
    return %0 : f64
}

// -----

func.func private @inner(%arg0: f64) -> f64 {
    %sin = math.sin %arg0 : f64
    return %sin : f64
}

// CHECK-LABEL: Activity for '@funcCall' [0]:
// CHECK-DAG: "x": Active
// CHECK-DAG: "call": Active
func.func @funcCall(%arg0: f64 {activity.id = "x"}) -> f64 attributes {qnode, diff_method = "parameter-shift"} {
    %call = call @inner(%arg0) {activity.id = "call"} : (f64) -> f64
    return %call : f64
}

func.func @gradFuncCall(%arg0: f64) -> f64 {
    %0 = gradient.grad "defer" @funcCall(%arg0) : (f64) -> f64
    return %0 : f64
}

// -----

func.func private @inner(%arg0: f64) -> f64 {
    %cst = arith.constant 0.0 : f64
    return %cst : f64
}

// CHECK-LABEL: Activity for '@funcCallConst' [0]:
// CHECK-DAG: "x": Constant
// CHECK-DAG: "call": Constant
func.func @funcCallConst(%arg0: f64 {activity.id = "x"}) -> f64 attributes {qnode, diff_method = "parameter-shift"} {
    %call = call @inner(%arg0) {activity.id = "call"} : (f64) -> f64
    return %call : f64
}

func.func @gradFuncCallConst(%arg0: f64) -> f64 {
    %0 = gradient.grad "defer" @funcCallConst(%arg0) : (f64) -> f64
    return %0 : f64
}

// -----

// CHECK-LABEL: Activity for '@loop' [0]:
// CHECK-DAG: "loop_res": Active
func.func @loop(%arg0: f64) -> f64 attributes {qnode, diff_method = "parameter-shift"} {
    %c0 = index.constant 0
    %c1 = index.constant 1
    %res = scf.for %iv = %c0 to %c1 step %c1 iter_args(%it = %arg0) -> f64 {
        %it_next = arith.mulf %it, %it : f64
        scf.yield %it_next : f64
    } {activity.id = "loop_res"}
    return %res : f64
}

func.func @gradLoop(%arg0: f64) -> f64 {
    %0 = gradient.grad "defer" @loop(%arg0) : (f64) -> f64
    return %0 : f64
}

// -----

// CHECK-LABEL: Activity for '@funcMultiArg' [0, 1]:
// CHECK-DAG: "arg0": Active
// CHECK-DAG: "arg1": Constant

// CHECK-LABEL: Activity for '@funcMultiArg' [1]:
// CHECK-DAG: "arg0": Constant
// CHECK-DAG: "arg1": Constant

// CHECK-LABEL: Activity for '@funcMultiArg' [0]:
// CHECK-DAG: "arg0": Active
// CHECK-DAG: "arg1": Constant
func.func @funcMultiArg(%arg0: tensor<f64> {activity.id = "arg0"}, %arg1: tensor<2xf64> {activity.id = "arg1"}) -> tensor<f64> attributes {qnode, diff_method = "parameter-shift"} {
    func.return %arg0 : tensor<f64>
}

func.func @gradCallMultiArg(%arg0: tensor<f64>, %arg1: tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>, tensor<f64>, tensor<2xf64>)  {
    %0 = gradient.grad "defer"  @funcMultiArg(%arg0, %arg1) : (tensor<f64>, tensor<2xf64>) -> tensor<f64>
    %1 = gradient.grad "defer"  @funcMultiArg(%arg0, %arg1) {diffArgIndices = dense<[1]> : tensor<1xindex>} : (tensor<f64>, tensor<2xf64>) -> tensor<2xf64>
    %2:2 = gradient.grad "defer" @funcMultiArg(%arg0, %arg1) {diffArgIndices = dense<[0, 1]> : tensor<2xindex>} : (tensor<f64>, tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>)
    func.return %0, %1, %2#0, %2#1 : tensor<f64>, tensor<2xf64>, tensor<f64>, tensor<2xf64>
}

// -----

// CHECK-LABEL: Activity for '@if' [0]:
// CHECK-DAG: "if_res": Active
func.func @if(%arg0: f64, %p: i1) -> f64 attributes {qnode, diff_method = "parameter-shift"} {
    %cst = arith.constant 4.0 : f64
    %res = scf.if %p -> f64 {
        scf.yield %arg0 : f64
    } else {
        scf.yield %cst : f64
    } {activity.id = "if_res"}
    return %res : f64
}

func.func @gradIf(%arg0: f64, %p: i1) -> f64 {
    %0 = gradient.grad "defer" @if(%arg0, %p) : (f64, i1) -> f64
    return %0 : f64
}
