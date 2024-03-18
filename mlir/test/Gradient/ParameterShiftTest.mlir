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

// RUN: quantum-opt %s --lower-gradients | FileCheck %s

// Check scalar to scalar function
func.func private @funcScalarScalar(%arg0: f64) -> f64 attributes {qnode, diff_method = "parameter-shift"} {
    return %arg0 : f64
}

// CHECK-LABEL: @funcScalarScalar.qgrad(%arg0: f64, %arg1: index) -> tensor<?xf64>

// CHECK-LABEL: @funcScalarScalar.fullgrad0(%arg0: f64, %arg1: index) -> f64

// CHECK-LABEL: @gradCallScalarScalar
func.func @gradCallScalarScalar(%arg0: f64) -> f64 {
    // CHECK:   [[GRAD:%.+]] = call @funcScalarScalar.fullgrad0(%arg0
    %0 = gradient.grad "auto" @funcScalarScalar(%arg0) : (f64) -> f64

    // CHECK:   return [[GRAD]]
    func.return %0 : f64
}

// -----

// Check scalar to tensor function
func.func private @funcScalarTensor(%arg0: f64) -> tensor<2x3xf64> attributes {qnode, diff_method = "parameter-shift"} {
    %c0 = arith.constant 0.0 : f64
    %res = tensor.from_elements %c0, %c0, %c0, %c0, %c0, %c0 : tensor<2x3xf64>
    return %res : tensor<2x3xf64>
}

// CHECK-LABEL: @funcScalarTensor.qgrad(%arg0: f64, %arg1: index) -> tensor<?x2x3xf64>

// CHECK-LABEL: @funcScalarTensor.fullgrad0(%arg0: f64, %arg1: index) -> tensor<2x3xf64>

// CHECK-LABEL: @gradCallScalarTensor
func.func @gradCallScalarTensor(%arg0: f64) -> tensor<2x3xf64> {
    // CHECK:   [[jac:%.+]] = call @funcScalarTensor.fullgrad0(%arg0
    %0 = gradient.grad "auto"  @funcScalarTensor(%arg0) : (f64) -> tensor<2x3xf64>

    // CHECK:   return [[jac]]
    func.return %0 : tensor<2x3xf64>
}

// -----

// Check tensor to scalar
func.func private @funcTensorScalar(%arg0: tensor<3xf64>) -> f64 attributes {qnode, diff_method = "parameter-shift"} {
    %res = arith.constant 0.0 : f64
    return %res : f64
}

// CHECK-LABEL: @funcTensorScalar.qgrad(%arg0: tensor<3xf64>, %arg1: index) -> tensor<?xf64>

// CHECK-LABEL: @funcTensorScalar.fullgrad0(%arg0: tensor<3xf64>, %arg1: index) -> tensor<3xf64>

// CHECK-LABEL: @gradCallTensorScalar
func.func @gradCallTensorScalar(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    // CHECK:   [[GRAD:%.+]] = call @funcTensorScalar.fullgrad0(%arg0
    %2 = gradient.grad "auto"  @funcTensorScalar(%arg0) : (tensor<3xf64>) -> tensor<3xf64>

    // CHECK:   return [[GRAD]]
    func.return %2 : tensor<3xf64>
}

// -----

// Check tensor to tensor case
func.func private @funcTensorTensor(%arg0: tensor<7x3x2x1xf64>) -> tensor<2xf64> attributes {qnode, diff_method = "parameter-shift"} {
    %c0 = arith.constant 0.0 : f64
    %res = tensor.from_elements %c0, %c0 : tensor<2xf64>
    return %res : tensor<2xf64>
}

// CHECK-LABEL: @funcTensorTensor.qgrad(%arg0: tensor<7x3x2x1xf64>, %arg1: index) -> tensor<?x2xf64>

// CHECK-LABEL: @funcTensorTensor.fullgrad0(%arg0: tensor<7x3x2x1xf64>, %arg1: index) -> tensor<2x7x3x2x1xf64>

// CHECK-LABEL: @gradCallTensorTensor
func.func @gradCallTensorTensor(%arg0: tensor<7x3x2x1xf64>) -> tensor<2x7x3x2x1xf64> {
    // CHECK:   [[jac:%.+]] = call @funcTensorTensor.fullgrad0(%arg0
    %2 = gradient.grad "auto" @funcTensorTensor(%arg0) : (tensor<7x3x2x1xf64>) -> tensor<2x7x3x2x1xf64>

    // CHECK:   return [[jac]]
    func.return %2 : tensor<2x7x3x2x1xf64>
}

// -----

// Check the multiple results case
func.func @funcMultiRes(%arg0: tensor<f64>) -> (tensor<f64>, tensor<f64>) attributes {qnode, diff_method = "parameter-shift"} {
    func.return %arg0, %arg0 : tensor<f64>, tensor<f64>
}

// CHECK-LABEL: @funcMultiRes.qgrad(%arg0: tensor<f64>, %arg1: index) -> (tensor<?xf64>, tensor<?xf64>)

// CHECK-LABEL: @funcMultiRes.fullgrad0(%arg0: tensor<f64>, %arg1: index) -> (tensor<f64>, tensor<f64>)

// CHECK-LABEL: @gradCallMultiRes
func.func @gradCallMultiRes(%arg0: tensor<f64>) -> (tensor<f64>, tensor<f64>)  {
    // CHECK:   [[jac:%.+]]:2 = call @funcMultiRes.fullgrad0(%arg0
    %0:2 = gradient.grad "auto" @funcMultiRes(%arg0) : (tensor<f64>) -> (tensor<f64>, tensor<f64>)

    // CHECK:   return [[jac]]#0, [[jac]]#1
    func.return %0#0, %0#1 : tensor<f64>, tensor<f64>
}

// -----

// Check the multiple arguments case
func.func @funcMultiArg(%arg0: tensor<f64>, %arg1: tensor<2xf64>) -> tensor<f64> attributes {qnode, diff_method = "parameter-shift"} {
    func.return %arg0 : tensor<f64>
}

// CHECK-LABEL: @funcMultiArg.qgrad(%arg0: tensor<f64>, %arg1: tensor<2xf64>, %arg2: index) -> tensor<?xf64>

// CHECK-LABEL: @gradCallMultiArg
func.func @gradCallMultiArg(%arg0: tensor<f64>, %arg1: tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>, tensor<f64>, tensor<2xf64>)  {
    // CHECK:   [[GRAD0:%.+]] = call @funcMultiArg.fullgrad0(%arg0, %arg1{{.+}} -> tensor<f64>
    %0 = gradient.grad "auto"  @funcMultiArg(%arg0, %arg1) : (tensor<f64>, tensor<2xf64>) -> tensor<f64>
    // CHECK:   [[GRAD1:%.+]] = call @funcMultiArg.fullgrad1(%arg0, %arg1{{.+}} -> tensor<2xf64>
    %1 = gradient.grad "auto"  @funcMultiArg(%arg0, %arg1) {diffArgIndices = dense<[1]> : tensor<1xindex>} : (tensor<f64>, tensor<2xf64>) -> tensor<2xf64>
    // CHECK:   [[GRAD2:%.+]]:2 = call @funcMultiArg.fullgrad01(%arg0, %arg1{{.+}} -> (tensor<f64>, tensor<2xf64>)
    %2:2 = gradient.grad "auto" @funcMultiArg(%arg0, %arg1) {diffArgIndices = dense<[0, 1]> : tensor<2xindex>} : (tensor<f64>, tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>)

    // CHECK:   return [[GRAD0]], [[GRAD1]], [[GRAD2]]#0, [[GRAD2]]#1
    func.return %0, %1, %2#0, %2#1 : tensor<f64>, tensor<2xf64>, tensor<f64>, tensor<2xf64>
}

// -----

// Check multiple grad calls to same function
func.func private @funcMultiCall(%arg0: tensor<f64>) -> tensor<f64> attributes {qnode, diff_method = "parameter-shift"} {
    func.return %arg0 : tensor<f64>
}

// CHECK-LABEL: @funcMultiCall.qgrad(%arg0: tensor<f64>, %arg1: index) -> tensor<?xf64>

// CHECK-LABEL: @funcMultiCall.fullgrad0(%arg0: tensor<f64>, %arg1: index) -> tensor<f64>

// CHECK-LABEL: @gradCallMultiCall
func.func @gradCallMultiCall(%arg0: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
    // CHECK:   [[GRAD0:%.+]] = call @funcMultiCall.fullgrad0(%arg0
    %0 = gradient.grad "auto" @funcMultiCall(%arg0) : (tensor<f64>) -> tensor<f64>
    // CHECK:   [[GRAD1:%.+]] = call @funcMultiCall.fullgrad0(%arg0
    %1 = gradient.grad "auto" @funcMultiCall(%arg0) : (tensor<f64>) -> tensor<f64>

    // CHECK:   return [[GRAD0]], [[GRAD1]]
    func.return %0, %1 : tensor<f64>, tensor<f64>
}
