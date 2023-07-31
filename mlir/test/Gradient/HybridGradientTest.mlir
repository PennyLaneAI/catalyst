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

// RUN: quantum-opt %s --lower-gradients=only=ps  | FileCheck %s


// Check point tensor to point tensor
func.func private @funcPointTensorPointTensor(%arg0: tensor<f64>) -> tensor<f64> attributes {qnode, diff_method = "parameter-shift"} {
    return %arg0 : tensor<f64>
}

// CHECK-LABEL: @funcPointTensorPointTensor.fullgrad0ps(%arg0: tensor<f64>) -> tensor<f64> {
   // CHECK:    [[PCOUNT:%.+]] = call @funcPointTensorPointTensor.pcount(%arg0) : (tensor<f64>) -> index
   // CHECK:    [[QGRAD:%.+]] = call @funcPointTensorPointTensor.qgrad(%arg0, [[PCOUNT]]) : (tensor<f64>, index) -> tensor<?xf64>
   // CHECK:    gradient.backprop @funcPointTensorPointTensor.argmap(%arg0) cotangents([[QGRAD]] : tensor<?xf64>) : (tensor<f64>) -> tensor<f64>
// }

func.func @gradCallPointTensorPointTensor(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = gradient.grad "defer" @funcPointTensorPointTensor(%arg0) : (tensor<f64>) -> tensor<f64>
    func.return %0 : tensor<f64>
}

// -----


// Check tensor to tensor case
func.func private @funcTensorTensor(%arg0: tensor<7x3x2x1xf64>) -> tensor<2xf64> attributes {qnode, diff_method = "parameter-shift"} {
    %c0 = arith.constant 0.0 : f64
    %res = tensor.from_elements %c0, %c0 : tensor<2xf64>
    return %res : tensor<2xf64>
}

// CHECK-LABEL:  @funcTensorTensor.fullgrad0ps(%arg0: tensor<7x3x2x1xf64>) -> tensor<7x3x2x1x2xf64> {
   // CHECK:     %idx0 = index.constant 0
   // CHECK:     [[PCOUNT:%.+]] = call @funcTensorTensor.pcount(%arg0) : (tensor<7x3x2x1xf64>) -> index
   // CHECK:     [[QGRAD:%.+]] = call @funcTensorTensor.qgrad(%arg0, %0) : (tensor<7x3x2x1xf64>, index) -> tensor<?x2xf64>
   // CHECK:     [[EMPTYTENSOR:%.+]] = tensor.empty() : tensor<7x3x2x1x2xf64>
   // CHECK:     [[DIM:%.+]] = tensor.dim [[QGRAD]], %idx0 : tensor<?x2xf64>
   // CHECK:     [[EXTRACTEDQGRAD0:%.+]] = tensor.extract_slice [[QGRAD]][0, 0] [[[DIM]], 1] [1, 1] : tensor<?x2xf64> to tensor<?xf64>
   // CHECK:     [[GRAD0:%.+]] = gradient.backprop @funcTensorTensor.argmap(%arg0) cotangents([[EXTRACTEDQGRAD0]] : tensor<?xf64>) : (tensor<7x3x2x1xf64>) -> tensor<7x3x2x1xf64>
   // CHECK:     [[EXTRACTEDQGRAD1:%.+]] = tensor.extract_slice %1[0, 1] [[[DIM]], 1] [1, 1] : tensor<?x2xf64> to tensor<?xf64>
   // CHECK:     [[GRAD1:%.+]] = gradient.backprop @funcTensorTensor.argmap(%arg0) cotangents([[EXTRACTEDQGRAD1]] : tensor<?xf64>) : (tensor<7x3x2x1xf64>) -> tensor<7x3x2x1xf64>
   // CHECK:     [[INSERTQGRAD0:%.+]] = tensor.insert_slice [[GRAD0]] into [[EMPTYTENSOR]][0, 0, 0, 0, 0] [7, 3, 2, 1, 1] [1, 1, 1, 1, 1] : tensor<7x3x2x1xf64> into tensor<7x3x2x1x2xf64>
   // CHECK:     [[INSERTQGRAD1:%.+]] = tensor.insert_slice [[GRAD1]] into [[INSERTQGRAD0]][0, 0, 0, 0, 1] [7, 3, 2, 1, 1] [1, 1, 1, 1, 1] : tensor<7x3x2x1xf64> into tensor<7x3x2x1x2xf64>
   // CHECK:     return [[INSERTQGRAD1]] : tensor<7x3x2x1x2xf64>
// }
func.func @gradCallTensorTensor(%arg0: tensor<7x3x2x1xf64>) -> tensor<7x3x2x1x2xf64> {
    %2 = gradient.grad "defer" @funcTensorTensor(%arg0) : (tensor<7x3x2x1xf64>) -> tensor<7x3x2x1x2xf64>
    func.return %2 : tensor<7x3x2x1x2xf64>
}

// -----

// Check the multiple arguments case
func.func @funcMultiArg(%arg0: tensor<f64>, %arg1: tensor<2xf64>) -> tensor<f64> attributes {qnode, diff_method = "parameter-shift"} {
    func.return %arg0 : tensor<f64>
}

// CHECK-LABEL:  @funcMultiArg.fullgrad0ps(%arg0: tensor<f64>, %arg1: tensor<2xf64>) -> tensor<f64> {
   // CHECK:    [[PCOUNT:%.+]] = call @funcMultiArg.pcount(%arg0, %arg1) : (tensor<f64>, tensor<2xf64>) -> index
   // CHECK:    [[QGRAD:%.+]] = call @funcMultiArg.qgrad(%arg0, %arg1, [[PCOUNT]]) : (tensor<f64>, tensor<2xf64>, index) -> tensor<?xf64>
   // CHECK:    [[GRAD:%.+]] = gradient.backprop @funcMultiArg.argmap(%arg0, %arg1) cotangents([[QGRAD]] : tensor<?xf64>) : (tensor<f64>, tensor<2xf64>) -> tensor<f64>
// }

// CHECK-LABEL:  func.func private @funcMultiArg.fullgrad1ps(%arg0: tensor<f64>, %arg1: tensor<2xf64>) -> tensor<2xf64> {
   // CHECK:    [[PCOUNT:%.+]] = call @funcMultiArg.pcount(%arg0, %arg1) : (tensor<f64>, tensor<2xf64>) -> index
   // CHECK:    [[QGRAD:%.+]] = call @funcMultiArg.qgrad(%arg0, %arg1, [[PCOUNT]]) : (tensor<f64>, tensor<2xf64>, index) -> tensor<?xf64>
   // CHECK:    [[GRAD:%.+]] = gradient.backprop @funcMultiArg.argmap(%arg0, %arg1) cotangents([[QGRAD]] : tensor<?xf64>) {diffArgIndices = dense<1> : tensor<1xindex>} : (tensor<f64>, tensor<2xf64>) -> tensor<2xf64>
// }

// CHECK-LABEL:  @funcMultiArg.fullgrad01ps(%arg0: tensor<f64>, %arg1: tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>) {
   // CHECK:    [[PCOUNT:%.+]] = call @funcMultiArg.pcount(%arg0, %arg1) : (tensor<f64>, tensor<2xf64>) -> index
   // CHECK:    [[QGRAD:%.+]] = call @funcMultiArg.qgrad(%arg0, %arg1, [[PCOUNT]]) : (tensor<f64>, tensor<2xf64>, index) -> tensor<?xf64>
   // CHECK:    [[GRAD:%.+]]:2 = gradient.backprop @funcMultiArg.argmap(%arg0, %arg1) cotangents([[QGRAD]] : tensor<?xf64>) {diffArgIndices = dense<[0, 1]> : tensor<2xindex>} : (tensor<f64>, tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>)
// }

func.func @gradCallMultiArg(%arg0: tensor<f64>, %arg1: tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>, tensor<f64>, tensor<2xf64>)  {
    %0 = gradient.grad "defer"  @funcMultiArg(%arg0, %arg1) : (tensor<f64>, tensor<2xf64>) -> tensor<f64>
    %1 = gradient.grad "defer"  @funcMultiArg(%arg0, %arg1) {diffArgIndices = dense<[1]> : tensor<1xindex>} : (tensor<f64>, tensor<2xf64>) -> tensor<2xf64>
    %2:2 = gradient.grad "defer" @funcMultiArg(%arg0, %arg1) {diffArgIndices = dense<[0, 1]> : tensor<2xindex>} : (tensor<f64>, tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>)
    func.return %0, %1, %2#0, %2#1 : tensor<f64>, tensor<2xf64>, tensor<f64>, tensor<2xf64>
}
