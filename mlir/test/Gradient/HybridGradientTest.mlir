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

// Check scalar to scalar function
func.func private @funcScalarScalar(%arg0: f64) -> f64 attributes {qnode, diff_method = "parameter-shift"} {
    return %arg0 : f64
}

// CHECK-LABEL: @funcScalarScalar.fullgrad0ps(%arg0: f64) -> f64
    // CHECK:    [[PCOUNT:%.+]] = call @funcScalarScalar.pcount(%arg0) : (f64) -> index
    // CHECK:    [[QGRAD:%.+]] = call @funcScalarScalar.qgrad(%arg0, [[PCOUNT]]) : (f64, index) -> tensor<?xf64>
    // CHECK:    gradient.backprop @funcScalarScalar.argmap(%arg0) cotangents([[QGRAD]] : tensor<?xf64>) : (f64) -> f64

func.func @gradCallScalarScalar(%arg0: f64) -> f64 {
    %0 = gradient.grad "defer" @funcScalarScalar(%arg0) : (f64) -> f64
    func.return %0 : f64
}

// -----

// Check scalar to point tensor
func.func private @funcScalarPointTensor(%arg0: f64) -> tensor<f64> attributes {qnode, diff_method = "parameter-shift"} {
    %0 = tensor.empty() : tensor<f64>
    %1 = linalg.fill ins(%arg0 : f64) outs(%0 : tensor<f64>) -> tensor<f64>
    return %1 : tensor<f64>
}

// CHECK-LABEL: @funcScalarPointTensor.fullgrad0ps(%arg0: f64) -> tensor<f64>
    // CHECK:    [[PCOUNT:%.+]] = call @funcScalarPointTensor.pcount(%arg0) : (f64) -> index
    // CHECK:    [[QGRAD:%.+]] = call @funcScalarPointTensor.qgrad(%arg0, [[PCOUNT]]) : (f64, index) -> tensor<?xf64>
    // CHECK:    gradient.backprop @funcScalarPointTensor.argmap(%arg0) cotangents([[QGRAD]] : tensor<?xf64>) : (f64) -> f64

func.func @gradCallScalarPointTensor(%arg0: f64) -> tensor<f64> {
    %0 = gradient.grad "defer" @funcScalarPointTensor(%arg0) : (f64) -> tensor<f64>
    func.return %0 : tensor<f64>
}


// -----

// Check point tensor to scalar
func.func private @funcPointTensorScalar(%arg0: tensor<f64>) -> f64 attributes {qnode, diff_method = "parameter-shift"} {
    %0 = tensor.extract %arg0[] : tensor<f64>
    return %0 : f64
}

// CHECK-LABEL: @funcPointTensorScalar.fullgrad0ps(%arg0: tensor<f64>) -> f64
    // CHECK:   [[PCOUNT:%.+]] = call @funcPointTensorScalar.pcount(%arg0) : (tensor<f64>) -> index
    // CHECK:   [[QGRAD:%.+]] = call @funcPointTensorScalar.qgrad(%arg0, [[PCOUNT]]) : (tensor<f64>, index) -> tensor<?xf64>
    // CHECK:   [[GRAD_RESHAPED:%.+]] = gradient.backprop @funcPointTensorScalar.argmap(%arg0) cotangents([[QGRAD]] : tensor<?xf64>) : (tensor<f64>) -> tensor<f64>
    // CHECK:   [[GRAD_SCALAR:%.+]] = tensor.extract [[GRAD_RESHAPED]][]
    // CHECK:   return [[GRAD_SCALAR]]

func.func @gradCallPointTensorScalar(%arg0: tensor<f64>) -> f64 {
    %0 = gradient.grad "defer" @funcPointTensorScalar(%arg0) : (tensor<f64>) -> f64
    func.return %0 : f64
}


// -----

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

// Check scalar to tensor function
func.func private @funcScalarTensor(%arg0: f64) -> tensor<2x3xf64> attributes {qnode, diff_method = "parameter-shift"} {
    %c0 = arith.constant 0.0 : f64
    %res = tensor.from_elements %c0, %c0, %c0, %c0, %c0, %c0 : tensor<2x3xf64>
    return %res : tensor<2x3xf64>
}

// CHECK-LABEL: @funcScalarTensor.fullgrad0ps(%arg0: f64) -> tensor<2x3xf64>
    // CHECK:   [[pcount:%.+]] = call @funcScalarTensor.pcount(%arg0) : (f64) -> index
    // CHECK:   [[qgrad:%.+]] = call @funcScalarTensor.qgrad(%arg0, [[pcount]]) : (f64, index) -> tensor<?x2x3xf64>
    // CHECK:   [[jacobian0:%.+]] = tensor.empty() : tensor<2x3xf64>
    // CHECK:   [[qgrad00:%.+]] = tensor.extract_slice [[qgrad]][0, 0, 0]
    // CHECK:   [[jacvl00:%.+]] = gradient.backprop @funcScalarTensor.argmap(%arg0) cotangents([[qgrad00]]
    // CHECK:   [[qgrad01:%.+]] = tensor.extract_slice [[qgrad]][0, 0, 1]
    // CHECK:   [[jacvl01:%.+]] = gradient.backprop @funcScalarTensor.argmap(%arg0) cotangents([[qgrad01]]
    // CHECK:   [[qgrad02:%.+]] = tensor.extract_slice [[qgrad]][0, 0, 2]
    // CHECK:   [[jacvl02:%.+]] = gradient.backprop @funcScalarTensor.argmap(%arg0) cotangents([[qgrad02]]
    // CHECK:   [[qgrad10:%.+]] = tensor.extract_slice [[qgrad]][0, 1, 0]
    // CHECK:   [[jacvl10:%.+]] = gradient.backprop @funcScalarTensor.argmap(%arg0) cotangents([[qgrad10]]
    // CHECK:   [[qgrad11:%.+]] = tensor.extract_slice [[qgrad]][0, 1, 1]
    // CHECK:   [[jacvl11:%.+]] = gradient.backprop @funcScalarTensor.argmap(%arg0) cotangents([[qgrad11]]
    // CHECK:   [[qgrad12:%.+]] = tensor.extract_slice [[qgrad]][0, 1, 2]
    // CHECK:   [[jacvl12:%.+]] = gradient.backprop @funcScalarTensor.argmap(%arg0) cotangents([[qgrad12]]

    // CHECK:   [[jacobian1:%.+]] = tensor.insert [[jacvl00]] into [[jacobian0]][%idx0, %idx0]
    // CHECK:   [[jacobian2:%.+]] = tensor.insert [[jacvl01]] into [[jacobian1]][%idx0, %idx1]
    // CHECK:   [[jacobian3:%.+]] = tensor.insert [[jacvl02]] into [[jacobian2]][%idx0, %idx2]
    // CHECK:   [[jacobian4:%.+]] = tensor.insert [[jacvl10]] into [[jacobian3]][%idx1, %idx0]
    // CHECK:   [[jacobian5:%.+]] = tensor.insert [[jacvl11]] into [[jacobian4]][%idx1, %idx1]
    // CHECK:   [[jacobian6:%.+]] = tensor.insert [[jacvl12]] into [[jacobian5]][%idx1, %idx2]
    // CHECK:   return [[jacobian6]]
func.func @gradCallScalarTensor(%arg0: f64) -> tensor<2x3xf64> {
    %0 = gradient.grad "defer"  @funcScalarTensor(%arg0) : (f64) -> tensor<2x3xf64>
    func.return %0 : tensor<2x3xf64>
}

// -----

// Check tensor to scalar
func.func private @funcTensorScalar(%arg0: tensor<3xf64>) -> f64 attributes {qnode, diff_method = "parameter-shift"} {
    %res = arith.constant 0.0 : f64
    return %res : f64
}

// CHECK-LABEL: @funcTensorScalar.fullgrad0ps(%arg0: tensor<3xf64>) -> tensor<3xf64>
    // CHECK:   [[pcount:%.+]] = call @funcTensorScalar.pcount(%arg0) : (tensor<3xf64>) -> index
    // CHECK:   [[qgrad:%.+]] = call @funcTensorScalar.qgrad(%arg0, [[pcount]]) : (tensor<3xf64>, index) -> tensor<?xf64>
    // CHECK:   [[grad:%.+]] = gradient.backprop @funcTensorScalar.argmap(%arg0) cotangents([[qgrad]] : tensor<?xf64>) : (tensor<3xf64>) -> tensor<3xf64>
    // CHECK:   return [[grad]]

func.func @gradCallTensorScalar(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %2 = gradient.grad "defer"  @funcTensorScalar(%arg0) : (tensor<3xf64>) -> tensor<3xf64>
    func.return %2 : tensor<3xf64>
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
