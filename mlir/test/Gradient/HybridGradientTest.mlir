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

// RUN: quantum-opt %s --lower-gradients=only=ps -cse | FileCheck %s

// Check scalar to scalar function
func.func private @funcScalarScalar(%arg0: f64) -> f64 attributes {qnode, diff_method = "parameter-shift"} {
    return %arg0 : f64
}

// CHECK-LABEL: @gradCallScalarScalar(%arg0: f64) -> f64
    // CHECK-DAG:    [[zero:%.+]] = arith.constant 0.0{{.+}}+00 : f64
    // CHECK-DAG:    [[one:%.+]] = arith.constant 1.0{{.+}}+00 : f64
    // CHECK:        [[cotangent0:%.+]] = tensor.empty() : tensor<f64>
    // CHECK:        [[cotangent1:%.+]] = linalg.fill ins([[zero]] : f64) outs([[cotangent0]]
    // CHECK:        [[cotangent:%.+]] = tensor.insert [[one]] into [[cotangent1]]
    // CHECK:        [[grad:%.+]] = gradient.backprop @funcScalarScalar.cloned(%arg0) cotangents([[cotangent]]
    // CHECK:        return [[grad]]

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

// CHECK-LABEL: @gradCallScalarPointTensor(%arg0: f64) -> tensor<f64>
    // CHECK-DAG:    [[zero:%.+]] = arith.constant 0.0{{.+}}+00 : f64
    // CHECK-DAG:    [[one:%.+]] = arith.constant 1.0{{.+}}+00 : f64
    // CHECK:        [[empty:%.+]] = tensor.empty() : tensor<f64>
    // CHECK:        [[cotangent1:%.+]] = linalg.fill ins([[zero]] : f64) outs([[empty]]
    // CHECK:        [[cotangent:%.+]] = tensor.insert [[one]] into [[cotangent1]]
    // CHECK:        [[grad:%.+]] = gradient.backprop @funcScalarPointTensor.cloned(%arg0) cotangents([[cotangent]]
    // CHECK:        [[gradTensor:%.+]] = tensor.insert [[grad]] into [[empty]]
    // CHECK:        return [[gradTensor]]

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

// CHECK-LABEL: @gradCallPointTensorScalar(%arg0: tensor<f64>) -> f64
    // CHECK-DAG:    [[zero:%.+]] = arith.constant 0.0{{.+}}+00 : f64
    // CHECK-DAG:    [[one:%.+]] = arith.constant 1.0{{.+}}+00 : f64
    // CHECK:        [[empty:%.+]] = tensor.empty() : tensor<f64>
    // CHECK:        [[cotangent1:%.+]] = linalg.fill ins([[zero]] : f64) outs([[empty]]
    // CHECK:        [[cotangent:%.+]] = tensor.insert [[one]] into [[cotangent1]]
    // CHECK:        [[gradTensor:%.+]] = gradient.backprop @funcPointTensorScalar.cloned(%arg0) cotangents([[cotangent]]
    // CHECK:        [[grad:%.+]] = tensor.extract [[gradTensor]]
    // CHECK:        return [[grad]]

func.func @gradCallPointTensorScalar(%arg0: tensor<f64>) -> f64 {
    %0 = gradient.grad "defer" @funcPointTensorScalar(%arg0) : (tensor<f64>) -> f64
    func.return %0 : f64
}


// -----

// Check point tensor to point tensor
func.func private @funcPointTensorPointTensor(%arg0: tensor<f64>) -> tensor<f64> attributes {qnode, diff_method = "parameter-shift"} {
    return %arg0 : tensor<f64>
}

// CHECK-LABEL: @gradCallPointTensorPointTensor(%arg0: tensor<f64>) -> tensor<f64>
    // CHECK-DAG:    [[zero:%.+]] = arith.constant 0.0{{.+}}+00 : f64
    // CHECK-DAG:    [[one:%.+]] = arith.constant 1.0{{.+}}+00 : f64
    // CHECK:        [[empty:%.+]] = tensor.empty() : tensor<f64>
    // CHECK:        [[cotangent1:%.+]] = linalg.fill ins([[zero]] : f64) outs([[empty]]
    // CHECK:        [[cotangent:%.+]] = tensor.insert [[one]] into [[cotangent1]]
    // CHECK:        [[gradTensor:%.+]] = gradient.backprop @funcPointTensorPointTensor.cloned(%arg0) cotangents([[cotangent]]
    // CHECK:        return [[gradTensor]]

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

// CHECK-LABEL: @gradCallScalarTensor(%arg0: f64) -> tensor<2x3xf64>
    // CHECK-DAG:    [[zero:%.+]] = arith.constant 0.0{{.+}}+00 : f64
    // CHECK-DAG:    [[one:%.+]] = arith.constant 1.0{{.+}}+00 : f64
    // CHECK-DAG:    [[idx0:%.+]] = index.constant 0
    // CHECK-DAG:    [[idx1:%.+]] = index.constant 1
    // CHECK-DAG:    [[idx2:%.+]] = index.constant 2
    // CHECK:        [[empty:%.+]] = tensor.empty() : tensor<2x3xf64>
    // CHECK:        [[zeroed:%.+]] = linalg.fill ins([[zero]] : f64) outs([[empty]]

    // CHECK:        [[cotangent0:%.+]] = tensor.insert [[one]] into [[zeroed]][[[idx0]], [[idx0]]]
    // CHECK:        [[jacEntry00:%.+]] = gradient.backprop @funcScalarTensor.cloned(%arg0) cotangents([[cotangent0]]
    // CHECK:        [[jac0:%.+]] = tensor.insert [[jacEntry00]] into [[empty]][[[idx0]], [[idx0]]]

    // CHECK:        [[cotangent1:%.+]] = tensor.insert [[one]] into [[zeroed]][[[idx0]], [[idx1]]]
    // CHECK:        [[jacEntry01:%.+]] = gradient.backprop @funcScalarTensor.cloned(%arg0) cotangents([[cotangent1]]
    // CHECK:        [[jac1:%.+]] = tensor.insert [[jacEntry01]] into [[jac0]][[[idx0]], [[idx1]]]

    // CHECK:        [[cotangent2:%.+]] = tensor.insert [[one]] into [[zeroed]][[[idx0]], [[idx2]]]
    // CHECK:        [[jacEntry02:%.+]] = gradient.backprop @funcScalarTensor.cloned(%arg0) cotangents([[cotangent2]]
    // CHECK:        [[jac2:%.+]] = tensor.insert [[jacEntry02]] into [[jac1]][[[idx0]], [[idx2]]]

    // CHECK:        [[cotangent3:%.+]] = tensor.insert [[one]] into [[zeroed]][[[idx1]], [[idx0]]]
    // CHECK:        [[jacEntry10:%.+]] = gradient.backprop @funcScalarTensor.cloned(%arg0) cotangents([[cotangent3]]
    // CHECK:        [[jac3:%.+]] = tensor.insert [[jacEntry10]] into [[jac2]][[[idx1]], [[idx0]]]

    // CHECK:        [[cotangent4:%.+]] = tensor.insert [[one]] into [[zeroed]][[[idx1]], [[idx1]]]
    // CHECK:        [[jacEntry11:%.+]] = gradient.backprop @funcScalarTensor.cloned(%arg0) cotangents([[cotangent4]]
    // CHECK:        [[jac4:%.+]] = tensor.insert [[jacEntry11]] into [[jac3]][[[idx1]], [[idx1]]]

    // CHECK:        [[cotangent5:%.+]] = tensor.insert [[one]] into [[zeroed]][[[idx1]], [[idx2]]]
    // CHECK:        [[jacEntry12:%.+]] = gradient.backprop @funcScalarTensor.cloned(%arg0) cotangents([[cotangent5]]
    // CHECK:        [[jac5:%.+]] = tensor.insert [[jacEntry12]] into [[jac4]][[[idx1]], [[idx2]]]

    // CHECK:        return [[jac5]]

func.func @gradCallScalarTensor(%arg0: f64) -> tensor<2x3xf64> {
    %0 = gradient.grad "defer" @funcScalarTensor(%arg0) : (f64) -> tensor<2x3xf64>
    func.return %0 : tensor<2x3xf64>
}

// -----

// Check tensor to scalar
func.func private @funcTensorScalar(%arg0: tensor<3xf64>) -> f64 attributes {qnode, diff_method = "parameter-shift"} {
    %res = arith.constant 0.0 : f64
    return %res : f64
}

// CHECK-LABEL: @gradCallTensorScalar(%arg0: tensor<3xf64>) -> tensor<3xf64>
    // CHECK-DAG:    [[zero:%.+]] = arith.constant 0.0{{.+}}+00 : f64
    // CHECK-DAG:    [[one:%.+]] = arith.constant 1.0{{.+}}+00 : f64
    // CHECK:        [[empty:%.+]] = tensor.empty() : tensor<f64>
    // CHECK:        [[cotangent1:%.+]] = linalg.fill ins([[zero]] : f64) outs([[empty]]
    // CHECK:        [[cotangent:%.+]] = tensor.insert [[one]] into [[cotangent1]]
    // CHECK:        [[gradTensor:%.+]] = gradient.backprop @funcTensorScalar.cloned(%arg0) cotangents([[cotangent]]
    // CHECK:        return [[gradTensor]]

func.func @gradCallTensorScalar(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %2 = gradient.grad "defer" @funcTensorScalar(%arg0) : (tensor<3xf64>) -> tensor<3xf64>
    func.return %2 : tensor<3xf64>
}

// -----

// Check tensor to tensor case
func.func private @funcTensorTensor(%arg0: tensor<7x3x2x1xf64>) -> tensor<2xf64> attributes {qnode, diff_method = "parameter-shift"} {
    %c0 = arith.constant 0.0 : f64
    %res = tensor.from_elements %c0, %c0 : tensor<2xf64>
    return %res : tensor<2xf64>
}

// CHECK-LABEL: @gradCallTensorTensor(%arg0: tensor<7x3x2x1xf64>) -> tensor<7x3x2x1x2xf64>
    // CHECK-DAG:    [[zero:%.+]] = arith.constant 0.0{{.+}}+00 : f64
    // CHECK-DAG:    [[one:%.+]] = arith.constant 1.0{{.+}}+00 : f64
    // CHECK-DAG:    [[idx0:%.+]] = index.constant 0
    // CHECK-DAG:    [[idx1:%.+]] = index.constant 1
    // CHECK-DAG:    [[jacobian0:%.+]] = tensor.empty() : tensor<7x3x2x1x2xf64>
    // CHECK-DAG:    [[empty:%.+]] = tensor.empty() : tensor<2xf64>
    // CHECK:        [[zeroed:%.+]] = linalg.fill ins([[zero]] : f64) outs([[empty]]

    // CHECK:        [[cotangent0:%.+]] = tensor.insert [[one]] into [[zeroed]][[[idx0]]]
    // CHECK:        [[jacSlice0:%.+]] = gradient.backprop @funcTensorTensor.cloned(%arg0) cotangents([[cotangent0]]
    // CHECK:        [[jacobian1:%.+]] = tensor.insert_slice [[jacSlice0]] into [[jacobian0]][0, 0, 0, 0, [[idx0]]] [7, 3, 2, 1, 1] [1, 1, 1, 1, 1]

    // CHECK:        [[cotangent1:%.+]] = tensor.insert [[one]] into [[zeroed]][[[idx1]]]
    // CHECK:        [[jacSlice1:%.+]] = gradient.backprop @funcTensorTensor.cloned(%arg0) cotangents([[cotangent1]]
    // CHECK:        [[jacobian:%.+]] = tensor.insert_slice [[jacSlice1]] into [[jacobian1]][0, 0, 0, 0, [[idx1]]] [7, 3, 2, 1, 1] [1, 1, 1, 1, 1]

    // CHECK:        return [[jacobian]]

func.func @gradCallTensorTensor(%arg0: tensor<7x3x2x1xf64>) -> tensor<7x3x2x1x2xf64> {
    %2 = gradient.grad "defer" @funcTensorTensor(%arg0) : (tensor<7x3x2x1xf64>) -> tensor<7x3x2x1x2xf64>
    func.return %2 : tensor<7x3x2x1x2xf64>
}

// -----

// Check the multiple arguments case
func.func @funcMultiArg(%arg0: tensor<f64>, %arg1: tensor<2xf64>) -> tensor<f64> attributes {qnode, diff_method = "parameter-shift"} {
    func.return %arg0 : tensor<f64>
}

// CHECK-LABEL:  @gradCallMultiArg(%arg0: tensor<f64>, %arg1: tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>, tensor<f64>, tensor<2xf64>)
    // CHECK-DAG:    [[zero:%.+]] = arith.constant 0.0{{.+}}+00 : f64
    // CHECK-DAG:    [[one:%.+]] = arith.constant 1.0{{.+}}+00 : f64
    // CHECK:        [[cotangent0:%.+]] = tensor.empty() : tensor<f64>
    // CHECK:        [[cotangent1:%.+]] = linalg.fill ins([[zero]] : f64) outs([[cotangent0]]
    // CHECK:        [[cotangent:%.+]] = tensor.insert [[one]] into [[cotangent1]]
    // CHECK:        [[grad0:%.+]] = gradient.backprop @funcMultiArg.cloned(%arg0, %arg1) cotangents([[cotangent]]
    // CHECK:        [[grad1:%.+]] = gradient.backprop @funcMultiArg.cloned(%arg0, %arg1) cotangents([[cotangent]] {{.+}} {diffArgIndices = dense<1>
    // CHECK:        [[grad2:%.+]]:2 = gradient.backprop @funcMultiArg.cloned(%arg0, %arg1) cotangents([[cotangent]] {{.+}} {diffArgIndices = dense<[0, 1]>
    // CHECK:        return [[grad0]], [[grad1]], [[grad2]]#0, [[grad2]]#1

func.func @gradCallMultiArg(%arg0: tensor<f64>, %arg1: tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>, tensor<f64>, tensor<2xf64>)  {
    %0 = gradient.grad "defer"  @funcMultiArg(%arg0, %arg1) : (tensor<f64>, tensor<2xf64>) -> tensor<f64>
    %1 = gradient.grad "defer"  @funcMultiArg(%arg0, %arg1) {diffArgIndices = dense<[1]> : tensor<1xindex>} : (tensor<f64>, tensor<2xf64>) -> tensor<2xf64>
    %2:2 = gradient.grad "defer" @funcMultiArg(%arg0, %arg1) {diffArgIndices = dense<[0, 1]> : tensor<2xindex>} : (tensor<f64>, tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>)
    func.return %0, %1, %2#0, %2#1 : tensor<f64>, tensor<2xf64>, tensor<f64>, tensor<2xf64>
}
