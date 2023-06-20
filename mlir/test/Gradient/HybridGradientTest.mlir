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
    // CHECK:   [[CJAC:%.+]] = gradient.grad "fd" @funcScalarScalar.argmap(%arg0) : (f64) -> tensor<?xf64>
    // CHECK:   [[PCOUNT:%.+]] = tensor.dim [[CJAC]]
    // CHECK:   [[QGRAD:%.+]] = call @funcScalarScalar.qgrad(%arg0, [[PCOUNT]]) :  (f64, index) -> tensor<?xf64>

    // CHECK:   [[GRAD:%.+]] = tensor.generate
    // CHECK:       [[C_SLICE:%.+]] = tensor.extract_slice [[CJAC]][0] [[[PCOUNT]]] [1]
    // CHECK:       [[Q_SLICE:%.+]] = tensor.extract_slice [[QGRAD]][0] [[[PCOUNT]]] [1]
    // CHECK:       [[RES_T:%.+]] = linalg.dot ins([[C_SLICE]], [[Q_SLICE]] : tensor<?xf64>, tensor<?xf64>)
    // CHECK:       [[RES:%.+]] = tensor.extract [[RES_T]][]
    // CHECK:       tensor.yield [[RES]]

    // CHECK:   [[GRAD_RESHAPED:%.+]] = tensor.collapse_shape [[GRAD]] [] : tensor<1xf64> into tensor<f64>
    // CHECK:   [[GRAD_SCALAR:%.+]] = tensor.extract [[GRAD_RESHAPED]][]
    // CHECK:   return [[GRAD_SCALAR]]
// }

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
    // CHECK:   [[CJAC:%.+]] = gradient.grad "fd" @funcScalarPointTensor.argmap(%arg0) : (f64) -> tensor<?xf64>
    // CHECK:   [[PCOUNT:%.+]] = tensor.dim [[CJAC]]
    // CHECK:   [[QGRAD:%.+]] = call @funcScalarPointTensor.qgrad(%arg0, [[PCOUNT]]) :  (f64, index) -> tensor<?xf64>

    // CHECK:   [[GRAD:%.+]] = tensor.generate
    // CHECK:       [[C_SLICE:%.+]] = tensor.extract_slice [[CJAC]][0] [[[PCOUNT]]] [1]
    // CHECK:       [[Q_SLICE:%.+]] = tensor.extract_slice [[QGRAD]][0] [[[PCOUNT]]] [1]
    // CHECK:       [[RES_T:%.+]] = linalg.dot ins([[C_SLICE]], [[Q_SLICE]] : tensor<?xf64>, tensor<?xf64>)
    // CHECK:       [[RES:%.+]] = tensor.extract [[RES_T]][]
    // CHECK:       tensor.yield [[RES]]

    // CHECK:   [[GRAD_RESHAPED:%.+]] = tensor.collapse_shape [[GRAD]] [] : tensor<1xf64> into tensor<f64>
    // CHECK:   return [[GRAD_RESHAPED]]
// }

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
    // CHECK:   [[CJAC:%.+]] = gradient.grad "fd" @funcPointTensorScalar.argmap(%arg0) : (tensor<f64>) -> tensor<?xf64>
    // CHECK:   [[PCOUNT:%.+]] = tensor.dim [[CJAC]]
    // CHECK:   [[QGRAD:%.+]] = call @funcPointTensorScalar.qgrad(%arg0, [[PCOUNT]]) :  (tensor<f64>, index) -> tensor<?xf64>

    // CHECK:   [[GRAD:%.+]] = tensor.generate
    // CHECK:       [[C_SLICE:%.+]] = tensor.extract_slice [[CJAC]][0] [[[PCOUNT]]] [1]
    // CHECK:       [[Q_SLICE:%.+]] = tensor.extract_slice [[QGRAD]][0] [[[PCOUNT]]] [1]
    // CHECK:       [[RES_T:%.+]] = linalg.dot ins([[C_SLICE]], [[Q_SLICE]] : tensor<?xf64>, tensor<?xf64>)
    // CHECK:       [[RES:%.+]] = tensor.extract [[RES_T]][]
    // CHECK:       tensor.yield [[RES]]

    // CHECK:   [[GRAD_RESHAPED:%.+]] = tensor.collapse_shape [[GRAD]] [] : tensor<1xf64> into tensor<f64>
    // CHECK:   [[GRAD_SCALAR:%.+]] = tensor.extract [[GRAD_RESHAPED]][]
    // CHECK:   return [[GRAD_SCALAR]]
// }

func.func @gradCallPointTensorScalar(%arg0: tensor<f64>) -> f64 {
    %0 = gradient.grad "defer" @funcPointTensorScalar(%arg0) : (tensor<f64>) -> f64
    func.return %0 : f64
}


// -----

// Check point tensor to point tensor
func.func private @funcPointTensorPointTensor(%arg0: tensor<f64>) -> tensor<f64> attributes {qnode, diff_method = "parameter-shift"} {
    return %arg0 : tensor<f64>
}

// CHECK-LABEL: @funcPointTensorPointTensor.fullgrad0ps(%arg0: tensor<f64>) -> tensor<f64>
    // CHECK:   [[CJAC:%.+]] = gradient.grad "fd" @funcPointTensorPointTensor.argmap(%arg0) : (tensor<f64>) -> tensor<?xf64>
    // CHECK:   [[PCOUNT:%.+]] = tensor.dim [[CJAC]]
    // CHECK:   [[QGRAD:%.+]] = call @funcPointTensorPointTensor.qgrad(%arg0, [[PCOUNT]]) :  (tensor<f64>, index) -> tensor<?xf64>

    // CHECK:   [[GRAD:%.+]] = tensor.generate
    // CHECK:       [[C_SLICE:%.+]] = tensor.extract_slice [[CJAC]][0] [[[PCOUNT]]] [1]
    // CHECK:       [[Q_SLICE:%.+]] = tensor.extract_slice [[QGRAD]][0] [[[PCOUNT]]] [1]
    // CHECK:       [[RES_T:%.+]] = linalg.dot ins([[C_SLICE]], [[Q_SLICE]] : tensor<?xf64>, tensor<?xf64>)
    // CHECK:       [[RES:%.+]] = tensor.extract [[RES_T]][]
    // CHECK:       tensor.yield [[RES]]

    // CHECK:   [[GRAD_RESHAPED:%.+]] = tensor.collapse_shape [[GRAD]] [] : tensor<1xf64> into tensor<f64>
    // CHECK:   return [[GRAD_RESHAPED]]
// }

func.func @gradCallPointTensorPointTensor(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = gradient.grad "defer" @funcPointTensorPointTensor(%arg0) : (tensor<f64>) -> tensor<f64>
    func.return %0 : tensor<f64>
}

// -----

// Check scalar to tensor function
func.func private @funcScalarTensor(%arg0: f32) -> tensor<2x3xf64> attributes {qnode, diff_method = "parameter-shift"} {
    %c0 = arith.constant 0.0 : f64
    %res = tensor.from_elements %c0, %c0, %c0, %c0, %c0, %c0 : tensor<2x3xf64>
    return %res : tensor<2x3xf64>
}

// CHECK-LABEL: @funcScalarTensor.fullgrad0ps(%arg0: f32) -> tensor<2x3xf64>
    // CHECK:        [[CJAC:%.+]] = gradient.grad "fd" @funcScalarTensor.argmap(%arg0) : (f32) -> tensor<?xf64>
    // CHECK:        [[PCOUNT:%.+]] = tensor.dim [[CJAC]]
    // CHECK:        [[QGRAD:%.+]] = call @funcScalarTensor.qgrad(%arg0, [[PCOUNT]]) : (f32, index) -> tensor<?x2x3xf64>

    // CHECK:        [[GRAD:%.+]] = tensor.generate
    // CHECK-NEXT:   ^bb0([[i0:%.+]]: index, [[i1:%.+]]: index):
    // CHECK:            [[C_SLICE:%.+]] = tensor.extract_slice [[CJAC]][0] [[[PCOUNT]]] [1]
    // CHECK:            [[Q_SLICE:%.+]] = tensor.extract_slice [[QGRAD]][0, [[i0]], [[i1]]] [[[PCOUNT]], 1, 1] [1, 1, 1]
    // CHECK:            [[RES_T:%.+]] = linalg.dot ins([[C_SLICE]], [[Q_SLICE]] : tensor<?xf64>, tensor<?xf64>)
    // CHECK:            [[RES:%.+]] = tensor.extract [[RES_T]][]
    // CHECK:            tensor.yield [[RES]]

    // CHECK:        return [[GRAD]]
// }

func.func @gradCallScalarTensor(%arg0: f32) -> tensor<2x3xf64> {
    %0 = gradient.grad "defer"  @funcScalarTensor(%arg0) : (f32) -> tensor<2x3xf64>
    func.return %0 : tensor<2x3xf64>
}

// -----

// Check tensor to scalar
func.func private @funcTensorScalar(%arg0: tensor<3xf64>) -> f128 attributes {qnode, diff_method = "parameter-shift"} {
    %res = arith.constant 0.0 : f128
    return %res : f128
}

// CHECK-LABEL: @funcTensorScalar.fullgrad0ps(%arg0: tensor<3xf64>) -> tensor<3xf128>
    // CHECK:        [[c1:%.+]] = arith.constant 1 : index
    // CHECK:        [[CJAC:%.+]] = gradient.grad "fd" @funcTensorScalar.argmap(%arg0) : (tensor<3xf64>) -> tensor<3x?xf64>
    // CHECK:        [[PCOUNT:%.+]] = tensor.dim [[CJAC]], [[c1]]
    // CHECK:        [[QGRAD:%.+]] = call @funcTensorScalar.qgrad(%arg0, [[PCOUNT]]) : (tensor<3xf64>, index) -> tensor<?xf128>
    // CHECK:        [[CJAC_e:%.+]] = arith.extf [[CJAC]] : tensor<3x?xf64> to tensor<3x?xf128>

    // CHECK:        [[GRAD:%.+]] = tensor.generate
    // CHECK-NEXT:   ^bb0([[i0:%.+]]: index):
    // CHECK:            [[C_SLICE:%.+]] = tensor.extract_slice [[CJAC_e]][[[i0]], 0] [1, [[PCOUNT]]] [1, 1]
    // CHECK:            [[Q_SLICE:%.+]] = tensor.extract_slice [[QGRAD]][0] [[[PCOUNT]]] [1]
    // CHECK:            [[RES_T:%.+]] = linalg.dot ins([[C_SLICE]], [[Q_SLICE]] : tensor<?xf128>, tensor<?xf128>)
    // CHECK:            [[RES:%.+]] = tensor.extract [[RES_T]][]
    // CHECK:            tensor.yield [[RES]]

    // CHECK:        return [[GRAD]]
// }

func.func @gradCallTensorScalar(%arg0: tensor<3xf64>) -> tensor<3xf128> {
    %2 = gradient.grad "defer"  @funcTensorScalar(%arg0) : (tensor<3xf64>) -> tensor<3xf128>
    func.return %2 : tensor<3xf128>
}

// -----

// Check tensor to tensor case
func.func private @funcTensorTensor(%arg0: tensor<7x3x2x1xf64>) -> tensor<2xf32> attributes {qnode, diff_method = "parameter-shift"} {
    %c0 = arith.constant 0.0 : f32
    %res = tensor.from_elements %c0, %c0 : tensor<2xf32>
    return %res : tensor<2xf32>
}

// CHECK-LABEL: @funcTensorTensor.fullgrad0ps(%arg0: tensor<7x3x2x1xf64>) -> tensor<7x3x2x1x2xf32>
    // CHECK:        [[c4:%.+]] = arith.constant 4 : index
    // CHECK:        [[CJAC:%.+]] = gradient.grad "fd" @funcTensorTensor.argmap(%arg0) : (tensor<7x3x2x1xf64>) -> tensor<7x3x2x1x?xf64>
    // CHECK:        [[PCOUNT:%.+]] = tensor.dim [[CJAC]], [[c4]]
    // CHECK:        [[QGRAD:%.+]] = call @funcTensorTensor.qgrad(%arg0, [[PCOUNT]]) : (tensor<7x3x2x1xf64>, index) -> tensor<?x2xf32>
    // CHECK:        [[CJAC_t:%.+]] = arith.truncf [[CJAC]] : tensor<7x3x2x1x?xf64> to tensor<7x3x2x1x?xf32>

    // CHECK:        [[GRAD:%.+]] = tensor.generate
    // CHECK-NEXT:   ^bb0([[i0:%.+]]: index, [[i1:%.+]]: index, [[i2:%.+]]: index, [[i3:%.+]]: index, [[i4:%.+]]: index):
    // CHECK:            [[C_SLICE:%.+]] = tensor.extract_slice [[CJAC_t]][[[i0]], [[i1]], [[i2]], [[i3]], 0] [1, 1, 1, 1, [[PCOUNT]]] [1, 1, 1, 1, 1]
    // CHECK:            [[Q_SLICE:%.+]] = tensor.extract_slice [[QGRAD]][0, [[i4]]] [[[PCOUNT]], 1] [1, 1]
    // CHECK:            [[RES_T:%.+]] = linalg.dot ins([[C_SLICE]], [[Q_SLICE]] : tensor<?xf32>, tensor<?xf32>)
    // CHECK:            [[RES:%.+]] = tensor.extract [[RES_T]][]
    // CHECK:            tensor.yield [[RES]]

    // CHECK:        return [[GRAD]]
// }

func.func @gradCallTensorTensor(%arg0: tensor<7x3x2x1xf64>) -> tensor<7x3x2x1x2xf32> {
    %2 = gradient.grad "defer" @funcTensorTensor(%arg0) : (tensor<7x3x2x1xf64>) -> tensor<7x3x2x1x2xf32>
    func.return %2 : tensor<7x3x2x1x2xf32>
}

// -----

// Check the multiple results case
func.func @funcMultiRes(%arg0: f64) -> (f64, tensor<2xf64>) attributes {qnode, diff_method = "parameter-shift"} {
    %res = tensor.from_elements %arg0, %arg0 : tensor<2xf64>
    func.return %arg0, %res : f64, tensor<2xf64>
}

// CHECK-LABEL: @funcMultiRes.fullgrad0ps(%arg0: f64) -> (f64, tensor<2xf64>)
    // CHECK:         [[CJAC:%.+]] = gradient.grad "fd" @funcMultiRes.argmap(%arg0) : (f64) -> tensor<?xf64>
    // CHECK:         [[PCOUNT:%.+]] = tensor.dim [[CJAC]]
    // CHECK:         [[QGRAD:%.+]]:2 = call @funcMultiRes.qgrad(%arg0, [[PCOUNT]]) : (f64, index) -> (tensor<?xf64>, tensor<?x2xf64>)

    // CHECK:         [[GRAD:%.+]] = tensor.generate
    // CHECK:             [[C_SLICE:%.+]] = tensor.extract_slice [[CJAC]][0] [[[PCOUNT]]] [1]
    // CHECK:             [[Q_SLICE:%.+]] = tensor.extract_slice [[QGRAD]]#0[0] [[[PCOUNT]]] [1]
    // CHECK:             [[RES_T:%.+]] = linalg.dot ins([[C_SLICE]], [[Q_SLICE]] : tensor<?xf64>, tensor<?xf64>)
    // CHECK:             [[RES:%.+]] = tensor.extract [[RES_T]][]
    // CHECK:             tensor.yield [[RES]]
    // CHECK:         [[GRAD_RESHAPED:%.+]] = tensor.collapse_shape [[GRAD]] [] : tensor<1xf64> into tensor<f64>
    // CHECK:         [[GRAD0:%.+]] = tensor.extract [[GRAD_RESHAPED]][]

    // CHECK:         [[GRAD1:%.+]] = tensor.generate
    // CHECK-NEXT:    ^bb0([[i0:%.+]]: index):
    // CHECK:             [[C_SLICE:%.+]] = tensor.extract_slice [[CJAC]][0] [[[PCOUNT]]] [1]
    // CHECK:             [[Q_SLICE:%.+]] = tensor.extract_slice [[QGRAD]]#1[0, [[i0]]] [[[PCOUNT]], 1] [1, 1]
    // CHECK:             [[RES_T:%.+]] = linalg.dot ins([[C_SLICE]], [[Q_SLICE]] : tensor<?xf64>, tensor<?xf64>)
    // CHECK:             [[RES:%.+]] = tensor.extract [[RES_T]][]
    // CHECK:             tensor.yield [[RES]]

    // CHECK:         return [[GRAD0]], [[GRAD1]] : f64, tensor<2xf64>
// }

func.func @gradCallMultiRes(%arg0: f64) -> (f64, tensor<2xf64>)  {
    %0:2 = gradient.grad "defer" @funcMultiRes(%arg0) : (f64) -> (f64, tensor<2xf64>)
    func.return %0#0, %0#1 : f64, tensor<2xf64>
}

// -----

// Check the multiple arguments case
func.func @funcMultiArg(%arg0: f64, %arg1: tensor<2xf64>) -> f64 attributes {qnode, diff_method = "parameter-shift"} {
    func.return %arg0 : f64
}

// CHECK-LABEL: @funcMultiArg.fullgrad0ps(%arg0: f64, %arg1: tensor<2xf64>) -> f64
    // CHECK:       [[CJAC:%.+]] = gradient.grad "fd" @funcMultiArg.argmap(%arg0, %arg1) : {{.+}} -> tensor<?xf64>
    // CHECK:       [[PCOUNT:%.+]] = tensor.dim [[CJAC]]
    // CHECK:       [[QGRAD:%.+]] = call @funcMultiArg.qgrad(%arg0, %arg1, [[PCOUNT]]) : {{.+}} -> tensor<?xf64>

    // CHECK:       [[GRAD:%.+]] = tensor.generate
    // CHECK:           [[C_SLICE:%.+]] = tensor.extract_slice [[CJAC]][0] [[[PCOUNT]]] [1]
    // CHECK:           [[Q_SLICE:%.+]] = tensor.extract_slice [[QGRAD]][0] [[[PCOUNT]]] [1]
    // CHECK:           [[RES_T:%.+]] = linalg.dot ins([[C_SLICE]], [[Q_SLICE]] : tensor<?xf64>, tensor<?xf64>)
    // CHECK:           [[RES:%.+]] = tensor.extract [[RES_T]][]
    // CHECK:           tensor.yield [[RES]]
    // CHECK:       [[GRAD_RESHAPED:%.+]] = tensor.collapse_shape [[GRAD]] [] : tensor<1xf64> into tensor<f64>
    // CHECK:       [[GRAD0:%.+]] = tensor.extract [[GRAD_RESHAPED]][]

    // CHECK:       return [[GRAD0]] : f64
// }

// CHECK-LABEL: @funcMultiArg.fullgrad1ps(%arg0: f64, %arg1: tensor<2xf64>) -> tensor<2xf64>
    // CHECK:        [[c1:%.+]] = arith.constant 1 : index
    // CHECK:       [[CJAC:%.+]] = gradient.grad "fd" @funcMultiArg.argmap(%arg0, %arg1) {diffArgIndices = dense<1> : tensor<1xindex>} : {{.+}} -> tensor<2x?xf64>
    // CHECK:       [[PCOUNT:%.+]] = tensor.dim [[CJAC]], [[c1]]
    // CHECK:       [[QGRAD:%.+]] = call @funcMultiArg.qgrad(%arg0, %arg1, [[PCOUNT]]) : {{.+}} -> tensor<?xf64>

    // CHECK:       [[GRAD1:%.+]] = tensor.generate
    // CHECK-NEXT:  ^bb0([[i0:%.+]]: index):
    // CHECK:           [[C_SLICE:%.+]] = tensor.extract_slice [[CJAC]][[[i0]], 0] [1, [[PCOUNT]]] [1, 1]
    // CHECK:           [[Q_SLICE:%.+]] = tensor.extract_slice [[QGRAD]][0] [[[PCOUNT]]] [1]
    // CHECK:           [[RES_T:%.+]] = linalg.dot ins([[C_SLICE]], [[Q_SLICE]] : tensor<?xf64>, tensor<?xf64>)
    // CHECK:           [[RES:%.+]] = tensor.extract [[RES_T]][]
    // CHECK:           tensor.yield [[RES]]

    // CHECK:       return [[GRAD1]] : tensor<2xf64>
// }

// CHECK-LABEL: @funcMultiArg.fullgrad01ps(%arg0: f64, %arg1: tensor<2xf64>) -> (f64, tensor<2xf64>)
    // CHECK:        [[CJAC:%.+]]:2 = gradient.grad "fd" @funcMultiArg.argmap(%arg0, %arg1) {diffArgIndices = dense<[0, 1]> : tensor<2xindex>} : {{.+}} -> (tensor<?xf64>, tensor<2x?xf64>)
    // CHECK:        [[PCOUNT:%.+]] = tensor.dim [[CJAC]]#0
    // CHECK:        [[QGRAD:%.+]] = call @funcMultiArg.qgrad(%arg0, %arg1, [[PCOUNT]]) : {{.+}} -> tensor<?xf64>

    // CHECK:        [[GRAD:%.+]] = tensor.generate
    // CHECK:            [[C_SLICE:%.+]] = tensor.extract_slice [[CJAC]]#0[0] [[[PCOUNT]]] [1]
    // CHECK:            [[Q_SLICE:%.+]] = tensor.extract_slice [[QGRAD]][0] [[[PCOUNT]]] [1]
    // CHECK:            [[RES_T:%.+]] = linalg.dot ins([[C_SLICE]], [[Q_SLICE]] : tensor<?xf64>, tensor<?xf64>)
    // CHECK:            [[RES:%.+]] = tensor.extract [[RES_T]][]
    // CHECK:            tensor.yield [[RES]]
    // CHECK:        [[GRAD_RESHAPED:%.+]] = tensor.collapse_shape [[GRAD]] [] : tensor<1xf64> into tensor<f64>
    // CHECK:        [[GRAD0:%.+]] = tensor.extract [[GRAD_RESHAPED]][]

    // CHECK:        [[GRAD1:%.+]] = tensor.generate
    // CHECK-NEXT:   ^bb0([[i0:%.+]]: index):
    // CHECK:            [[C_SLICE:%.+]] = tensor.extract_slice [[CJAC]]#1[[[i0]], 0] [1, [[PCOUNT]]] [1, 1]
    // CHECK:            [[Q_SLICE:%.+]] = tensor.extract_slice [[QGRAD]][0] [[[PCOUNT]]] [1]
    // CHECK:            [[RES_T:%.+]] = linalg.dot ins([[C_SLICE]], [[Q_SLICE]] : tensor<?xf64>, tensor<?xf64>)
    // CHECK:            [[RES:%.+]] = tensor.extract [[RES_T]][]
    // CHECK:            tensor.yield [[RES]]

    // CHECK:        return [[GRAD0]], [[GRAD1]] : f64, tensor<2xf64>
// }

func.func @gradCallMultiArg(%arg0: f64, %arg1: tensor<2xf64>) -> (f64, tensor<2xf64>, f64, tensor<2xf64>)  {
    %0 = gradient.grad "defer"  @funcMultiArg(%arg0, %arg1) : (f64, tensor<2xf64>) -> f64
    %1 = gradient.grad "defer"  @funcMultiArg(%arg0, %arg1) {diffArgIndices = dense<[1]> : tensor<1xindex>} : (f64, tensor<2xf64>) -> tensor<2xf64>
    %2:2 = gradient.grad "defer" @funcMultiArg(%arg0, %arg1) {diffArgIndices = dense<[0, 1]> : tensor<2xindex>} : (f64, tensor<2xf64>) -> (f64, tensor<2xf64>)
    func.return %0, %1, %2#0, %2#1 : f64, tensor<2xf64>, f64, tensor<2xf64>
}
