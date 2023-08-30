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

// RUN: quantum-opt %s --lower-gradients=only=adj --split-input-file --verify-diagnostics --allow-unregistered-dialect | FileCheck %s

// Check scalar to scalar function
func.func private @funcScalarScalar(%arg0: f64) -> f64 attributes {qnode, diff_method = "adjoint"} {
    %0 = quantum.alloc(1) : !quantum.reg
    "do-not.dce-the-quantum-reg"(%0) : (!quantum.reg) -> ()
    quantum.dealloc %0 : !quantum.reg
    return %arg0 : f64
}

// CHECK-LABEL: @funcScalarScalar.adjoint(%arg0: f64, %arg1: index) -> tensor<?xf64>
    // CHECK-NEXT:   [[GRAD:%.+]] = gradient.adjoint @funcScalarScalar.nodealloc(%arg0)
    // CHECK-NEXT:   return [[GRAD]]
// }

// CHECK-LABEL: @funcScalarScalar.preprocess(%arg0: f64, %arg1: index) -> f64

// CHECK-LABEL: @gradCallScalarScalar
func.func @gradCallScalarScalar(%arg0: f64) -> f64 {
    // CHECK:   [[GRAD:%.+]] = gradient.backprop @funcScalarScalar.preprocess(%arg0
    %0 = gradient.grad "defer" @funcScalarScalar(%arg0) : (f64) -> f64

    // CHECK:   return [[GRAD]]
    func.return %0 : f64
}

// -----

// Check scalar to tensor function
func.func private @funcScalarTensor(%arg0: f64) -> tensor<2x3xf64> attributes {qnode, diff_method = "adjoint"} {
    %0 = quantum.alloc(1) : !quantum.reg
    "do-not.dce-the-quantum-reg"(%0) : (!quantum.reg) -> ()
    quantum.dealloc %0 : !quantum.reg
    %c0 = arith.constant 0.0 : f64
    %res = tensor.from_elements %c0, %c0, %c0, %c0, %c0, %c0 : tensor<2x3xf64>
    return %res : tensor<2x3xf64>
}

// CHECK-LABEL: @funcScalarTensor.adjoint(%arg0: f64, %arg1: index) -> tensor<?x2x3xf64>
    // CHECK-NEXT:   [[GRAD:%.+]] = gradient.adjoint @funcScalarTensor.nodealloc(%arg0)
    // CHECK-NEXT:   return [[GRAD]]
// }

// CHECK-LABEL: @funcScalarTensor.preprocess(%arg0: f64, %arg1: index) -> tensor<2x3xf64>

// CHECK-LABEL: @gradCallScalarTensor
func.func @gradCallScalarTensor(%arg0: f64) -> tensor<2x3xf64> {
    // CHECK:   [[jacEntry:%.+]] = gradient.backprop @funcScalarTensor.preprocess(%arg0
    %0 = gradient.grad "defer"  @funcScalarTensor(%arg0) : (f64) -> tensor<2x3xf64>
    // CHECK:   [[jacobian:%.+]] = tensor.insert [[jacEntry]] into
    // CHECK:   return [[jacobian]]
    func.return %0 : tensor<2x3xf64>
}

// -----

// Check tensor to scalar
func.func private @funcTensorScalar(%arg0: tensor<3xf64>) -> f64 attributes {qnode, diff_method = "adjoint"} {
    %0 = quantum.alloc(1) : !quantum.reg
    "do-not.dce-the-quantum-reg"(%0) : (!quantum.reg) -> ()
    quantum.dealloc %0 : !quantum.reg
    %res = arith.constant 0.0 : f64
    return %res : f64
}

// CHECK-LABEL: @funcTensorScalar.adjoint(%arg0: tensor<3xf64>, %arg1: index) -> tensor<?xf64>
    // CHECK-NEXT:   [[GRAD:%.+]] = gradient.adjoint @funcTensorScalar.nodealloc(%arg0)
    // CHECK-NEXT:   return [[GRAD]]
// }

// CHECK-LABEL: @funcTensorScalar.preprocess(%arg0: tensor<3xf64>, %arg1: index) -> f64

// CHECK-LABEL: @gradCallTensorScalar
func.func @gradCallTensorScalar(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    // CHECK:   [[GRAD:%.+]] = gradient.backprop @funcTensorScalar.preprocess(%arg0
    %2 = gradient.grad "defer"  @funcTensorScalar(%arg0) : (tensor<3xf64>) -> tensor<3xf64>

    // CHECK:   return [[GRAD]]
    func.return %2 : tensor<3xf64>
}

// -----

// Check tensor to tensor case
func.func private @funcTensorTensor(%arg0: tensor<7x3x2x1xf64>) -> tensor<2xf64> attributes {qnode, diff_method = "adjoint"} {
    %0 = quantum.alloc(1) : !quantum.reg
    "do-not.dce-the-quantum-reg"(%0) : (!quantum.reg) -> ()
    quantum.dealloc %0 : !quantum.reg
    %c0 = arith.constant 0.0 : f64
    %res = tensor.from_elements %c0, %c0 : tensor<2xf64>
    return %res : tensor<2xf64>
}

// CHECK-LABEL: @funcTensorTensor.adjoint(%arg0: tensor<7x3x2x1xf64>, %arg1: index) -> tensor<?x2xf64>
    // CHECK-NEXT:   [[GRAD:%.+]] = gradient.adjoint @funcTensorTensor.nodealloc(%arg0)
    // CHECK-NEXT:   return [[GRAD]]
// }

// CHECK-LABEL: @funcTensorTensor.preprocess(%arg0: tensor<7x3x2x1xf64>, %arg1: index) -> tensor<2xf64>

// CHECK-LABEL: @gradCallTensorTensor
func.func @gradCallTensorTensor(%arg0: tensor<7x3x2x1xf64>) -> tensor<7x3x2x1x2xf64> {
    // CHECK:   [[jacSlice:%.+]] = gradient.backprop @funcTensorTensor.preprocess(%arg0
    %2 = gradient.grad "defer" @funcTensorTensor(%arg0) : (tensor<7x3x2x1xf64>) -> tensor<7x3x2x1x2xf64>

    // CHECK:   [[jacobian:%.+]] = tensor.insert_slice [[jacSlice]] into
    // CHECK:   return [[jacobian]]
    func.return %2 : tensor<7x3x2x1x2xf64>
}


// -----

// Check the multiple results case
func.func @funcMultiRes(%arg0: f64) -> (f64, tensor<2xf64>) attributes {qnode, diff_method = "adjoint"} {
    %0 = quantum.alloc(1) : !quantum.reg
    "do-not.dce-the-quantum-reg"(%0) : (!quantum.reg) -> ()
    quantum.dealloc %0 : !quantum.reg
    %res = tensor.from_elements %arg0, %arg0 : tensor<2xf64>
    func.return %arg0, %res : f64, tensor<2xf64>
}

// CHECK-LABEL: @funcMultiRes.adjoint(%arg0: f64, %arg1: index) -> (tensor<?xf64>, tensor<?x2xf64>)
    // CHECK-NEXT:   [[GRAD:%.+]]:2 = gradient.adjoint @funcMultiRes.nodealloc(%arg0)
    // CHECK-NEXT:   return [[GRAD]]#0, [[GRAD]]#1
// }

// CHECK-LABEL: @funcMultiRes.preprocess(%arg0: f64, %arg1: index) -> (f64, tensor<2xf64>)

// CHECK-LABEL: @gradCallMultiRes
func.func @gradCallMultiRes(%arg0: f64) -> (f64, tensor<2xf64>)  {
    // CHECK:   [[grad0:%.+]] = gradient.backprop @funcMultiRes.preprocess(%arg0{{.+}} -> f64
    // CHECK:   [[grad1:%.+]] = gradient.backprop @funcMultiRes.preprocess(%arg0{{.+}} -> f64
    // CHECK:   [[jac0:%.+]] = tensor.insert [[grad1]] into
    // CHECK:   [[grad2:%.+]] = gradient.backprop @funcMultiRes.preprocess(%arg0{{.+}} -> f64
    // CHECK:   [[jac1:%.+]] = tensor.insert [[grad2]] into
    %0:2 = gradient.grad "defer" @funcMultiRes(%arg0) : (f64) -> (f64, tensor<2xf64>)

    // CHECK:   return [[grad0]], [[jac1]]
    func.return %0#0, %0#1 : f64, tensor<2xf64>
}

// -----

// Check the case with multiple grad invocations with varying diffArgIndices
func.func @funcMultiArg(%arg0: f64, %arg1: tensor<2xf64>) -> f64 attributes {qnode, diff_method = "adjoint"} {
    %0 = quantum.alloc(1) : !quantum.reg
    "do-not.dce-the-quantum-reg"(%0) : (!quantum.reg) -> ()
    quantum.dealloc %0 : !quantum.reg
    func.return %arg0 : f64
}

// CHECK-LABEL: @funcMultiArg.adjoint(%arg0: f64, %arg1: tensor<2xf64>, %arg2: index) -> tensor<?xf64>
    // CHECK-NEXT:   [[GRAD:%.+]] = gradient.adjoint @funcMultiArg.nodealloc(%arg0, %arg1)
    // CHECK-NEXT:   return [[GRAD]]
// }

// CHECK-LABEL: @funcMultiArg.preprocess(%arg0: f64, %arg1: tensor<2xf64>, %arg2: index) -> f64

// CHECK-LABEL: @gradCallMultiArg
func.func @gradCallMultiArg(%arg0: f64, %arg1: tensor<2xf64>) -> (f64, tensor<2xf64>, f64, tensor<2xf64>) {
    // CHECK:   [[GRAD0:%.+]] = gradient.backprop @funcMultiArg.preprocess(%arg0, %arg1{{.+}} -> f64
    %0 = gradient.grad "defer"  @funcMultiArg(%arg0, %arg1) : (f64, tensor<2xf64>) -> f64
    // CHECK:   [[GRAD1:%.+]] = gradient.backprop @funcMultiArg.preprocess(%arg0, %arg1{{.+}} -> tensor<2xf64>
    %1 = gradient.grad "defer"  @funcMultiArg(%arg0, %arg1) {diffArgIndices = dense<[1]> : tensor<1xindex>} : (f64, tensor<2xf64>) -> tensor<2xf64>
    // CHECK:   [[GRAD2:%.+]]:2 = gradient.backprop @funcMultiArg.preprocess(%arg0, %arg1{{.+}} -> (f64, tensor<2xf64>)
    %2:2 = gradient.grad "defer" @funcMultiArg(%arg0, %arg1) {diffArgIndices = dense<[0, 1]> : tensor<2xindex>} : (f64, tensor<2xf64>) -> (f64, tensor<2xf64>)

    // CHECK:   return [[GRAD0]], [[GRAD1]], [[GRAD2]]#0, [[GRAD2]]#1
    func.return %0, %1, %2#0, %2#1 : f64, tensor<2xf64>, f64, tensor<2xf64>
}

// -----

// Check multiple grad calls to same function with the same diffArgIndices
func.func private @funcMultiCall(%arg0: f64) -> f64 attributes {qnode, diff_method = "adjoint"} {
    %0 = quantum.alloc(1) : !quantum.reg
    "do-not.dce-the-quantum-reg"(%0) : (!quantum.reg) -> ()
    quantum.dealloc %0 : !quantum.reg
    func.return %arg0 : f64
}

// CHECK-LABEL: @funcMultiCall.adjoint(%arg0: f64, %arg1: index) -> tensor<?xf64>

// CHECK-LABEL: @funcMultiCall.preprocess(%arg0: f64, %arg1: index) -> f64

// CHECK-LABEL: @gradCallMultiCall
func.func @gradCallMultiCall(%arg0: f64) -> (f64, f64) {
    // CHECK:   [[GRAD0:%.+]] = gradient.backprop @funcMultiCall.preprocess(%arg0
    %0 = gradient.grad "defer" @funcMultiCall(%arg0) : (f64) -> f64
    // CHECK:   [[GRAD1:%.+]] = gradient.backprop @funcMultiCall.preprocess(%arg0
    %1 = gradient.grad "defer" @funcMultiCall(%arg0) : (f64) -> f64
    // CHECK:   return [[GRAD0]], [[GRAD1]]
    func.return %0, %1 : f64, f64
}

// -----

// Check invalid function with no deallocation
func.func private @funcScalarScalar(%arg0: f64) -> f64 attributes {qnode, diff_method = "adjoint"} {
    %0 = quantum.alloc(1) : !quantum.reg
    return %arg0 : f64
}

func.func @gradCallScalarScalar(%arg0: f64) -> f64 {
    // expected-error@-6 {{Invalid number of quantum registers: 0}}
    %0 = gradient.grad "defer" @funcScalarScalar(%arg0) : (f64) -> f64
    func.return %0 : f64
}
