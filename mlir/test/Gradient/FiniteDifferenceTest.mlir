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

// RUN: quantum-opt %s --lower-gradients --split-input-file | FileCheck %s

// Check scalar to scalar function
func.func private @funcScalarScalar(%arg0: f64) -> f64 attributes {qnode, diff_method = "finite-diff"}

// CHECK-LABEL: @funcScalarScalar.finitediff0(%arg0: f64) -> f64
    // CHECK:        [[CONSTANT:%.+]] = arith.constant 2.000000e+00 : f64
    // CHECK:        [[CALLPOS:%.+]] = call @funcScalarScalar(%arg0) : (f64) -> f64
    // CHECK:        [[OPERANDFRWRD:%.+]] = arith.addf %arg0, [[CONSTANT]] : f64
    // CHECK-NEXT:   [[CALLFRWRD:%.+]] = call @funcScalarScalar([[OPERANDFRWRD]]) : (f64) -> f64
    // CHECK-NEXT:   [[ABSDIFF:%.+]] = arith.subf [[CALLFRWRD]], [[CALLPOS]] : f64
    // CHECK-NEXT:   [[RESULT:%.+]] = arith.divf [[ABSDIFF]], [[CONSTANT]] : f64
    // CHECK-NEXT:   return [[RESULT]]
// }

// CHECK-LABEL: @gradCallScalarScalar
func.func @gradCallScalarScalar(%arg0: f64) -> f64 {
    // CHECK:   [[GRAD:%.+]] = call @funcScalarScalar.finitediff0(%arg0) : (f64) -> f64
    %0 = gradient.grad "fd" @funcScalarScalar(%arg0) { finiteDiffParam = 2.000000e+00 : f64 } : (f64) -> f64
    // CHECK:   return [[GRAD]]
    func.return %0 : f64
}

// -----

// Check scalar to tensor function
func.func private @funcScalarTensor(%arg0: f64) -> tensor<2x3xf64> attributes {qnode, diff_method = "finite-diff"}

// CHECK-LABEL: @funcScalarTensor.finitediff0(%arg0: f64) -> tensor<2x3xf64>
    // CHECK-DAG:    [[RESULTCONST:%.+]] = arith.constant dense<2.000000e+00> : tensor<2x3xf64>
    // CHECK-DAG:    [[OPERANDCONST:%.+]] = arith.constant 2.000000e+00 : f64
    // CHECK:        [[CALLPOS:%.+]] = call @funcScalarTensor(%arg0) : (f64) -> tensor<2x3xf64>
    // CHECK:        [[OPERANDFRWRD:%.+]] = arith.addf %arg0, [[OPERANDCONST]] : f64
    // CHECK-NEXT:   [[CALLFRWRD:%.+]] = call @funcScalarTensor([[OPERANDFRWRD]]) : (f64) -> tensor<2x3xf64>
    // CHECK-NEXT:   [[ABSDIFF:%.+]] = arith.subf [[CALLFRWRD]], [[CALLPOS]] : tensor<2x3xf64>
    // CHECK-NEXT:   [[RESULT:%.+]] = arith.divf [[ABSDIFF]], [[RESULTCONST]] : tensor<2x3xf64>
    // CHECK-NEXT:   return [[RESULT]]
// }

// CHECK-LABEL: @gradCallScalarTensor
func.func @gradCallScalarTensor(%arg0: f64) -> tensor<2x3xf64> {
    // CHECK:   [[GRAD:%.+]] = call @funcScalarTensor.finitediff0(%arg0) : (f64) -> tensor<2x3xf64>
    %0 = gradient.grad "fd"  @funcScalarTensor(%arg0) { finiteDiffParam = 2.000000e+00 : f64 } : (f64) -> tensor<2x3xf64>
    // CHECK:   return [[GRAD]]
    func.return %0 : tensor<2x3xf64>
}

// -----

// Check scalar tensor to scalar
func.func private @funcTensorScalar(%arg0: tensor<f64>) -> f64 attributes {qnode, diff_method = "finite-diff"}

// CHECK-LABEL: @funcTensorScalar.finitediff0(%arg0: tensor<f64>) -> f64
    // CHECK-DAG:    [[CONSTANT:%.+]] = arith.constant {{.+}} : f64
    // CHECK-DAG:    [[CONSTANTTENSOR:%.+]] = arith.constant dense<{{.+}}> : tensor<f64>
    // CHECK:        [[CALLPOS:%.+]] = call @funcTensorScalar(%arg0) : (tensor<f64>) -> f64
    // CHECK:        [[OPERANDFRWD:%.+]] = arith.addf %arg0, [[CONSTANTTENSOR]] : tensor<f64>
    // CHECK-NEXT:   [[CALLFRWRD:%.+]] = call @funcTensorScalar([[OPERANDFRWD]]) : (tensor<f64>) -> f64
    // CHECK-NEXT:   [[ABSDIFF:%.+]] = arith.subf [[CALLFRWRD]], [[CALLPOS]] : f64
    // CHECK-NEXT:   [[RESULT:%.+]] = arith.divf [[ABSDIFF]], [[CONSTANT]] : f64
    // CHECK-NEXT:   return [[RESULT]]
// }

// CHECK-LABEL: @gradCallTensorScalar
func.func @gradCallTensorScalar(%arg0: tensor<f64>) -> f64 {
    // CHECK:   [[GRAD:%.+]] = call @funcTensorScalar.finitediff0(%arg0) : (tensor<f64>) -> f64
    %2 = gradient.grad "fd"  @funcTensorScalar(%arg0) : (tensor<f64>) -> f64
    // CHECK:   return [[GRAD]]
    func.return %2 : f64
}

// -----

// Check scalar to scalar tensor function
func.func private @funcScalarScalarTensor(%arg0: f64) -> tensor<f64> attributes {qnode, diff_method = "finite-diff"}

// CHECK-LABEL: @funcScalarScalarTensor.finitediff0(%arg0: f64) -> tensor<f64>
    // CHECK-DAG:    [[RESULTCONST:%.+]] = arith.constant dense<{{.+}}> : tensor<f64>
    // CHECK-DAG:    [[OPERANDCONST:%.+]] = arith.constant {{.+}} : f64
    // CHECK:        [[CALLPOS:%.+]] = call @funcScalarScalarTensor(%arg0) : (f64) -> tensor<f64>
    // CHECK:        [[OPERANDFRWRD:%.+]] = arith.addf %arg0, [[OPERANDCONST]] : f64
    // CHECK-NEXT:   [[CALLFRWRD:%.+]] = call @funcScalarScalarTensor([[OPERANDFRWRD]]) : (f64) -> tensor<f64>
    // CHECK-NEXT:   [[ABSDIFF:%.+]] = arith.subf [[CALLFRWRD]], [[CALLPOS]] : tensor<f64>
    // CHECK-NEXT:   [[RESULT:%.+]] = arith.divf [[ABSDIFF]], [[RESULTCONST]] : tensor<f64>
    // CHECK-NEXT:   return [[RESULT]]
// }

// CHECK-LABEL: @gradCallScalarScalarTensor
func.func @gradCallScalarScalarTensor(%arg0: f64) -> tensor<f64> {
    // CHECK:   [[GRAD:%.+]] = call @funcScalarScalarTensor.finitediff0(%arg0) : (f64) -> tensor<f64>
    %0 = gradient.grad "fd"  @funcScalarScalarTensor(%arg0) : (f64) -> tensor<f64>
    // CHECK:   return [[GRAD]]
    func.return %0 : tensor<f64>
}


// -----

// Check tensor to scalar
func.func private @funcTensorScalar(%arg0: tensor<3xf64>) -> f64 attributes {qnode, diff_method = "finite-diff"}

// CHECK-LABEL: @funcTensorScalar.finitediff0(%arg0: tensor<3xf64>) -> tensor<3xf64>
    // CHECK-DAG:    [[OPERANDCONST:%.+]] = arith.constant dense<3.000000e+00> : tensor<3xf64>
    // CHECK-DAG:    [[RESULTCONST:%.+]] = arith.constant 3.000000e+00 : f64
    // CHECK:        [[CALLPOS:%.+]] = call @funcTensorScalar(%arg0) : (tensor<3xf64>) -> f64
    // CHECK:        [[ABSDIFF:%.+]] = tensor.generate
    // CHECK:        [[RESULT:%.+]] = arith.divf [[ABSDIFF]], [[OPERANDCONST]] : tensor<3xf64>
    // CHECK-NEXT:   return [[RESULT]]
// }

// CHECK-LABEL: @gradCallTensorScalar
func.func @gradCallTensorScalar(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    // CHECK:   [[GRAD:%.+]] = call @funcTensorScalar.finitediff0(%arg0) : (tensor<3xf64>) -> tensor<3xf64>
    %2 = gradient.grad "fd"  @funcTensorScalar(%arg0) { finiteDiffParam = 3.000000e+00 : f64 } : (tensor<3xf64>) -> tensor<3xf64>
    // CHECK:   return [[GRAD]]
    func.return %2 : tensor<3xf64>
}

// -----

// Check tensor to tensor case
func.func private @funcTensorTensor(%arg0: tensor<7x3x2x1xf64>) -> tensor<2xf64> attributes {qnode, diff_method = "finite-diff"}

// CHECK-LABEL: @funcTensorTensor.finitediff0(%arg0: tensor<7x3x2x1xf64>) -> tensor<2x7x3x2x1xf64>
    // CHECK-DAG:    [[RESULTCONST:%.+]] = arith.constant dense<4.000000e+00> : tensor<2x7x3x2x1xf64>
    // CHECK-DAG:    [[BASETYPECONST:%.+]] = arith.constant 4.000000e+00 : f64
    // CHECK:        [[CALLPOS:%.+]] = call @funcTensorTensor(%arg0) : (tensor<7x3x2x1xf64>) -> tensor<2xf64>
    // CHECK:        [[ABSDIFF:%.+]] = tensor.generate
    // CHECK:        [[RESULT:%.+]] = arith.divf [[ABSDIFF]], [[RESULTCONST]] : tensor<2x7x3x2x1xf64>
    // CHECK-NEXT:   return [[RESULT]]
// }

// CHECK-LABEL: @gradCallTensorTensor
func.func @gradCallTensorTensor(%arg0: tensor<7x3x2x1xf64>) -> tensor<2x7x3x2x1xf64> {
    // CHECK:   [[GRAD:%.+]] = call @funcTensorTensor.finitediff0(%arg0) : (tensor<7x3x2x1xf64>) -> tensor<2x7x3x2x1xf64>
    %2 = gradient.grad "fd"  @funcTensorTensor(%arg0) { finiteDiffParam = 4.000000e+00 : f64 } : (tensor<7x3x2x1xf64>) -> tensor<2x7x3x2x1xf64>
    // CHECK:   return [[GRAD]]
    func.return %2 : tensor<2x7x3x2x1xf64>
}

// -----

// Check multiple arguments case
func.func private @funcMultiArg(%arg0: tensor<7xf64>, %arg1: f64) -> tensor<2xf64> attributes {qnode, diff_method = "finite-diff"}

// CHECK-LABEL: @funcMultiArg.finitediff0(%arg0: tensor<7xf64>, %arg1: f64) -> tensor<2x7xf64>
    // CHECK:        [[BASE:%.+]] = call @funcMultiArg(%arg0, %arg1)
    // CHECK:        [[DIFF:%.+]] = tensor.generate
    // CHECK-NEXT:   ^bb0(%arg2: index, %arg3: index):
    // CHECK:            [[VAL:%.+]] = tensor.extract %arg0[%arg3]
    // CHECK:            [[ADD:%.+]] = arith.addf [[VAL]]
    // CHECK:            [[SHIFTED:%.+]] = tensor.insert [[ADD]] into %arg0[%arg3]
    // CHECK:            [[EVAL:%.+]] = func.call @funcMultiArg([[SHIFTED]], %arg1)
    // CHECK:            [[SUB:%.+]] = arith.subf [[EVAL]], [[BASE]]
    // CHECK:            [[RES:%.+]] = tensor.extract [[SUB]][%arg2]
    // CHECK:            tensor.yield [[RES]]
    // CHECK:        [[RESULT:%.+]] = arith.divf [[DIFF]]
    // CHECK-NEXT:   return [[RESULT]]
// }

// CHECK-LABEL: @funcMultiArg.finitediff1(%arg0: tensor<7xf64>, %arg1: f64) -> tensor<2xf64>
    // CHECK:        [[BASE:%.+]] = call @funcMultiArg(%arg0, %arg1)
    // CHECK:        [[SHIFTED:%.+]] = arith.addf %arg1
    // CHECK:        [[EVAL:%.+]] = call @funcMultiArg(%arg0, [[SHIFTED]])
    // CHECK:        [[DIFF:%.+]] = arith.subf [[EVAL]], [[BASE]]
    // CHECK:        [[RESULT:%.+]] = arith.divf [[DIFF]]
    // CHECK-NEXT:   return [[RESULT]]
// }

// CHECK-LABEL: @funcMultiArg.finitediff01(%arg0: tensor<7xf64>, %arg1: f64) -> (tensor<2x7xf64>, tensor<2xf64>)
    // CHECK:        [[BASE:%.+]] = call @funcMultiArg(%arg0, %arg1)
    // CHECK:        [[DIFF:%.+]] = tensor.generate
    // CHECK-NEXT:   ^bb0(%arg2: index, %arg3: index):
    // CHECK:            [[VAL:%.+]] = tensor.extract %arg0[%arg3]
    // CHECK:            [[ADD:%.+]] = arith.addf [[VAL]]
    // CHECK:            [[SHIFTED:%.+]] = tensor.insert [[ADD]] into %arg0[%arg3]
    // CHECK:            [[EVAL:%.+]] = func.call @funcMultiArg([[SHIFTED]], %arg1)
    // CHECK:            [[SUB:%.+]] = arith.subf [[EVAL]], [[BASE]]
    // CHECK:            [[RES:%.+]] = tensor.extract [[SUB]][%arg2]
    // CHECK:            tensor.yield [[RES]]
    // CHECK:        [[R0:%.+]] = arith.divf [[DIFF]]
    // CHECK:        [[SHIFTED:%.+]] = arith.addf %arg1
    // CHECK:        [[EVAL:%.+]] = call @funcMultiArg(%arg0, [[SHIFTED]])
    // CHECK:        [[DIFF:%.+]] = arith.subf [[EVAL]], [[BASE]]
    // CHECK:        [[R1:%.+]] = arith.divf [[DIFF]]
    // CHECK-NEXT:   return [[R0]], [[R1]]
// }

// CHECK-LABEL: @gradCallMultiArg
func.func @gradCallMultiArg(%arg0: tensor<7xf64>, %arg1: f64) -> (tensor<2x7xf64>, tensor<2xf64>, tensor<2x7xf64>, tensor<2xf64>)  {
    // CHECK:   [[GRAD0:%.+]] = call @funcMultiArg.finitediff0(%arg0, %arg1) : (tensor<7xf64>, f64) -> tensor<2x7xf64>
    %0 = gradient.grad "fd"  @funcMultiArg(%arg0, %arg1) : (tensor<7xf64>, f64) -> tensor<2x7xf64>
    // CHECK:   [[GRAD1:%.+]] = call @funcMultiArg.finitediff1(%arg0, %arg1) : (tensor<7xf64>, f64) -> tensor<2xf64>
    %1 = gradient.grad "fd"  @funcMultiArg(%arg0, %arg1) {diffArgIndices = dense<[1]> : tensor<1xindex>} : (tensor<7xf64>, f64) -> tensor<2xf64>
    // CHECK:   [[GRAD2:%.+]]:2 = call @funcMultiArg.finitediff01(%arg0, %arg1) : (tensor<7xf64>, f64) -> (tensor<2x7xf64>, tensor<2xf64>)
    %2:2 = gradient.grad "fd"  @funcMultiArg(%arg0, %arg1) {diffArgIndices = dense<[0, 1]> : tensor<2xindex>} : (tensor<7xf64>, f64) -> (tensor<2x7xf64>, tensor<2xf64>)
    // CHECK:   return [[GRAD0]], [[GRAD1]], [[GRAD2]]#0, [[GRAD2]]#1
    func.return %0, %1, %2#0, %2#1 : tensor<2x7xf64>, tensor<2xf64>, tensor<2x7xf64>, tensor<2xf64>
}

// -----

// Check multiple results case
func.func private @funcMultiRes(%arg0: tensor<7xf64>) -> (f64, tensor<2xf64>) attributes {qnode, diff_method = "finite-diff"}

// CHECK-LABEL: @funcMultiRes.finitediff0(%arg0: tensor<7xf64>) -> (tensor<7xf64>, tensor<2x7xf64>)
    // CHECK:        [[BASE:%.+]]:2 = call @funcMultiRes(%arg0)
    // CHECK:        [[DIFF:%.+]] = tensor.generate
    // CHECK-NEXT:   ^bb0(%arg1: index):
    // CHECK:            [[VAL:%.+]] = tensor.extract %arg0[%arg1]
    // CHECK:            [[ADD:%.+]] = arith.addf [[VAL]]
    // CHECK:            [[SHIFTED:%.+]] = tensor.insert [[ADD]] into %arg0[%arg1]
    // CHECK:            [[EVAL:%.+]]:2 = func.call @funcMultiRes([[SHIFTED]])
    // CHECK:            [[RES:%.+]] = arith.subf [[EVAL]]#0, [[BASE]]#0
    // CHECK:            tensor.yield [[RES]]
    // CHECK:        [[R0:%.+]] = arith.divf [[DIFF]]
    // CHECK:        [[DIFF:%.+]] = tensor.generate
    // CHECK-NEXT:   ^bb0(%arg1: index, %arg2: index):
    // CHECK:            [[VAL:%.+]] = tensor.extract %arg0[%arg2]
    // CHECK:            [[ADD:%.+]] = arith.addf [[VAL]]
    // CHECK:            [[SHIFTED:%.+]] = tensor.insert [[ADD]] into %arg0[%arg2]
    // CHECK:            [[EVAL:%.+]]:2 = func.call @funcMultiRes([[SHIFTED]])
    // CHECK:            [[SUB:%.+]] = arith.subf [[EVAL]]#1, [[BASE]]#1
    // CHECK:            [[RES:%.+]] = tensor.extract [[SUB]][%arg1]
    // CHECK:            tensor.yield [[RES]]
    // CHECK:        [[R1:%.+]] = arith.divf [[DIFF]]
    // CHECK:        return [[R0]], [[R1]]
// }

// CHECK-LABEL: @gradCallMultiRes
func.func @gradCallMultiRes(%arg0: tensor<7xf64>) -> (tensor<7xf64>, tensor<2x7xf64>)  {
    // CHECK:   [[GRAD:%.+]]:2 = call @funcMultiRes.finitediff0(%arg0) : (tensor<7xf64>) -> (tensor<7xf64>, tensor<2x7xf64>)
    %0:2 = gradient.grad "fd"  @funcMultiRes(%arg0) : (tensor<7xf64>) -> (tensor<7xf64>, tensor<2x7xf64>)
    // CHECK:   return [[GRAD]]#0, [[GRAD]]#1
    func.return %0#0, %0#1 : tensor<7xf64>, tensor<2x7xf64>
}

// -----

// Check dynamic tensor shape case
func.func private @funcDynamicTensor(%arg0: tensor<?x3xf64>) -> tensor<2x?xf64> attributes {qnode, diff_method = "finite-diff"}

// CHECK-LABEL: @funcDynamicTensor.finitediff0(%arg0: tensor<?x3xf64>) -> tensor<2x?x?x3xf64>
    // CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
    // CHECK-DAG:    [[C1:%.+]] = arith.constant 1 : index
    // CHECK-DAG:    [[F64:%.+]] = arith.constant 1.000000e+00 : f64
    // CHECK-DAG:    [[BASE:%.+]] = call @funcDynamicTensor(%arg0)
    
    // CHECK:        [[DDIM0:%.+]] = tensor.dim [[BASE]], [[C1]]
    // CHECK:        [[DDIM1:%.+]] = tensor.dim %arg0, [[C0]]
    // CHECK:        [[INIT:%.+]] = tensor.empty([[DDIM0]], [[DDIM1]])
    // CHECK:        [[H:%.+]] = linalg.fill ins([[F64]] : f64) outs([[INIT]] : tensor<2x?x?x3xf64>)

    // CHECK:        [[DIFF:%.+]] = tensor.generate [[DDIM0]], [[DDIM1]]
    // CHECK-NEXT:   ^bb0([[i0:%.+]]: index, [[i1:%.+]]: index, [[i2:%.+]]: index, [[i3:%.+]]: index):
    // CHECK:            [[VAL:%.+]] = tensor.extract %arg0[[[i2]], [[i3]]]
    // CHECK:            [[ADD:%.+]] = arith.addf [[VAL]], [[F64]]
    // CHECK:            [[SHIFTED:%.+]] = tensor.insert [[ADD]] into %arg0[[[i2]], [[i3]]]
    // CHECK:            [[EVAL:%.+]] = func.call @funcDynamicTensor([[SHIFTED]])
    // CHECK:            [[SUB:%.+]] = arith.subf [[EVAL]], [[BASE]]
    // CHECK:            [[RES:%.+]] = tensor.extract [[SUB]][[[i0]], [[i1]]]
    // CHECK:            tensor.yield [[RES]]

    // CHECK:        [[RESULT:%.+]] = arith.divf [[DIFF]], [[H]]
    // CHECK-NEXT:   return [[RESULT]]
// }

// CHECK-LABEL: @gradCallDynamicTensor
func.func @gradCallDynamicTensor(%arg0: tensor<?x3xf64>) -> tensor<2x?x?x3xf64> {
    // CHECK:   [[GRAD:%.+]] = call @funcDynamicTensor.finitediff0(%arg0) : (tensor<?x3xf64>) -> tensor<2x?x?x3xf64>
    %0 = gradient.grad "fd"  @funcDynamicTensor(%arg0) { finiteDiffParam = 1.000000e+00 : f64 } : (tensor<?x3xf64>) -> tensor<2x?x?x3xf64>
    // CHECK:   return [[GRAD]]
    func.return %0 : tensor<2x?x?x3xf64>
}

// -----

// Check multiple grad calls to same function
func.func private @funcMultiCall(%arg0: f64) -> f64 attributes {qnode, diff_method = "finite-diff"}

// CHECK-LABEL: @funcMultiCall.finitediff0(%arg0: f64) -> f64

// CHECK-LABEL: @gradCallMultiCall
func.func @gradCallMultiCall(%arg0: f64) -> (f64, f64) {
    // CHECK:   [[GRAD0:%.+]] = call @funcMultiCall.finitediff0(%arg0) : (f64) -> f64
    %0 = gradient.grad "fd" @funcMultiCall(%arg0) : (f64) -> f64
    // CHECK:   [[GRAD1:%.+]] = call @funcMultiCall.finitediff0(%arg0) : (f64) -> f64
    %1 = gradient.grad "fd" @funcMultiCall(%arg0) : (f64) -> f64
    // CHECK:   return [[GRAD0]], [[GRAD1]]
    func.return %0, %1 : f64, f64
}
