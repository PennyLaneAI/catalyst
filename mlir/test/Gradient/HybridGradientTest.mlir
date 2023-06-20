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

