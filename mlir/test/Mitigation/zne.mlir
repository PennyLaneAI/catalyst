// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt %s --lower-mitigation --split-input-file --verify-diagnostics | FileCheck %s

func.func @simpleCircuit(%arg0: tensor<3xf64>) -> f64 attributes {qnode} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %f0 = tensor.extract %arg0[%c0] : tensor<3xf64>
    %f1 = tensor.extract %arg0[%c1] : tensor<3xf64>
    %f2 = tensor.extract %arg0[%c2] : tensor<3xf64>

    %idx = arith.index_cast %c0 : index to i64
    %r = quantum.alloc(1) : !quantum.reg

    %q_0 = quantum.extract %r[%idx] : !quantum.reg -> !quantum.bit
    %q_1 = quantum.custom "h"() %q_0 : !quantum.bit
    %q_2 = quantum.custom "rz"(%f0) %q_1 : !quantum.bit
    %q_3 = quantum.custom "u3"(%f0, %f1, %f2) %q_2 : !quantum.bit

    %12 = quantum.insert %r[ 0], %q_3 : !quantum.reg, !quantum.bit
    %obs = quantum.namedobs %q_3[PauliX] : !quantum.obs
    %expval = quantum.expval %obs : f64
    quantum.dealloc %r : !quantum.reg
    func.return %expval : f64
}
 
func.func @zneCallScalarScalar(%arg0: tensor<3xf64>) -> f64 {
    %scalarFactors = arith.constant dense<[1, 2, 3]> : tensor<3xindex>
    %0 = mitigation.zne @simpleCircuit(%arg0) scalarFactors (%scalarFactors : tensor<3xindex>) : (tensor<3xf64>) -> f64
    func.return %0 : f64
}