// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt %s --lower-mitigation | FileCheck %s

// Classical helpers inside a folded QNode retain their original calling convention.
// CHECK-DAG: call @angle({{%.+}}) : (tensor<1xf64>) -> f64
// CHECK-DAG: call @circuit.folded({{%.+}}, {{%.+}}) : (tensor<1xf64>, index) -> f64

func.func @angle(%arg0: tensor<1xf64>) -> f64 {
    %index = arith.constant 0 : index
    %value = tensor.extract %arg0[%index] : tensor<1xf64>
    return %value : f64
}

func.func @circuit(%arg0: tensor<1xf64>) -> f64 attributes {qnode} {
    %shots = arith.constant 0 : i64
    quantum.device shots(%shots) ["rtd_lightning.so", "LightningQubit", "{}"]
    %angle = call @angle(%arg0) : (tensor<1xf64>) -> f64
    %reg = quantum.alloc(1) : !quantum.reg
    %qubit = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %rotated = quantum.custom "RX"(%angle) %qubit : !quantum.bit
    %updated = quantum.insert %reg[0], %rotated : !quantum.reg, !quantum.bit
    %obs = quantum.namedobs %rotated[PauliZ] : !quantum.obs
    %result = quantum.expval %obs : f64
    quantum.dealloc %updated : !quantum.reg
    quantum.device_release
    return %result : f64
}

func.func @workflow(%arg0: tensor<1xf64>) -> f64 {
    %result = call @circuit(%arg0) : (tensor<1xf64>) -> f64
    return %result : f64
}

func.func @mitigated(%arg0: tensor<1xf64>) -> tensor<2xf64> {
    %folds = arith.constant dense<[0.0, 1.0]> : tensor<2xf64>
    %result = mitigation.zne @workflow(%arg0) folding(global) numFolds(%folds : tensor<2xf64>) : (tensor<1xf64>) -> tensor<2xf64>
    return %result : tensor<2xf64>
}
