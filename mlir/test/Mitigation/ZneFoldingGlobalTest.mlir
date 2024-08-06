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

// CHECK-LABEL: func.func private @simpleCircuit.folded(%arg0: tensor<3xf64>, %arg1: index) -> f64 {
    // CHECK:      [[nQubits:%.+]] = arith.constant 1
    // CHECK:      [[c0:%.+]] = index.constant 0
    // CHECK:      [[c1:%.+]] = index.constant 1
    // CHECK:      quantum.device["rtd_lightning.so", "LightningQubit", "{shots: 0}"]
    // CHECK:      [[qReg:%.+]] = call @simpleCircuit.quantumAlloc([[nQubits]]) : (i64) -> !quantum.reg
    // CHECK:      [[outQregFor:%.+]]  = scf.for %arg2 = [[c0]] to %arg1 step [[c1]] iter_args([[inQreg:%.+]] = [[qReg]]) -> (!quantum.reg) {
        // CHECK:      [[outQreg1:%.+]] = func.call @simpleCircuit.withoutMeasurements(%arg0, [[inQreg]]) : (tensor<3xf64>, !quantum.reg) -> !quantum.reg
        // CHECK:      [[outQreg2:%.+]] = quantum.adjoint([[outQreg1]]) : !quantum.reg {
        // CHECK:      ^bb0(%arg4: !quantum.reg):
            // CHECK:      [[callWithoutMeasurements:%.+]] = func.call @simpleCircuit.withoutMeasurements(%arg0, %arg4) : (tensor<3xf64>, !quantum.reg) -> !quantum.reg
            // CHECK:      quantum.yield [[callWithoutMeasurements]] : !quantum.reg
        // CHECK:      scf.yield [[outQreg2]] : !quantum.reg
    // CHECK:      [[results:%.+]]  = call @simpleCircuit.withMeasurements(%arg0, [[outQregFor]]) : (tensor<3xf64>, !quantum.reg) -> f64
    // CHECK:      quantum.device_release
    // CHECK:      return [[results]]

// CHECK-LABEL: func.func private @simpleCircuit.quantumAlloc(%arg0: i64) -> !quantum.reg {
    // CHECK:     [[allocQreg:%.+]] = quantum.alloc(%arg0) : !quantum.reg
    // CHECK:     return [[allocQreg]] : !quantum.reg

// CHECK-LABEL:    func.func private @simpleCircuit.withoutMeasurements(%arg0: tensor<3xf64>, %arg1: !quantum.reg) -> !quantum.reg {
    // CHECK:    [[q_0:%.+]] = quantum.extract %arg1[ 0] : !quantum.reg -> !quantum.bit
    // CHECK:    [[q_1:%.+]] = quantum.custom "h"() [[q_0]] : !quantum.bit
    // CHECK:    [[q_2:%.+]] = quantum.custom "rz"({{.*}}) [[q_1]] : !quantum.bit
    // CHECK:    [[q_3:%.+]] = quantum.custom "u3"({{.*}}, {{.*}}, {{.*}}) [[q_2]] : !quantum.bit
    // CHECK:    [[q_4:%.+]] = quantum.insert %arg1[ 0], [[q_3]] : !quantum.reg, !quantum.bit
    // CHECK:    return [[q_4]] : !quantum.reg

// CHECK-LABEL:    func.func private @simpleCircuit.withMeasurements(%arg0: tensor<3xf64>, %arg1: !quantum.reg) -> f64 {
    // CHECK:    [[q_0:%.+]] = quantum.extract %arg1[ 0] : !quantum.reg -> !quantum.bit
    // CHECK:    [[q_1:%.+]] = quantum.custom "h"() [[q_0]] : !quantum.bit
    // CHECK:    [[q_2:%.+]] = quantum.custom "rz"({{.*}}) [[q_1]] : !quantum.bit
    // CHECK:    [[q_3:%.+]] = quantum.custom "u3"({{.*}}, {{.*}}, {{.*}}) [[q_2]] : !quantum.bit
    // CHECK:    [[q_4:%.+]] = quantum.insert %arg1[ 0], [[q_3]] : !quantum.reg, !quantum.bit
    // CHECK:    [[q_5:%.+]] = quantum.namedobs [[q_3]][ PauliX] : !quantum.obs
    // CHECK:    [[results:%.+]] = quantum.expval [[q_5]] : f64
    // CHECK:    quantum.dealloc [[q_4]] : !quantum.reg
    // CHECK:    return [[results]] : f64

// CHECK-LABEL: func.func @simpleCircuit
func.func @simpleCircuit(%arg0: tensor<3xf64>) -> f64 attributes {qnode} {
    quantum.device ["rtd_lightning.so", "LightningQubit", "{shots: 0}"]
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
    quantum.dealloc %12 : !quantum.reg
    quantum.device_release
    func.return %expval : f64
}

// CHECK-LABEL:    func.func @zneCallScalarScalar(%arg0: tensor<3xf64>) -> tensor<5xf64> {
    // CHECK:    [[c0:%.+]] = index.constant 0
    // CHECK:    [[c1:%.+]] = index.constant 1
    // CHECK:    [[c5:%.+]] = index.constant 5
    // CHECK:    [[dense5:%.+]] = arith.constant dense<[1, 2, 3, 4, 5]>
    // CHECK:    [[emptyRes:%.+]] = tensor.empty() : tensor<5xf64>
    // CHECK:    [[results:%.+]] = scf.for [[idx:%.+]] = [[c0]] to [[c5]] step [[c1]] iter_args(%arg2 = [[emptyRes]]) -> (tensor<5xf64>) {
        // CHECK:    [[scalarFactor:%.+]] = tensor.extract [[dense5]][[[idx]]] : tensor<5xindex>
        // CHECK:    [[intermediateRes:%.+]] = func.call @simpleCircuit.folded(%arg0, [[scalarFactor]]) : (tensor<3xf64>, index) -> f64
        // CHECK:    [[tensorRes:%.+]] = tensor.from_elements [[intermediateRes]] : tensor<1xf64>
            // CHECK:    [[resultsFor:%.+]] = scf.for [[idxJ:%.+]] = [[c0]] to [[c1]] step [[c1]] iter_args(%arg4 = %arg2) -> (tensor<5xf64>) {
                // CHECK:    [[extracted:%.+]] = tensor.extract [[tensorRes]][%arg3] : tensor<1xf64>
                // CHECK:    [[insertedRes:%.+]] = tensor.insert [[extracted]] into %arg4[%arg1] : tensor<5xf64>
                // CHECK:    scf.yield [[insertedRes]]
        // CHECK:    scf.yield [[resultsFor]]
    // CHECK:    return [[results]]
func.func @zneCallScalarScalar(%arg0: tensor<3xf64>) -> tensor<5xf64> {
    %scaleFactors = arith.constant dense<[1, 2, 3, 4, 5]> : tensor<5xindex>
    %0 = mitigation.zne @simpleCircuit(%arg0) folding (global) scaleFactors (%scaleFactors : tensor<5xindex>) : (tensor<3xf64>) -> tensor<5xf64>
    func.return %0 : tensor<5xf64>
}
