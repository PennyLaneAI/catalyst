// Copyright 20234 Xanadu Quantum Technologies Inc.

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


func.func @circuit1(%arg0: tensor<3xf64>) -> f64 attributes {qnode} {
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



func.func @circuit2(%arg0: tensor<3xf64>) -> f64 attributes {qnode} {
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
    %q_1 = quantum.custom "x"() %q_0 : !quantum.bit
    %q_2 = quantum.custom "rx"(%f0) %q_1 : !quantum.bit
    %q_3 = quantum.custom "u3"(%f0, %f1, %f2) %q_2 : !quantum.bit

    %12 = quantum.insert %r[ 0], %q_3 : !quantum.reg, !quantum.bit
    %obs = quantum.namedobs %q_3[PauliX] : !quantum.obs
    %expval = quantum.expval %obs : f64
    quantum.dealloc %12 : !quantum.reg
    quantum.device_release
    func.return %expval : f64
}


func.func @multipleQnodes(%arg0: tensor<3xf64>) -> f64 {
    %0 = call @circuit1(%arg0) : (tensor<3xf64>) -> f64
    %1 = call @circuit2(%arg0) : (tensor<3xf64>) -> f64
    %2 = arith.addf %1, %0 : f64
    return %2 : f64
}

// CHECK:    func.func private @circuit2.folded(%arg0: tensor<3xf64>, %arg1: index) -> f64 {
// CHECK:    func.func private @circuit2.quantumAlloc(%arg0: i64) -> !quantum.reg {
// CHECK:    func.func private @circuit2.withoutMeasurements(%arg0: tensor<3xf64>, %arg1: !quantum.reg) -> !quantum.reg {
// CHECK:    func.func private @circuit2.withMeasurements(%arg0: tensor<3xf64>, %arg1: !quantum.reg) -> f64 {

// CHECK:    func.func private @circuit1.folded(%arg0: tensor<3xf64>, %arg1: index) -> f64 {
// CHECK:    func.func private @circuit1.quantumAlloc(%arg0: i64) -> !quantum.reg {
// CHECK:    func.func private @circuit1.withoutMeasurements(%arg0: tensor<3xf64>, %arg1: !quantum.reg) -> !quantum.reg {
// CHECK:    func.func private @circuit1.withMeasurements(%arg0: tensor<3xf64>, %arg1: !quantum.reg) -> f64 {

// CHECK:    func.func @multipleQnodes.zne(%arg0: tensor<3xf64>, %arg1: index) -> f64 {
// CHECK:      %0 = call @circuit1.folded(%arg0, %arg1) : (tensor<3xf64>, index) -> f64
// CHECK:      %1 = call @circuit2.folded(%arg0, %arg1) : (tensor<3xf64>, index) -> f64
// CHECK:      %2 = arith.addf %1, %0 : f64
// CHECK:      return %2 : f64


// CHECK:    func.func @qjitZne(%arg0: tensor<3xf64>) -> tensor<5xf64> {
// CHECK:    scf.for
// CHECK:      func.call @multipleQnodes.zne

func.func @qjitZne(%arg0: tensor<3xf64>) -> tensor<5xf64> {
    %numFolds = arith.constant dense<[1, 2, 3, 4, 5]> : tensor<5xindex>
    %0 = mitigation.zne @multipleQnodes(%arg0) folding (global) numFolds (%numFolds : tensor<5xindex>) : (tensor<3xf64>) -> tensor<5xf64>
    func.return %0 : tensor<5xf64>
}

// -----

func.func @circuit1(%arg0: tensor<3xf64>) -> f64 attributes {qnode} {
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



func.func @circuit2(%arg0: tensor<3xf64>) -> f64 attributes {qnode} {
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
    %q_1 = quantum.custom "x"() %q_0 : !quantum.bit
    %q_2 = quantum.custom "rx"(%f0) %q_1 : !quantum.bit
    %q_3 = quantum.custom "u3"(%f0, %f1, %f2) %q_2 : !quantum.bit

    %12 = quantum.insert %r[ 0], %q_3 : !quantum.reg, !quantum.bit
    %obs = quantum.namedobs %q_3[PauliX] : !quantum.obs
    %expval = quantum.expval %obs : f64
    quantum.dealloc %12 : !quantum.reg
    quantum.device_release
    func.return %expval : f64
}

func.func @intermediateClassical(%arg0: f64) -> f64 {
    func.return %arg0 : f64
}

func.func @intermediateWithQ(%arg0: tensor<3xf64>) -> f64 {
    %1 = call @circuit1(%arg0) : (tensor<3xf64>) -> f64
    func.return %1 : f64
}


func.func @multipleQnodes(%arg0: tensor<3xf64>, %arg1: f64) -> f64 {
    %0 = call @intermediateClassical(%arg1) : (f64) -> f64
    %1 = call @intermediateWithQ(%arg0) : (tensor<3xf64>) -> f64
    %2 = call @circuit2(%arg0) : (tensor<3xf64>) -> f64
    %3 = arith.addf %1, %0 : f64
    %4 = arith.addf %2, %3 : f64
    return %4 : f64
}

// CHECK:    func.func private @circuit1.folded(%arg0: tensor<3xf64>, %arg1: index) -> f64 {
// CHECK:    func.func private @circuit1.quantumAlloc(%arg0: i64) -> !quantum.reg {
// CHECK:    func.func private @circuit1.withoutMeasurements(%arg0: tensor<3xf64>, %arg1: !quantum.reg) -> !quantum.reg {
// CHECK:    func.func private @circuit1.withMeasurements(%arg0: tensor<3xf64>, %arg1: !quantum.reg) -> f64 {

// CHECK:    func.func @intermediateWithQ.zne(%arg0: tensor<3xf64>, %arg1: index) -> f64 {
// CHECK:      call @circuit1.folded(%arg0, %arg1) : (tensor<3xf64>, index) -> f64

// CHECK:    func.func @intermediateClassical.zne(%arg0: f64, %arg1: index) -> f64 {
// CHECK-NEXT:    return %arg0 : f64

// CHECK:    func.func private @circuit2.folded(%arg0: tensor<3xf64>, %arg1: index) -> f64 {
// CHECK:    func.func private @circuit2.quantumAlloc(%arg0: i64) -> !quantum.reg {
// CHECK:    func.func private @circuit2.withoutMeasurements(%arg0: tensor<3xf64>, %arg1: !quantum.reg) -> !quantum.reg {
// CHECK:    func.func private @circuit2.withMeasurements(%arg0: tensor<3xf64>, %arg1: !quantum.reg) -> f64 {

// CHECK:    func.func @multipleQnodes.zne(%arg0: tensor<3xf64>, %arg1: f64, %arg2: index) -> f64 {
// CHECK:      call @intermediateClassical.zne(%arg1, %arg2) : (f64, index) -> f64
// CHECK:      call @intermediateWithQ.zne(%arg0, %arg2) : (tensor<3xf64>, index) -> f64
// CHECK:     call @circuit2.folded(%arg0, %arg2) : (tensor<3xf64>, index) -> f64

// CHECK:    func.func @qjitZne(%arg0: tensor<3xf64>, %arg1: f64) -> tensor<5xf64> {
// CHECK:      scf.for
// CHECK:        func.call @multipleQnodes.zne(%arg0, %arg1, %extracted) : (tensor<3xf64>, f64, index) -> f64

func.func @qjitZne(%arg0: tensor<3xf64>, %arg1: f64) -> tensor<5xf64> {
    %numFolds = arith.constant dense<[1, 2, 3, 4, 5]> : tensor<5xindex>
    %0 = mitigation.zne @multipleQnodes(%arg0, %arg1) folding (global) numFolds (%numFolds : tensor<5xindex>) : (tensor<3xf64>, f64) -> tensor<5xf64>
    func.return %0 : tensor<5xf64>
}


// -----

func.func @circuit1(%arg0: tensor<3xf64>) -> f64 attributes {qnode} {
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


func.func @intermediate1(%arg0: tensor<3xf64>) -> f64 {
    %1 = call @intermediate2(%arg0) : (tensor<3xf64>) -> f64
    func.return %1 : f64
}

func.func @intermediate2(%arg0: tensor<3xf64>) -> f64 {
    %1 = call @intermediate3(%arg0) : (tensor<3xf64>) -> f64
    func.return %1 : f64
}

func.func @intermediate3(%arg0: tensor<3xf64>) -> f64 {
    %1 = call @intermediateLast(%arg0) : (tensor<3xf64>) -> f64
    func.return %1 : f64
}

func.func @intermediateLast(%arg0: tensor<3xf64>) -> f64 {
    %1 = call @circuit1(%arg0) : (tensor<3xf64>) -> f64
    func.return %1 : f64
}

func.func @multipleQnodes(%arg0: tensor<3xf64>, %arg1: f64) -> f64 {
    %0 = call @intermediate1(%arg0) : (tensor<3xf64>) -> f64
    %1 = arith.addf %arg1, %0 : f64
    %3 = arith.addf %0, %1 : f64
    return %3 : f64
}

// CHECK:    func.func private @circuit1.folded(%arg0: tensor<3xf64>, %arg1: index) -> f64 {
// CHECK:    func.func private @circuit1.quantumAlloc(%arg0: i64) -> !quantum.reg {
// CHECK:    func.func private @circuit1.withoutMeasurements(%arg0: tensor<3xf64>, %arg1: !quantum.reg) -> !quantum.reg {
// CHECK:    func.func private @circuit1.withMeasurements(%arg0: tensor<3xf64>, %arg1: !quantum.reg) -> f64 {

// CHECK:    func.func @intermediateLast.zne(%arg0: tensor<3xf64>, %arg1: index) -> f64 {
// CHECK:      call @circuit1.folded(%arg0, %arg1) : (tensor<3xf64>, index) -> f64

// CHECK:    func.func @intermediate3.zne(%arg0: tensor<3xf64>, %arg1: index) -> f64 {
// CHECK:      call @intermediateLast.zne(%arg0, %arg1) : (tensor<3xf64>, index) -> f64

// CHECK:    func.func @intermediate2.zne(%arg0: tensor<3xf64>, %arg1: index) -> f64 {
// CHECK:      call @intermediate3.zne(%arg0, %arg1) : (tensor<3xf64>, index) -> f64

// CHECK:    func.func @intermediate1.zne(%arg0: tensor<3xf64>, %arg1: index) -> f64 {
// CHECK:      call @intermediate2.zne(%arg0, %arg1) : (tensor<3xf64>, index) -> f64

// CHECK:    func.func @multipleQnodes.zne(%arg0: tensor<3xf64>, %arg1: f64, %arg2: index) -> f64 {
// CHECK:      call @intermediate1.zne(%arg0, %arg2) : (tensor<3xf64>, index) -> f64


func.func @qjitZne(%arg0: tensor<3xf64>, %arg1: f64) -> tensor<5xf64> {
    %numFolds = arith.constant dense<[1, 2, 3, 4, 5]> : tensor<5xindex>
    %0 = mitigation.zne @multipleQnodes(%arg0, %arg1) folding (global) numFolds (%numFolds : tensor<5xindex>) : (tensor<3xf64>, f64) -> tensor<5xf64>
    func.return %0 : tensor<5xf64>
}
