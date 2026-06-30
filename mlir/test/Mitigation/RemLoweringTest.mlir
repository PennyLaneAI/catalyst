// Copyright 2026 Haiqu, Inc.

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

// ===================================================================
// computeAllZeroesOnes(true) on a ProbsMP callee: the pass must keep the
// original func.call, then emit two cloned quantum.device / quantum.alloc /
// quantum.compbasis / quantum.probs blocks. The all-ones clone is
// distinguished from the all-zeroes one by per-qubit PauliX gates inserted
// before the compbasis observation.
// ===================================================================

func.func @probsCircuit() -> tensor<4xf64> attributes {qnode} {
    %shots = arith.constant 0 : i64
    quantum.device shots(%shots) ["rtd_lightning.so", "LightningQubit", "{}"]
    %r = quantum.alloc(2) : !quantum.reg
    %idx0 = arith.constant 0 : i64
    %idx1 = arith.constant 1 : i64
    %q_0 = quantum.extract %r[%idx0] : !quantum.reg -> !quantum.bit
    %q_1 = quantum.extract %r[%idx1] : !quantum.reg -> !quantum.bit
    %h_0 = quantum.custom "h"() %q_0 : !quantum.bit
    %r_0 = quantum.insert %r[ 0], %h_0 : !quantum.reg, !quantum.bit
    %r_1 = quantum.insert %r_0[ 1], %q_1 : !quantum.reg, !quantum.bit
    %obs = quantum.compbasis qubits %h_0, %q_1 : !quantum.obs
    %probs = quantum.probs %obs : tensor<4xf64>
    quantum.dealloc %r_1 : !quantum.reg
    quantum.device_release
    func.return %probs : tensor<4xf64>
}

func.func @remProbsCalibrate() -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) {
    %out:3 = mitigation.rem @probsCircuit() computeAllZeroesOnes(true) : () -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>)
    func.return %out#0, %out#1, %out#2 : tensor<4xf64>, tensor<4xf64>, tensor<4xf64>
}

// CHECK-LABEL: func.func @remProbsCalibrate
// CHECK:       call @probsCircuit() : () -> tensor<4xf64>
// CHECK:       quantum.device shots({{.*}}) ["rtd_lightning.so", "LightningQubit", "{}"]
// CHECK:       quantum.alloc(%{{.*}}) : !quantum.reg
// CHECK:       quantum.compbasis
// CHECK:       quantum.probs {{.*}} : tensor<4xf64>
// CHECK:       quantum.device_release
// CHECK:       quantum.device shots({{.*}}) ["rtd_lightning.so", "LightningQubit", "{}"]
// CHECK:       quantum.alloc(%{{.*}}) : !quantum.reg
// CHECK:       quantum.custom "PauliX"
// CHECK:       quantum.custom "PauliX"
// CHECK:       quantum.compbasis
// CHECK:       quantum.probs {{.*}} : tensor<4xf64>
// CHECK:       quantum.device_release

// -----

// ===================================================================
// computeAllZeroesOnes(true) on a CountsMP callee: callee result is the
// (eigvals, counts) pair, and the rem op surfaces two i64 calibration
// tensors -- one per all-zeroes / all-ones run.
// ===================================================================

func.func @countsCircuit() -> (tensor<4xf64>, tensor<4xi64>) attributes {qnode} {
    %shots = arith.constant 100 : i64
    quantum.device shots(%shots) ["rtd_lightning.so", "LightningQubit", "{}"]
    %r = quantum.alloc(2) : !quantum.reg
    %idx0 = arith.constant 0 : i64
    %idx1 = arith.constant 1 : i64
    %q_0 = quantum.extract %r[%idx0] : !quantum.reg -> !quantum.bit
    %q_1 = quantum.extract %r[%idx1] : !quantum.reg -> !quantum.bit
    %h_0 = quantum.custom "h"() %q_0 : !quantum.bit
    %r_0 = quantum.insert %r[ 0], %h_0 : !quantum.reg, !quantum.bit
    %r_1 = quantum.insert %r_0[ 1], %q_1 : !quantum.reg, !quantum.bit
    %obs = quantum.compbasis qubits %h_0, %q_1 : !quantum.obs
    %eig, %cnt = quantum.counts %obs : tensor<4xf64>, tensor<4xi64>
    quantum.dealloc %r_1 : !quantum.reg
    quantum.device_release
    func.return %eig, %cnt : tensor<4xf64>, tensor<4xi64>
}

func.func @remCountsCalibrate() -> (tensor<4xf64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) {
    %out:4 = mitigation.rem @countsCircuit() computeAllZeroesOnes(true) : () -> (tensor<4xf64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>)
    func.return %out#0, %out#1, %out#2, %out#3 : tensor<4xf64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>
}

// CHECK-LABEL: func.func @remCountsCalibrate
// CHECK:       call @countsCircuit() : () -> (tensor<4xf64>, tensor<4xi64>)
// CHECK:       quantum.counts {{.*}} : tensor<4xf64>, tensor<4xi64>
// CHECK:       quantum.custom "PauliX"
// CHECK:       quantum.counts {{.*}} : tensor<4xf64>, tensor<4xi64>

// -----

// ===================================================================
// computeAllZeroesOnes(true) on a SampleMP callee with shots = 200: the
// calibration tensors take the (shots, qubits) shape inferred from the
// callee's quantum.device + quantum.alloc.
// ===================================================================

func.func @sampleCircuit() -> tensor<200x2xf64> attributes {qnode} {
    %shots = arith.constant 200 : i64
    quantum.device shots(%shots) ["rtd_lightning.so", "LightningQubit", "{}"]
    %r = quantum.alloc(2) : !quantum.reg
    %idx0 = arith.constant 0 : i64
    %idx1 = arith.constant 1 : i64
    %q_0 = quantum.extract %r[%idx0] : !quantum.reg -> !quantum.bit
    %q_1 = quantum.extract %r[%idx1] : !quantum.reg -> !quantum.bit
    %h_0 = quantum.custom "h"() %q_0 : !quantum.bit
    %r_0 = quantum.insert %r[ 0], %h_0 : !quantum.reg, !quantum.bit
    %r_1 = quantum.insert %r_0[ 1], %q_1 : !quantum.reg, !quantum.bit
    %obs = quantum.compbasis qubits %h_0, %q_1 : !quantum.obs
    %samples = quantum.sample %obs : tensor<200x2xf64>
    quantum.dealloc %r_1 : !quantum.reg
    quantum.device_release
    func.return %samples : tensor<200x2xf64>
}

func.func @remSampleCalibrate() -> (tensor<200x2xf64>, tensor<200x2xf64>, tensor<200x2xf64>) {
    %out:3 = mitigation.rem @sampleCircuit() computeAllZeroesOnes(true) : () -> (tensor<200x2xf64>, tensor<200x2xf64>, tensor<200x2xf64>)
    func.return %out#0, %out#1, %out#2 : tensor<200x2xf64>, tensor<200x2xf64>, tensor<200x2xf64>
}

// CHECK-LABEL: func.func @remSampleCalibrate
// CHECK:       call @sampleCircuit() : () -> tensor<200x2xf64>
// CHECK:       quantum.sample {{.*}} : tensor<200x2xf64>
// CHECK:       quantum.custom "PauliX"
// CHECK:       quantum.sample {{.*}} : tensor<200x2xf64>

// -----

// ===================================================================
// computeAllZeroesOnes(false): the pass forwards the callee unchanged and
// stamps two zero-filled placeholder tensors for the calibration slots --
// no cloned circuits and no extra quantum.device calls.
// ===================================================================

func.func @passthroughCircuit() -> tensor<4xf64> attributes {qnode} {
    %shots = arith.constant 0 : i64
    quantum.device shots(%shots) ["rtd_lightning.so", "LightningQubit", "{}"]
    %r = quantum.alloc(2) : !quantum.reg
    %idx0 = arith.constant 0 : i64
    %q_0 = quantum.extract %r[%idx0] : !quantum.reg -> !quantum.bit
    %h_0 = quantum.custom "h"() %q_0 : !quantum.bit
    %r_0 = quantum.insert %r[ 0], %h_0 : !quantum.reg, !quantum.bit
    %obs = quantum.compbasis qubits %h_0 : !quantum.obs
    %probs = quantum.probs %obs : tensor<4xf64>
    quantum.dealloc %r_0 : !quantum.reg
    quantum.device_release
    func.return %probs : tensor<4xf64>
}

func.func @remProbsPassthrough() -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) {
    %out:3 = mitigation.rem @passthroughCircuit() computeAllZeroesOnes(false) : () -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>)
    func.return %out#0, %out#1, %out#2 : tensor<4xf64>, tensor<4xf64>, tensor<4xf64>
}

// CHECK-LABEL: func.func @remProbsPassthrough
// CHECK:       arith.constant dense<0.000000e+00> : tensor<4xf64>
// CHECK:       call @passthroughCircuit() : () -> tensor<4xf64>
// CHECK-NOT:   quantum.compbasis
// CHECK-NOT:   quantum.custom "PauliX"

// -----

// ===================================================================
// A missing callee leaves the mitigation.rem op untouched: the lowering
// pattern bails out via notifyMatchFailure, which keeps the IR in a state
// that downstream passes can either re-try or surface as a hard error.
// ===================================================================

func.func @remMissingCallee() -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) {
    %out:3 = mitigation.rem @doesNotExist() computeAllZeroesOnes(true) : () -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>)
    func.return %out#0, %out#1, %out#2 : tensor<4xf64>, tensor<4xf64>, tensor<4xf64>
}

// CHECK-LABEL: func.func @remMissingCallee
// CHECK:       mitigation.rem @doesNotExist() computeAllZeroesOnes(true)
