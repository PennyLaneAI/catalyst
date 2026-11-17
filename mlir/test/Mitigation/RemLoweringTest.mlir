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
// runCalibration(true) on a ProbsMP callee: the pass must keep the
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
    %out:3 = mitigation.rem @probsCircuit() runCalibration(true) {resultSegmentSizes = array<i32: 1, 1, 1>} : () -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>)
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
// runCalibration(true) on a CountsMP callee: callee result is the
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
    %out:4 = mitigation.rem @countsCircuit() runCalibration(true) {resultSegmentSizes = array<i32: 2, 1, 1>} : () -> (tensor<4xf64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>)
    func.return %out#0, %out#1, %out#2, %out#3 : tensor<4xf64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>
}

// CHECK-LABEL: func.func @remCountsCalibrate
// CHECK:       call @countsCircuit() : () -> (tensor<4xf64>, tensor<4xi64>)
// CHECK:       quantum.counts {{.*}} : tensor<4xf64>, tensor<4xi64>
// CHECK:       quantum.custom "PauliX"
// CHECK:       quantum.counts {{.*}} : tensor<4xf64>, tensor<4xi64>

// -----

// ===================================================================
// runCalibration(true) on a SampleMP callee with shots = 200: the
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
    %out:3 = mitigation.rem @sampleCircuit() runCalibration(true) {resultSegmentSizes = array<i32: 1, 1, 1>} : () -> (tensor<200x2xf64>, tensor<200x2xf64>, tensor<200x2xf64>)
    func.return %out#0, %out#1, %out#2 : tensor<200x2xf64>, tensor<200x2xf64>, tensor<200x2xf64>
}

// CHECK-LABEL: func.func @remSampleCalibrate
// CHECK:       call @sampleCircuit() : () -> tensor<200x2xf64>
// CHECK:       quantum.sample {{.*}} : tensor<200x2xf64>
// CHECK:       quantum.custom "PauliX"
// CHECK:       quantum.sample {{.*}} : tensor<200x2xf64>

// -----

// ===================================================================
// SampleMP callee that measures only wire 1 of its 2 qubits and converts
// its samples to integers. The calibration circuits still measure the
// whole register (tensor<200x2xf64>), and nothing after the callee's
// sample is cloned into them: neither the conversion, whose type follows
// the callee's measured wires, nor the quantum.insert of its measured
// qubit (at an index given by a constant, as in traced programs).
// ===================================================================

func.func @subsetSampleCircuit() -> tensor<200x1xi64> attributes {qnode} {
    %shots = arith.constant 200 : i64
    quantum.device shots(%shots) ["rtd_lightning.so", "LightningQubit", "{}"]
    %r = quantum.alloc(2) : !quantum.reg
    %q_1 = quantum.extract %r[ 1] : !quantum.reg -> !quantum.bit
    %obs = quantum.compbasis qubits %q_1 : !quantum.obs
    %samples = quantum.sample %obs : tensor<200x1xf64>
    %converted = arith.fptosi %samples : tensor<200x1xf64> to tensor<200x1xi64>
    %idx1 = arith.constant 1 : i64
    %r_0 = quantum.insert %r[%idx1], %q_1 : !quantum.reg, !quantum.bit
    quantum.dealloc %r_0 : !quantum.reg
    quantum.device_release
    func.return %converted : tensor<200x1xi64>
}

func.func @remSubsetSampleCalibrate() -> (tensor<200x1xi64>, tensor<200x2xf64>, tensor<200x2xf64>) {
    %out:3 = mitigation.rem @subsetSampleCircuit() runCalibration(true) {resultSegmentSizes = array<i32: 1, 1, 1>} : () -> (tensor<200x1xi64>, tensor<200x2xf64>, tensor<200x2xf64>)
    func.return %out#0, %out#1, %out#2 : tensor<200x1xi64>, tensor<200x2xf64>, tensor<200x2xf64>
}

// CHECK-LABEL: func.func @remSubsetSampleCalibrate
// CHECK:       call @subsetSampleCircuit() : () -> tensor<200x1xi64>
// CHECK:       [[REG:%.+]] = quantum.alloc(%{{.*}}) : !quantum.reg
// CHECK-NEXT:  [[OBS:%.+]] = quantum.compbasis qreg [[REG]]
// CHECK-NEXT:  [[SAMPLES:%.+]] = quantum.sample [[OBS]] : tensor<200x2xf64>
// CHECK-NEXT:  quantum.dealloc [[REG]]
// CHECK:       return {{.*}}[[SAMPLES]],

// -----

// ===================================================================
// runCalibration(false): the pass forwards the callee unchanged. With the
// zeros/ones result groups empty, the op replacement is just the callee's
// SSA values -- no placeholder constants and no calibration circuits.
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

func.func @remProbsPassthrough() -> tensor<4xf64> {
    %out = mitigation.rem @passthroughCircuit() runCalibration(false) {resultSegmentSizes = array<i32: 1, 0, 0>} : () -> tensor<4xf64>
    func.return %out : tensor<4xf64>
}

// CHECK-LABEL: func.func @remProbsPassthrough
// CHECK:       call @passthroughCircuit() : () -> tensor<4xf64>
// CHECK-NOT:   arith.constant dense<0.000000e+00>
// CHECK-NOT:   quantum.compbasis
// CHECK-NOT:   quantum.custom "PauliX"
