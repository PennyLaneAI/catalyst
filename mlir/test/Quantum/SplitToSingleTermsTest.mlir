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


// RUN: quantum-opt %s --split-to-single-terms --split-input-file --verify-diagnostics | FileCheck %s

// Test Split Hamiltonian expval into individual leaf expvals
//
// Input circuit has:
//   - Gates: RY(0.4) on qubit 0, RX(0.6) on qubit 1, RZ(0.8) on qubit 2
//   - Hamiltonian H = 2.0 * (Z(0) @ X(1)) + 3.0 * Y(2)
//   - expval(H) returns weighted sum
//   - expval(Z) on qubit 1 (non-Hamiltonian)
//
// After transformation:
//   - circ.quantum: returns individual expvals [<Z x X>, <Y>, <Z>]
//   - circ: calls circ.quantum, computes weighted sum, returns [<H>, <Z>]

// CHECK-LABEL: func.func public @circ.quantum
// CHECK-SAME: () -> (tensor<f64>, tensor<f64>, tensor<f64>) attributes {quantum.node}
// CHECK: quantum.device
// CHECK: quantum.alloc
// CHECK: quantum.custom "RY"
// CHECK: quantum.custom "RX"
// CHECK: quantum.custom "RZ"
// CHECK: %[[EXPVAL_ZX:.*]] = quantum.expval %[[OBS_ZX:.*]]
// CHECK: %[[EXPVAL_Y:.*]] = quantum.expval %[[OBS_Y:.*]]
// CHECK: %[[EXPVAL_Z:.*]] = quantum.expval %[[OBS_Z2:.*]]
// CHECK: quantum.dealloc
// CHECK: quantum.device_release
// CHECK: return %[[TENSOR_ZX:.*]], %[[TENSOR_Y:.*]], %[[TENSOR_Z:.*]]

// CHECK-LABEL: func.func public @circ
// CHECK-SAME: (%arg0: tensor<1xf64>, %arg1: tensor<1xf64>) -> (tensor<f64>, tensor<f64>)
// CHECK-NOT: attributes {quantum.node}
// CHECK-NOT: quantum.hamiltonian
// CHECK-NOT: quantum.custom "RY"
// CHECK-NOT: quantum.custom "RX"
// CHECK-NOT: quantum.custom "RZ"
// CHECK-NOT: quantum.expval
// CHECK-NOT: quantum.namedobs
// CHECK-NOT: quantum.tensor
// CHECK: %[[CALL:.*]]:3 = call @circ.quantum
// CHECK: %[[CONCAT:.*]] = stablehlo.concatenate
// CHECK: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00>
// CHECK: %[[RESULT:.*]] = stablehlo.reduce(%[[CONCAT]] init: %[[ZERO]]) applies stablehlo.add
// CHECK: return %[[RESULT]], %[[CALL]]#2

module {
  func.func public @circ(%arg0: tensor<1xf64>, %arg1: tensor<1xf64>) -> (tensor<f64>, tensor<f64>) attributes {quantum.node} {
    %shots = arith.constant 0 : i64
    quantum.device shots(%shots) ["/path/to/lightning.dylib", "LightningSimulator", "{}"]
    %reg = quantum.alloc(3) : !quantum.reg
    %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %ry_angle = arith.constant 4.000000e-01 : f64
    %q0_out = quantum.custom "RY"(%ry_angle) %q0 : !quantum.bit
    %q1 = quantum.extract %reg[1] : !quantum.reg -> !quantum.bit
    %rx_angle = arith.constant 6.000000e-01 : f64
    %q1_out = quantum.custom "RX"(%rx_angle) %q1 : !quantum.bit
    %q2 = quantum.extract %reg[2] : !quantum.reg -> !quantum.bit
    %rz_angle = arith.constant 8.000000e-01 : f64
    %q2_out = quantum.custom "RZ"(%rz_angle) %q2 : !quantum.bit
    %obs_z0 = quantum.namedobs %q0_out[PauliZ] : !quantum.obs
    %obs_x1 = quantum.namedobs %q1_out[PauliX] : !quantum.obs
    %obs_zx = quantum.tensor %obs_z0, %obs_x1 : !quantum.obs
    %ham_zx = quantum.hamiltonian(%arg0 : tensor<1xf64>) %obs_zx : !quantum.obs
    %obs_y2 = quantum.namedobs %q2_out[PauliY] : !quantum.obs
    %ham_y = quantum.hamiltonian(%arg1 : tensor<1xf64>) %obs_y2 : !quantum.obs
    %coeff_broadcast = stablehlo.constant dense<[1.000000e+00, 1.000000e+00]> : tensor<2xf64>
    %ham_total = quantum.hamiltonian(%coeff_broadcast : tensor<2xf64>) %ham_zx, %ham_y : !quantum.obs
    %expval_h = quantum.expval %ham_total : f64
    %result_h = tensor.from_elements %expval_h : tensor<f64>
    %obs_z1 = quantum.namedobs %q1_out[PauliZ] : !quantum.obs
    %expval_z = quantum.expval %obs_z1 : f64
    %result_z = tensor.from_elements %expval_z : tensor<f64>
    quantum.dealloc %reg : !quantum.reg
    quantum.device_release
    return %result_h, %result_z : tensor<f64>, tensor<f64>
  }
}

// -----

// Test Split Hamiltonian expval with Identity observable
//
// Input circuit has:
//   - Gates: RY(0.5) on qubit 0, RX(0.3) on qubit 1
//   - Hamiltonian H = 1.0 * Z(0) + 2.0 * X(1) + 0.7 * Identity(2)
//
// After transformation:
//   - circ.quantum: returns individual expvals [<Z(0)>, <X(1)>, 1.0]
//   - circ: calls circ.quantum, computes weighted sum, returns <H>

// CHECK-LABEL: func.func public @circ.quantum
// CHECK-SAME: () -> (tensor<f64>, tensor<f64>, tensor<f64>) attributes {quantum.node}
// CHECK: quantum.device
// CHECK: quantum.alloc
// CHECK: quantum.custom "RY"
// CHECK: quantum.custom "RX"
// CHECK: %[[EXPVAL_Z:.*]] = quantum.expval %[[OBS_Z:.*]]
// CHECK: %[[EXPVAL_X:.*]] = quantum.expval %[[OBS_X:.*]]
// CHECK-NOT: quantum.namedobs.*Identity
// CHECK-NOT: quantum.expval.*Identity
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f64
// CHECK: %[[TENSOR_IDENTITY:.*]] = tensor.from_elements %[[ONE]]
// CHECK: quantum.dealloc
// CHECK: quantum.device_release
// CHECK: return %[[TENSOR_Z:.*]], %[[TENSOR_X:.*]], %[[TENSOR_IDENTITY]]

// CHECK-LABEL: func.func public @circ
// CHECK-SAME: (%arg0: tensor<1xf64>, %arg1: tensor<1xf64>) -> tensor<f64>
// CHECK-NOT: attributes {quantum.node}
// CHECK-NOT: quantum.hamiltonian
// CHECK-NOT: quantum.custom "RY"
// CHECK-NOT: quantum.custom "RX"
// CHECK-NOT: quantum.expval
// CHECK-NOT: quantum.namedobs
// CHECK: %[[CALL:.*]]:3 = call @circ.quantum()
// CHECK: %[[W0:.*]] = stablehlo.multiply %[[COEFF0:.*]], %[[CALL]]#0
// CHECK: %[[W1:.*]] = stablehlo.multiply %[[COEFF1:.*]], %[[CALL]]#1
// CHECK: %[[W2:.*]] = stablehlo.multiply %[[COEFF2:.*]], %[[CALL]]#2
// CHECK: %[[CONCAT:.*]] = stablehlo.concatenate
// CHECK: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00>
// CHECK: %[[RESULT:.*]] = stablehlo.reduce(%[[CONCAT]] init: %[[ZERO]]) applies stablehlo.add
// CHECK: return %[[RESULT]]
module {
  func.func public @circ(%arg0: tensor<1xf64>, %arg1: tensor<1xf64>) -> tensor<f64> attributes {quantum.node} {
    %shots = arith.constant 0 : i64
    quantum.device shots(%shots) ["/path/to/lightning.dylib", "LightningSimulator", "{}"]
    %reg = quantum.alloc(3) : !quantum.reg
    %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %ry_angle = arith.constant 5.000000e-01 : f64
    %q0_out = quantum.custom "RY"(%ry_angle) %q0 : !quantum.bit
    %q1 = quantum.extract %reg[1] : !quantum.reg -> !quantum.bit
    %rx_angle = arith.constant 3.000000e-01 : f64
    %q1_out = quantum.custom "RX"(%rx_angle) %q1 : !quantum.bit
    %obs_z0 = quantum.namedobs %q0_out[PauliZ] : !quantum.obs
    %obs_x1 = quantum.namedobs %q1_out[PauliX] : !quantum.obs
    %ham_x = quantum.hamiltonian(%arg0 : tensor<1xf64>) %obs_x1 : !quantum.obs
    %q2 = quantum.extract %reg[2] : !quantum.reg -> !quantum.bit
    %obs_id = quantum.namedobs %q2[Identity] : !quantum.obs
    %ham_id = quantum.hamiltonian(%arg1 : tensor<1xf64>) %obs_id : !quantum.obs
    %coeff_broadcast = stablehlo.constant dense<[1.000000e+00, 1.000000e+00, 1.000000e+00]> : tensor<3xf64>
    %ham_total = quantum.hamiltonian(%coeff_broadcast : tensor<3xf64>) %obs_z0, %ham_x, %ham_id : !quantum.obs
    %expval_h = quantum.expval %ham_total : f64
    %result = tensor.from_elements %expval_h : tensor<f64>
    quantum.dealloc %reg : !quantum.reg
    quantum.device_release
    return %result : tensor<f64>
  }
}
