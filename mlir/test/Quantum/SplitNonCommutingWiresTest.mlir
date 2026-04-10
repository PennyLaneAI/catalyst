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

// RUN: quantum-opt %s --split-non-commuting="grouping-strategy=wires" --split-input-file --verify-diagnostics | FileCheck %s

// Test non-overlapping observables on separate wires
// Z(0), X(1), Y(2) are all on different wire, so they should form in a single group.

// CHECK-LABEL: func.func public @circ_no_overlap
// CHECK-SAME: () -> (f64, f64, f64) attributes {quantum.node}
// CHECK: %[[Q0:.*]] = quantum.extract %{{.*}}[ 0]
// CHECK: %[[Q1:.*]] = quantum.extract %{{.*}}[ 1]
// CHECK: %[[Q2:.*]] = quantum.extract %{{.*}}[ 2]
// CHECK: quantum.namedobs %[[Q0]][ PauliZ]
// CHECK: quantum.namedobs %[[Q1]][ PauliX]
// CHECK: quantum.namedobs %[[Q2]][ PauliY]
// CHECK: quantum.expval
// CHECK: quantum.expval
// CHECK: quantum.expval
// CHECK: return

module {
  func.func public @circ_no_overlap() -> (f64, f64, f64) attributes {quantum.node} {
    %shots = arith.constant 100 : i64
    quantum.device shots(%shots) ["", "", ""]
    %reg = quantum.alloc(3) : !quantum.reg
    %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %reg[2] : !quantum.reg -> !quantum.bit
    %obs_z0 = quantum.namedobs %q0[PauliZ] : !quantum.obs
    %obs_x1 = quantum.namedobs %q1[PauliX] : !quantum.obs
    %obs_y2 = quantum.namedobs %q2[PauliY] : !quantum.obs
    %expval_z0 = quantum.expval %obs_z0 : f64
    %expval_x1 = quantum.expval %obs_x1 : f64
    %expval_y2 = quantum.expval %obs_y2 : f64
    quantum.dealloc %reg : !quantum.reg
    quantum.device_release
    return %expval_z0, %expval_x1, %expval_y2 : f64, f64, f64
  }
}

// -----

// Test overlapping observables on the same wire
// Z(0) and X(1) are on the same wire -> group 1
// Y(1) overlaps with X(1) on wire 1 -> group 2

// CHECK-LABEL: func.func public @circ_overlap
// CHECK-SAME: () -> (f64, f64, f64)
// CHECK-NOT: quantum.node
// CHECK: %[[CALL0:.*]]:2 = call @circ_overlap.group.0
// CHECK: %[[CALL1:.*]] = call @circ_overlap.group.1
// CHECK: return %[[CALL0]]#0, %[[CALL0]]#1, %[[CALL1]]

// CHECK-LABEL: func.func private @circ_overlap.group.0
// CHECK-SAME: () -> (f64, f64) attributes {quantum.node}
// CHECK: %[[SHOTS0:.*]] = arith.constant 50
// CHECK: quantum.device shots(%[[SHOTS0]])
// CHECK: quantum.alloc
// CHECK: %[[Q0:.*]] = quantum.extract %{{.*}}[ 0]
// CHECK: %[[Q1:.*]] = quantum.extract %{{.*}}[ 1]
// CHECK: %[[OBS_Z0:.*]] = quantum.namedobs %[[Q0]][ PauliZ]
// CHECK: %[[OBS_X1:.*]] = quantum.namedobs %[[Q1]][ PauliX]
// CHECK: %[[EV0:.*]] = quantum.expval %[[OBS_Z0]]
// CHECK: %[[EV1:.*]] = quantum.expval %[[OBS_X1]]
// CHECK: quantum.dealloc
// CHECK: quantum.device_release
// CHECK: return %[[EV0]], %[[EV1]]

// CHECK-LABEL: func.func private @circ_overlap.group.1
// CHECK-SAME: () -> f64 attributes {quantum.node}
// CHECK: %[[SHOTS1:.*]] = arith.constant 50
// CHECK: quantum.device shots(%[[SHOTS1]])
// CHECK: quantum.alloc
// CHECK: %[[Q1:.*]] = quantum.extract %{{.*}}[ 1]
// CHECK: %[[OBS_Y1:.*]] = quantum.namedobs %[[Q1]][ PauliY]
// CHECK: %[[EV2:.*]] = quantum.expval %[[OBS_Y1]]
// CHECK: quantum.dealloc
// CHECK: quantum.device_release
// CHECK: return %[[EV2]]

module {
  func.func public @circ_overlap() -> (f64, f64, f64) attributes {quantum.node} {
    %shots = arith.constant 100 : i64
    quantum.device shots(%shots) ["", "", ""]
    %reg = quantum.alloc(2) : !quantum.reg
    %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[1] : !quantum.reg -> !quantum.bit
    %obs_z0 = quantum.namedobs %q0[PauliZ] : !quantum.obs
    %obs_x1 = quantum.namedobs %q1[PauliX] : !quantum.obs
    %obs_y1 = quantum.namedobs %q1[PauliY] : !quantum.obs
    %expval_z0 = quantum.expval %obs_z0 : f64
    %expval_x1 = quantum.expval %obs_x1 : f64
    %expval_y1 = quantum.expval %obs_y1 : f64
    quantum.dealloc %reg : !quantum.reg
    quantum.device_release
    return %expval_z0, %expval_x1, %expval_y1 : f64, f64, f64
  }
}

// -----

// Test tensor product of observables on overlapping wires
// Z(0)@Z(1) uses wires {0,1} -> group 1
// X(0) uses wire {0} -> group 2

// CHECK-LABEL: func.func public @circ_tensor_overlap
// CHECK-SAME: () -> (f64, f64)
// CHECK-NOT: quantum.node
// CHECK: %[[CALL0:.*]] = call @circ_tensor_overlap.group.0
// CHECK: %[[CALL1:.*]] = call @circ_tensor_overlap.group.1
// CHECK: return %[[CALL0]], %[[CALL1]]

// CHECK-LABEL: func.func private @circ_tensor_overlap.group.0
// CHECK-SAME: () -> f64 attributes {quantum.node}
// CHECK: %[[SHOTS0:.*]] = arith.constant 50
// CHECK: quantum.device shots(%[[SHOTS0]])
// CHECK: %[[Q0:.*]] = quantum.extract %{{.*}}[ 0]
// CHECK: %[[Q1:.*]] = quantum.extract %{{.*}}[ 1]
// CHECK: %[[OBS_Z0:.*]] = quantum.namedobs %[[Q0]][ PauliZ]
// CHECK: %[[OBS_Z1:.*]] = quantum.namedobs %[[Q1]][ PauliZ]
// CHECK: %[[TENSOR:.*]] = quantum.tensor %[[OBS_Z0]], %[[OBS_Z1]]
// CHECK: %[[EV0:.*]] = quantum.expval %[[TENSOR]]
// CHECK: return %[[EV0]]

// CHECK-LABEL: func.func private @circ_tensor_overlap.group.1
// CHECK-SAME: () -> f64 attributes {quantum.node}
// CHECK: %[[SHOTS1:.*]] = arith.constant 50
// CHECK: quantum.device shots(%[[SHOTS1]])
// CHECK: %[[Q0:.*]] = quantum.extract %{{.*}}[ 0]
// CHECK-NOT: quantum.tensor
// CHECK: %[[OBS_X0:.*]] = quantum.namedobs %[[Q0]][ PauliX]
// CHECK: %[[EV1:.*]] = quantum.expval %[[OBS_X0]]
// CHECK: return %[[EV1]]

module {
  func.func public @circ_tensor_overlap() -> (f64, f64) attributes {quantum.node} {
    %shots = arith.constant 100 : i64
    quantum.device shots(%shots) ["", "", ""]
    %reg = quantum.alloc(2) : !quantum.reg
    %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[1] : !quantum.reg -> !quantum.bit
    %obs_z0 = quantum.namedobs %q0[PauliZ] : !quantum.obs
    %obs_z1 = quantum.namedobs %q1[PauliZ] : !quantum.obs
    %tensor_obs = quantum.tensor %obs_z0, %obs_z1 : !quantum.obs
    %expval_tensor = quantum.expval %tensor_obs : f64
    %obs_x0 = quantum.namedobs %q0[PauliX] : !quantum.obs
    %expval_x0 = quantum.expval %obs_x0 : f64
    quantum.dealloc %reg : !quantum.reg
    quantum.device_release
    return %expval_tensor, %expval_x0 : f64, f64
  }
}

// -----

// Test duplicate observables
// Z(0) appears twice, X(1) once -> reused Z(0) result
// Both on different wires, so they form in a single group.

// CHECK-LABEL: func.func public @circ_dup
// CHECK-SAME: () -> (f64, f64, f64)
// CHECK-NOT: quantum.node
// CHECK: %[[CALL0:.*]]:2 = call @circ_dup.group.0
// CHECK: return %[[CALL0]]#0, %[[CALL0]]#1, %[[CALL0]]#0

// CHECK-LABEL: func.func private @circ_dup.group.0
// CHECK-SAME: () -> (f64, f64) attributes {quantum.node}
// CHECK: quantum.alloc
// CHECK: %[[Q0:.*]] = quantum.extract %{{.*}}[ 0]
// CHECK: %[[Q1:.*]] = quantum.extract %{{.*}}[ 1]
// CHECK: %[[OBS_Z0:.*]] = quantum.namedobs %[[Q0]][ PauliZ]
// CHECK: %[[OBS_X1:.*]] = quantum.namedobs %[[Q1]][ PauliX]
// CHECK: %[[EV0:.*]] = quantum.expval %[[OBS_Z0]]
// CHECK: %[[EV1:.*]] = quantum.expval %[[OBS_X1]]
// CHECK-NOT: quantum.expval
// CHECK: quantum.dealloc
// CHECK: quantum.device_release
// CHECK: return %[[EV0]], %[[EV1]]

module {
  func.func public @circ_dup() -> (f64, f64, f64) attributes {quantum.node} {
    %shots = arith.constant 100 : i64
    quantum.device shots(%shots) ["", "", ""]
    %reg = quantum.alloc(2) : !quantum.reg
    %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[1] : !quantum.reg -> !quantum.bit
    %obs_z0a = quantum.namedobs %q0[PauliZ] : !quantum.obs
    %obs_x1 = quantum.namedobs %q1[PauliX] : !quantum.obs
    %obs_z0b = quantum.namedobs %q0[PauliZ] : !quantum.obs
    %expval_z0a = quantum.expval %obs_z0a : f64
    %expval_x1 = quantum.expval %obs_x1 : f64
    %expval_z0b = quantum.expval %obs_z0b : f64
    quantum.dealloc %reg : !quantum.reg
    quantum.device_release
    return %expval_z0a, %expval_x1, %expval_z0b : f64, f64, f64
  }
}

// -----

// Test multiple measurements with overlapping observables
// Y(0) and X(1) on different wires -> group 1
// X(0)@X(1) uses wires {0,1} -> group 2

// CHECK-LABEL: func.func public @circ_multi_mp
// CHECK-SAME: () -> (f64, f64, f64)
// CHECK-NOT: quantum.node
// CHECK: %[[CALL0:.*]]:2 = call @circ_multi_mp.group.0
// CHECK: %[[CALL1:.*]] = call @circ_multi_mp.group.1
// CHECK: return %[[CALL0]]#0, %[[CALL0]]#1, %[[CALL1]]

// CHECK-LABEL: func.func private @circ_multi_mp.group.0
// CHECK-SAME: () -> (f64, f64) attributes {quantum.node}
// CHECK: %[[SHOTS0:.*]] = arith.constant 50
// CHECK: quantum.device shots(%[[SHOTS0]])
// CHECK: quantum.alloc
// CHECK: %[[Q0:.*]] = quantum.extract %{{.*}}[ 0]
// CHECK: %[[Q1:.*]] = quantum.extract %{{.*}}[ 1]
// CHECK: %[[OBS_Y0:.*]] = quantum.namedobs %[[Q0]][ PauliY]
// CHECK: %[[OBS_X1:.*]] = quantum.namedobs %[[Q1]][ PauliX]
// CHECK: %[[EV0:.*]] = quantum.expval %[[OBS_Y0]]
// CHECK: %[[EV1:.*]] = quantum.expval %[[OBS_X1]]
// CHECK-NOT: quantum.tensor
// CHECK: return %[[EV0]], %[[EV1]]

// CHECK-LABEL: func.func private @circ_multi_mp.group.1
// CHECK-SAME: () -> f64 attributes {quantum.node}
// CHECK: %[[SHOTS1:.*]] = arith.constant 50
// CHECK: quantum.device shots(%[[SHOTS1]])
// CHECK: quantum.alloc
// CHECK: %[[Q0:.*]] = quantum.extract %{{.*}}[ 0]
// CHECK: %[[Q1:.*]] = quantum.extract %{{.*}}[ 1]
// CHECK: quantum.namedobs %[[Q0]][ PauliX]
// CHECK: quantum.namedobs %[[Q1]][ PauliX]
// CHECK: %[[TENSOR:.*]] = quantum.tensor
// CHECK: %[[EV2:.*]] = quantum.expval %[[TENSOR]]
// CHECK: return %[[EV2]]

module {
  func.func public @circ_multi_mp() -> (f64, f64, f64) attributes {quantum.node} {
    %shots = arith.constant 100 : i64
    quantum.device shots(%shots) ["", "", ""]
    %reg = quantum.alloc(2) : !quantum.reg
    %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[1] : !quantum.reg -> !quantum.bit
    %obs_y0 = quantum.namedobs %q0[PauliY] : !quantum.obs
    %obs_x1 = quantum.namedobs %q1[PauliX] : !quantum.obs
    %obs_x0 = quantum.namedobs %q0[PauliX] : !quantum.obs
    %obs_x1b = quantum.namedobs %q1[PauliX] : !quantum.obs
    %tensor_x0x1 = quantum.tensor %obs_x0, %obs_x1b : !quantum.obs
    %expval_y0 = quantum.expval %obs_y0 : f64
    %expval_x1 = quantum.expval %obs_x1 : f64
    %expval_tensor = quantum.expval %tensor_x0x1 : f64
    quantum.dealloc %reg : !quantum.reg
    quantum.device_release
    return %expval_y0, %expval_x1, %expval_tensor : f64, f64, f64
  }
}

// -----

// Test Hamiltonian with non-overlapping terms
// expval(Z(0) + X(1) + 2*Y(2)) -> terms on wires {0}, {1}, {2} -> all same group
// The .single_terms function should remain unsplit

// CHECK-LABEL: func.func public @circ_ham.single_terms
// CHECK-SAME: attributes {quantum.node}
// CHECK: %[[Q0:.*]] = quantum.extract %{{.*}}[ 0]
// CHECK: %[[Q1:.*]] = quantum.extract %{{.*}}[ 1]
// CHECK: %[[Q2:.*]] = quantum.extract %{{.*}}[ 2]
// CHECK: quantum.namedobs %[[Q0]][ PauliZ]
// CHECK: quantum.namedobs %[[Q1]][ PauliX]
// CHECK: quantum.namedobs %[[Q2]][ PauliY]
// CHECK: quantum.expval
// CHECK: quantum.expval
// CHECK: quantum.expval
// CHECK: return

// CHECK-LABEL: func.func public @circ_ham
// CHECK-SAME: () -> tensor<f64>
// CHECK-NOT: quantum.hamiltonian
// CHECK: call @circ_ham.single_terms
// CHECK: stablehlo.multiply
// CHECK: stablehlo.reduce

module {
  func.func public @circ_ham() -> tensor<f64> attributes {quantum.node} {
    %shots = arith.constant 100 : i64
    quantum.device shots(%shots) ["", "", ""]
    %reg = quantum.alloc(3) : !quantum.reg
    %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %reg[2] : !quantum.reg -> !quantum.bit
    %obs_z0 = quantum.namedobs %q0[PauliZ] : !quantum.obs
    %obs_x1 = quantum.namedobs %q1[PauliX] : !quantum.obs
    %obs_y2 = quantum.namedobs %q2[PauliY] : !quantum.obs
    %coeffs = stablehlo.constant dense<[1.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf64>
    %ham = quantum.hamiltonian(%coeffs : tensor<3xf64>) %obs_z0, %obs_x1, %obs_y2 : !quantum.obs
    %expval_h = quantum.expval %ham : f64
    %result = tensor.from_elements %expval_h : tensor<f64>
    quantum.dealloc %reg : !quantum.reg
    quantum.device_release
    return %result : tensor<f64>
  }
}

// -----

// Test Hamiltonian with overlapping terms
// expval(0.5*Z(0) + 3*X(1) + Y(1)) -> Z(0) on wire 0, X(1) on wire 1, Y(1) on wire 1
// Wire grouping: {Z(0), X(1)} and {Y(1)} -> 2 groups

// CHECK-LABEL: func.func public @circ_ham_overlap.single_terms
// CHECK-SAME: () -> (tensor<f64>, tensor<f64>, tensor<f64>)
// CHECK-NOT: quantum.node
// CHECK: %[[CALL0:.*]]:2 = call @circ_ham_overlap.single_terms.group.0
// CHECK: %[[CALL1:.*]] = call @circ_ham_overlap.single_terms.group.1
// CHECK: return %[[CALL0]]#0, %[[CALL0]]#1, %[[CALL1]]

// CHECK-LABEL: func.func public @circ_ham_overlap
// CHECK-SAME: () -> tensor<f64>
// CHECK: call @circ_ham_overlap.single_terms
// CHECK: stablehlo.multiply
// CHECK: stablehlo.reduce

// CHECK-LABEL: func.func private @circ_ham_overlap.single_terms.group.0
// CHECK-SAME: () -> (tensor<f64>, tensor<f64>) attributes {quantum.node}
// CHECK: %[[SHOTS0:.*]] = arith.constant 50
// CHECK: quantum.device shots(%[[SHOTS0]])
// CHECK: %[[Q0:.*]] = quantum.extract %{{.*}}[ 0]
// CHECK: %[[Q1:.*]] = quantum.extract %{{.*}}[ 1]
// CHECK: quantum.namedobs %[[Q0]][ PauliZ]
// CHECK: quantum.namedobs %[[Q1]][ PauliX]
// CHECK: quantum.expval
// CHECK: quantum.expval
// CHECK: return

// CHECK-LABEL: func.func private @circ_ham_overlap.single_terms.group.1
// CHECK-SAME: () -> tensor<f64> attributes {quantum.node}
// CHECK: %[[SHOTS1:.*]] = arith.constant 50
// CHECK: quantum.device shots(%[[SHOTS1]])
// CHECK: %[[Q1:.*]] = quantum.extract %{{.*}}[ 1]
// CHECK: quantum.namedobs %[[Q1]][ PauliY]
// CHECK: quantum.expval
// CHECK: return

module {
  func.func public @circ_ham_overlap() -> tensor<f64> attributes {quantum.node} {
    %shots = arith.constant 100 : i64
    quantum.device shots(%shots) ["", "", ""]
    %reg = quantum.alloc(2) : !quantum.reg
    %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[1] : !quantum.reg -> !quantum.bit
    %obs_z0 = quantum.namedobs %q0[PauliZ] : !quantum.obs
    %obs_x1 = quantum.namedobs %q1[PauliX] : !quantum.obs
    %obs_y1 = quantum.namedobs %q1[PauliY] : !quantum.obs
    %coeffs = stablehlo.constant dense<[5.000000e-01, 3.000000e+00, 1.000000e+00]> : tensor<3xf64>
    %ham = quantum.hamiltonian(%coeffs : tensor<3xf64>) %obs_z0, %obs_x1, %obs_y1 : !quantum.obs
    %expval_h = quantum.expval %ham : f64
    %result = tensor.from_elements %expval_h : tensor<f64>
    quantum.dealloc %reg : !quantum.reg
    quantum.device_release
    return %result : tensor<f64>
  }
}
