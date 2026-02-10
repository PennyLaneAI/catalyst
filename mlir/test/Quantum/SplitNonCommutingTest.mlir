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


// RUN: quantum-opt %s --split-non-commuting --split-input-file --verify-diagnostics | FileCheck %s

// Test Split expvals to multiple executions

// CHECK-LABEL: func.func public @circ
// CHECK-SAME: () -> (f64, f64, f64)
// CHECK: %[[CALL0:.*]] = call @circ.group.0
// CHECK: %[[CALL1:.*]] = call @circ.group.1
// CHECK: %[[CALL2:.*]] = call @circ.group.2
// CHECK: return %[[CALL0]], %[[CALL1]], %[[CALL2]]

// CHECK-LABEL: func.func private @circ.group.0
// CHECK-SAME: () -> f64 attributes {qnode}
// CHECK: %[[SHOTS:.*]] = arith.constant 100
// CHECK: quantum.device shots(%[[SHOTS]])
// CHECK: quantum.alloc
// CHECK: %[[OBS_Z0:.*]] = quantum.namedobs %[[Q0:.*]][ PauliZ]
// CHECK: %[[EXPVAL_Z0:.*]] = quantum.expval %[[OBS_Z0]]
// CHECK: quantum.dealloc
// CHECK: quantum.device_release
// CHECK: return %[[EXPVAL_Z0]]

// CHECK-LABEL: func.func private @circ.group.1
// CHECK-SAME: () -> f64 attributes {qnode}
// CHECK: %[[SHOTS:.*]] = arith.constant 100
// CHECK: quantum.device shots(%[[SHOTS]])
// CHECK: quantum.alloc
// CHECK: %[[OBS_X1:.*]] = quantum.namedobs %[[Q1:.*]][ PauliX]
// CHECK: %[[EXPVAL_X1:.*]] = quantum.expval %[[OBS_X1]]
// CHECK: quantum.dealloc
// CHECK: quantum.device_release
// CHECK: return %[[EXPVAL_X1]]

// CHECK-LABEL: func.func private @circ.group.2
// CHECK-SAME: () -> f64 attributes {qnode}
// CHECK: %[[SHOTS:.*]] = arith.constant 100
// CHECK: quantum.device shots(%[[SHOTS]])
// CHECK: quantum.alloc
// CHECK: %[[OBS_Y2:.*]] = quantum.namedobs %[[Q2:.*]][ PauliY]
// CHECK: %[[EXPVAL_Y2:.*]] = quantum.expval %[[OBS_Y2]]
// CHECK: quantum.dealloc
// CHECK: quantum.device_release
// CHECK: return %[[EXPVAL_Y2]]

module {
  func.func public @circ() -> (f64, f64, f64) attributes {qnode} {
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

// CHECK-LABEL: func.func public @circ
// CHECK-SAME: () -> (f64, f64, f64, f64)
// CHECK: %[[CALL0:.*]]:2 = call @circ.group.0
// CHECK: %[[CALL1:.*]] = call @circ.group.1
// CHECK: %[[CALL2:.*]] = call @circ.group.2
// CHECK: return %[[CALL0]]#0, %[[CALL1]], %[[CALL2]], %[[CALL0]]#1

// CHECK-LABEL: func.func private @circ.group.0
// CHECK-SAME: () -> (f64, f64) attributes {qnode}
// CHECK: %[[SHOTS:.*]] = arith.constant 100
// CHECK: quantum.device shots(%[[SHOTS]])
// CHECK: quantum.alloc
// CHECK: %[[OBS_Z0:.*]] = quantum.namedobs %[[Q0:.*]][ PauliZ]
// CHECK: %[[EXPVAL_Z0:.*]] = quantum.expval %[[OBS_Z0]]
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00
// CHECK: quantum.dealloc
// CHECK: quantum.device_release
// CHECK: return %[[EXPVAL_Z0]], %[[ONE]]

// CHECK-LABEL: func.func private @circ.group.1
// CHECK-SAME: () -> f64 attributes {qnode}
// CHECK: %[[SHOTS:.*]] = arith.constant 100
// CHECK: quantum.device shots(%[[SHOTS]])
// CHECK: quantum.alloc
// CHECK: %[[OBS_X1:.*]] = quantum.namedobs %[[Q1:.*]][ PauliX]
// CHECK: %[[EXPVAL_X1:.*]] = quantum.expval %[[OBS_X1]]
// CHECK: quantum.dealloc
// CHECK: quantum.device_release
// CHECK: return %[[EXPVAL_X1]]

// CHECK-LABEL: func.func private @circ.group.2
// CHECK-SAME: () -> f64 attributes {qnode}
// CHECK: %[[SHOTS:.*]] = arith.constant 100
// CHECK: quantum.device shots(%[[SHOTS]])
// CHECK: quantum.alloc
// CHECK: %[[OBS_Y2:.*]] = quantum.namedobs %[[Q2:.*]][ PauliY]
// CHECK: %[[EXPVAL_Y2:.*]] = quantum.expval %[[OBS_Y2]]
// CHECK: quantum.dealloc
// CHECK: quantum.device_release
// CHECK: return %[[EXPVAL_Y2]]
module {
  func.func public @circ() -> (f64, f64, f64, f64) attributes {qnode} {
    %shots = arith.constant 100 : i64
    quantum.device shots(%shots) ["", "", ""]
    %reg = quantum.alloc(3) : !quantum.reg
    %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %reg[2] : !quantum.reg -> !quantum.bit
    %obs_z0 = quantum.namedobs %q0[PauliZ] : !quantum.obs
    %obs_x1 = quantum.namedobs %q1[PauliX] : !quantum.obs
    %obs_y2 = quantum.namedobs %q2[PauliY] : !quantum.obs
    %identity = quantum.namedobs %q2[Identity] : !quantum.obs
    %expval_z0 = quantum.expval %obs_z0 : f64
    %expval_x1 = quantum.expval %obs_x1 : f64
    %expval_y2 = quantum.expval %obs_y2 : f64
    %expval_identity = quantum.expval %identity : f64
    quantum.dealloc %reg : !quantum.reg
    quantum.device_release
    return %expval_z0, %expval_x1, %expval_y2, %expval_identity : f64, f64, f64, f64
  }
}