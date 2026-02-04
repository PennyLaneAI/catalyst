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


module @circ {
  func.func public @jit_circ() -> (tensor<f64>, tensor<f64>) attributes {llvm.emit_c_interface} {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<1xf64>
    %c = stablehlo.constant dense<3> : tensor<1xi64>
    %0:2 = catalyst.launch_kernel @module_circ::@circ(%cst, %c) : (tensor<1xf64>, tensor<1xi64>) -> (tensor<f64>, tensor<f64>)
    return %0#0, %0#1 : tensor<f64>, tensor<f64>
  }
  module @module_circ {
    module attributes {transform.with_named_sequence} {
      transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
        transform.yield
      }
    }

    // CHECK-LABEL: func.func public @circ.quantum
    // CHECK-SAME: () -> (tensor<f64>, tensor<f64>, tensor<f64>)
    // CHECK: quantum.device
    // CHECK: quantum.alloc
    // CHECK: quantum.custom "RY"
    // CHECK: quantum.custom "RX"
    // CHECK: quantum.custom "RZ"
    // CHECK: %[[OBS_Z:.*]] = quantum.namedobs %{{.*}}[ PauliZ]
    // CHECK: %[[OBS_X:.*]] = quantum.namedobs %{{.*}}[ PauliX]
    // CHECK: %[[OBS_ZX:.*]] = quantum.tensor %[[OBS_Z]], %[[OBS_X]]
    // CHECK: %[[OBS_Y:.*]] = quantum.namedobs %{{.*}}[ PauliY]
    // CHECK: %[[EXPVAL_ZX:.*]] = quantum.expval %[[OBS_ZX]]
    // CHECK: %[[TENSOR_ZX:.*]] = tensor.from_elements %[[EXPVAL_ZX]]
    // CHECK: %[[EXPVAL_Y:.*]] = quantum.expval %[[OBS_Y]]
    // CHECK: %[[TENSOR_Y:.*]] = tensor.from_elements %[[EXPVAL_Y]]
    // CHECK: %[[OBS_Z2:.*]] = quantum.namedobs %{{.*}}[ PauliZ]
    // CHECK: %[[EXPVAL_Z:.*]] = quantum.expval %[[OBS_Z2]]
    // CHECK: %[[TENSOR_Z:.*]] = tensor.from_elements %[[EXPVAL_Z]]
    // CHECK: quantum.dealloc
    // CHECK: quantum.device_release
    // CHECK: return %[[TENSOR_ZX]], %[[TENSOR_Y]], %[[TENSOR_Z]]

    // CHECK-LABEL: func.func public @circ
    // CHECK-SAME: (%arg0: tensor<1xf64>, %arg1: tensor<1xi64>) -> (tensor<f64>, tensor<f64>)
    // CHECK-NOT: quantum.hamiltonian
    // CHECK-NOT: quantum.custom "RY"
    // CHECK-NOT: quantum.custom "RX"
    // CHECK-NOT: quantum.custom "RZ"
    // CHECK-NOT: quantum.expval
    // CHECK-NOT: quantum.namedobs
    // CHECK-NOT: quantum.tensor
    // CHECK: %[[CONVERT:.*]] = stablehlo.convert %arg1
    // CHECK: %[[CST:.*]] = stablehlo.constant dense<1.000000e+00>
    // CHECK: %[[BROADCAST:.*]] = stablehlo.broadcast_in_dim %[[CST]]
    // CHECK: %[[CALL:.*]]:3 = call @circ.quantum()
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[EXTRACTED:.*]] = tensor.extract %[[BROADCAST]][%[[C0]]]
    // CHECK: %[[FROM_ELEMS0:.*]] = tensor.from_elements %[[EXTRACTED]]
    // CHECK: %[[C0_1:.*]] = arith.constant 0 : index
    // CHECK: %[[EXTRACTED_1:.*]] = tensor.extract %arg0[%[[C0_1]]]
    // CHECK: %[[FROM_ELEMS1:.*]] = tensor.from_elements %[[EXTRACTED_1]]
    // CHECK: %[[MULT0:.*]] = stablehlo.multiply %[[FROM_ELEMS0]], %[[FROM_ELEMS1]]
    // CHECK: %[[C1:.*]] = arith.constant 1 : index
    // CHECK: %[[EXTRACTED_2:.*]] = tensor.extract %[[BROADCAST]][%[[C1]]]
    // CHECK: %[[FROM_ELEMS2:.*]] = tensor.from_elements %[[EXTRACTED_2]]
    // CHECK: %[[C0_2:.*]] = arith.constant 0 : index
    // CHECK: %[[EXTRACTED_3:.*]] = tensor.extract %[[CONVERT]][%[[C0_2]]]
    // CHECK: %[[FROM_ELEMS3:.*]] = tensor.from_elements %[[EXTRACTED_3]]
    // CHECK: %[[MULT1:.*]] = stablehlo.multiply %[[FROM_ELEMS2]], %[[FROM_ELEMS3]]
    // CHECK: %[[W0:.*]] = stablehlo.multiply %[[MULT0]], %[[CALL]]#0
    // CHECK: %[[W1:.*]] = stablehlo.multiply %[[MULT1]], %[[CALL]]#1
    // CHECK: stablehlo.broadcast_in_dim
    // CHECK: stablehlo.broadcast_in_dim
    // CHECK: %[[CONCAT:.*]] = stablehlo.concatenate
    // CHECK: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00>
    // CHECK: %[[RESULT:.*]] = stablehlo.reduce(%[[CONCAT]] init: %[[ZERO]]) applies stablehlo.add
    // CHECK: return %[[RESULT]], %[[CALL]]#2

    func.func public @circ(%arg0: tensor<1xf64>, %arg1: tensor<1xi64>) -> (tensor<f64>, tensor<f64>) attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
      %c = stablehlo.constant dense<0> : tensor<i64>
      %extracted = tensor.extract %c[] : tensor<i64>
      quantum.device shots(%extracted) ["/path/to/lightning.dylib", "LightningSimulator", "{}"]
      %c_0 = stablehlo.constant dense<3> : tensor<i64>
      %0 = quantum.alloc( 3) : !quantum.reg
      %extracted_1 = tensor.extract %c[] : tensor<i64>
      %1 = quantum.extract %0[%extracted_1] : !quantum.reg -> !quantum.bit
      %cst = stablehlo.constant dense<4.000000e-01> : tensor<f64>
      %extracted_2 = tensor.extract %cst[] : tensor<f64>
      %out_qubits = quantum.custom "RY"(%extracted_2) %1 : !quantum.bit
      %c_3 = stablehlo.constant dense<1> : tensor<i64>
      %extracted_4 = tensor.extract %c_3[] : tensor<i64>
      %2 = quantum.extract %0[%extracted_4] : !quantum.reg -> !quantum.bit
      %cst_5 = stablehlo.constant dense<6.000000e-01> : tensor<f64>
      %extracted_6 = tensor.extract %cst_5[] : tensor<f64>
      %out_qubits_7 = quantum.custom "RX"(%extracted_6) %2 : !quantum.bit
      %c_8 = stablehlo.constant dense<2> : tensor<i64>
      %extracted_9 = tensor.extract %c_8[] : tensor<i64>
      %3 = quantum.extract %0[%extracted_9] : !quantum.reg -> !quantum.bit
      %cst_10 = stablehlo.constant dense<8.000000e-01> : tensor<f64>
      %extracted_11 = tensor.extract %cst_10[] : tensor<f64>
      %out_qubits_12 = quantum.custom "RZ"(%extracted_11) %3 : !quantum.bit
      %4 = quantum.namedobs %out_qubits[ PauliZ] : !quantum.obs
      %5 = quantum.namedobs %out_qubits_7[ PauliX] : !quantum.obs
      %6 = quantum.tensor %4, %5 : !quantum.obs
      %7 = quantum.hamiltonian(%arg0 : tensor<1xf64>) %6 : !quantum.obs
      %8 = quantum.namedobs %out_qubits_12[ PauliY] : !quantum.obs
      %9 = stablehlo.convert %arg1 : (tensor<1xi64>) -> tensor<1xf64>
      %10 = quantum.hamiltonian(%9 : tensor<1xf64>) %8 : !quantum.obs
      %cst_13 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
      %11 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f64>) -> tensor<2xf64>
      %12 = quantum.hamiltonian(%11 : tensor<2xf64>) %7, %10 : !quantum.obs
      %13 = quantum.expval %12 : f64
      %from_elements = tensor.from_elements %13 : tensor<f64>
      %14 = quantum.namedobs %out_qubits_7[ PauliZ] : !quantum.obs
      %15 = quantum.expval %14 : f64
      %from_elements_14 = tensor.from_elements %15 : tensor<f64>
      %extracted_15 = tensor.extract %c[] : tensor<i64>
      %16 = quantum.insert %0[%extracted_15], %out_qubits : !quantum.reg, !quantum.bit
      %extracted_16 = tensor.extract %c_3[] : tensor<i64>
      %17 = quantum.insert %16[%extracted_16], %out_qubits_7 : !quantum.reg, !quantum.bit
      %extracted_17 = tensor.extract %c_8[] : tensor<i64>
      %18 = quantum.insert %17[%extracted_17], %out_qubits_12 : !quantum.reg, !quantum.bit
      quantum.dealloc %18 : !quantum.reg
      quantum.device_release
      return %from_elements, %from_elements_14 : tensor<f64>, tensor<f64>
    }
  }
  func.func @setup() {
    quantum.init
    return
  }
  func.func @teardown() {
    quantum.finalize
    return
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

module @circ {
  func.func public @jit_circ() -> tensor<f64> attributes {llvm.emit_c_interface} {
    %c = stablehlo.constant dense<2> : tensor<1xi64>
    %cst = stablehlo.constant dense<0.69999999999999996> : tensor<1xf64>
    %0 = catalyst.launch_kernel @module_circ::@circ(%c, %cst) : (tensor<1xi64>, tensor<1xf64>) -> tensor<f64>
    return %0 : tensor<f64>
  }
  module @module_circ {
    module attributes {transform.with_named_sequence} {
      transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
        transform.yield
      }
    }

    // CHECK-LABEL: func.func public @circ.quantum
    // CHECK-SAME: () -> (tensor<f64>, tensor<f64>, tensor<f64>)
    // CHECK: quantum.device
    // CHECK: quantum.alloc
    // CHECK: quantum.custom "RY"
    // CHECK: quantum.custom "RX"
    // CHECK: %[[OBS_Z:.*]] = quantum.namedobs %{{.*}}[ PauliZ]
    // CHECK: %[[OBS_X:.*]] = quantum.namedobs %{{.*}}[ PauliX]
    // CHECK: %[[EXPVAL_Z:.*]] = quantum.expval %[[OBS_Z]]
    // CHECK: %[[TENSOR_Z:.*]] = tensor.from_elements %[[EXPVAL_Z]]
    // CHECK: %[[EXPVAL_X:.*]] = quantum.expval %[[OBS_X]]
    // CHECK: %[[TENSOR_X:.*]] = tensor.from_elements %[[EXPVAL_X]]
    // CHECK-NOT: quantum.namedobs.*Identity
    // CHECK-NOT: quantum.expval.*Identity
    // CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f64
    // CHECK: %[[TENSOR_IDENTITY:.*]] = tensor.from_elements %[[ONE]]
    // CHECK: quantum.dealloc
    // CHECK: quantum.device_release
    // CHECK: return %[[TENSOR_Z]], %[[TENSOR_X]], %[[TENSOR_IDENTITY]]

    // CHECK-LABEL: func.func public @circ
    // CHECK-SAME: (%arg0: tensor<1xi64>, %arg1: tensor<1xf64>) -> tensor<f64>
    // CHECK-NOT: quantum.hamiltonian
    // CHECK-NOT: quantum.custom "RY"
    // CHECK-NOT: quantum.custom "RX"
    // CHECK-NOT: quantum.expval
    // CHECK-NOT: quantum.namedobs
    // CHECK: %[[CALL:.*]]:3 = call @circ.quantum()
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[EXTRACTED_0:.*]] = tensor.extract %{{.*}}[%[[C0]]]
    // CHECK: %[[COEFF0:.*]] = tensor.from_elements %[[EXTRACTED_0]]
    // CHECK: %[[C1:.*]] = arith.constant 1 : index
    // CHECK: %[[EXTRACTED_1:.*]] = tensor.extract %{{.*}}[%[[C1]]]
    // CHECK: %[[COEFF1_TEMP:.*]] = tensor.from_elements %[[EXTRACTED_1]]
    // CHECK: %[[C0_3:.*]] = arith.constant 0 : index
    // CHECK: %[[EXTRACTED_4:.*]] = tensor.extract %{{.*}}[%[[C0_3]]]
    // CHECK: %[[COEFF1_NESTED:.*]] = tensor.from_elements %[[EXTRACTED_4]]
    // CHECK: %[[COEFF1:.*]] = stablehlo.multiply %[[COEFF1_TEMP]], %[[COEFF1_NESTED]]
    // CHECK: %[[C2:.*]] = arith.constant 2 : index
    // CHECK: %[[EXTRACTED_6:.*]] = tensor.extract %{{.*}}[%[[C2]]]
    // CHECK: %[[COEFF2_TEMP:.*]] = tensor.from_elements %[[EXTRACTED_6]]
    // CHECK: %[[C0_8:.*]] = arith.constant 0 : index
    // CHECK: %[[EXTRACTED_9:.*]] = tensor.extract %arg1[%[[C0_8]]]
    // CHECK: %[[COEFF2_NESTED:.*]] = tensor.from_elements %[[EXTRACTED_9]]
    // CHECK: %[[COEFF2:.*]] = stablehlo.multiply %[[COEFF2_TEMP]], %[[COEFF2_NESTED]]
    // CHECK: %[[W0:.*]] = stablehlo.multiply %[[COEFF0]], %[[CALL]]#0
    // CHECK: %[[W1:.*]] = stablehlo.multiply %[[COEFF1]], %[[CALL]]#1
    // CHECK: %[[W2:.*]] = stablehlo.multiply %[[COEFF2]], %[[CALL]]#2
    // CHECK: stablehlo.broadcast_in_dim
    // CHECK: stablehlo.broadcast_in_dim
    // CHECK: stablehlo.broadcast_in_dim
    // CHECK: %[[CONCAT:.*]] = stablehlo.concatenate
    // CHECK: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00>
    // CHECK: %[[RESULT:.*]] = stablehlo.reduce(%[[CONCAT]] init: %[[ZERO]]) applies stablehlo.add
    // CHECK: return %[[RESULT]]

    func.func public @circ(%arg0: tensor<1xi64>, %arg1: tensor<1xf64>) -> tensor<f64> attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
      %c = stablehlo.constant dense<0> : tensor<i64>
      %extracted = tensor.extract %c[] : tensor<i64>
      quantum.device shots(%extracted) ["/path/to/lightning.dylib", "LightningSimulator", "{}"]
      %c_0 = stablehlo.constant dense<3> : tensor<i64>
      %0 = quantum.alloc( 3) : !quantum.reg
      %extracted_1 = tensor.extract %c[] : tensor<i64>
      %1 = quantum.extract %0[%extracted_1] : !quantum.reg -> !quantum.bit
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f64>
      %extracted_2 = tensor.extract %cst[] : tensor<f64>
      %out_qubits = quantum.custom "RY"(%extracted_2) %1 : !quantum.bit
      %c_3 = stablehlo.constant dense<1> : tensor<i64>
      %extracted_4 = tensor.extract %c_3[] : tensor<i64>
      %2 = quantum.extract %0[%extracted_4] : !quantum.reg -> !quantum.bit
      %cst_5 = stablehlo.constant dense<3.000000e-01> : tensor<f64>
      %extracted_6 = tensor.extract %cst_5[] : tensor<f64>
      %out_qubits_7 = quantum.custom "RX"(%extracted_6) %2 : !quantum.bit
      %3 = quantum.namedobs %out_qubits[ PauliZ] : !quantum.obs
      %4 = quantum.namedobs %out_qubits_7[ PauliX] : !quantum.obs
      %5 = stablehlo.convert %arg0 : (tensor<1xi64>) -> tensor<1xf64>
      %6 = quantum.hamiltonian(%5 : tensor<1xf64>) %4 : !quantum.obs
      %c_8 = stablehlo.constant dense<2> : tensor<i64>
      %extracted_9 = tensor.extract %c_8[] : tensor<i64>
      %7 = quantum.extract %0[%extracted_9] : !quantum.reg -> !quantum.bit
      %8 = quantum.namedobs %7[ Identity] : !quantum.obs
      %9 = quantum.hamiltonian(%arg1 : tensor<1xf64>) %8 : !quantum.obs
      %cst_10 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
      %10 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %11 = quantum.hamiltonian(%10 : tensor<3xf64>) %3, %6, %9 : !quantum.obs
      %12 = quantum.expval %11 : f64
      %from_elements = tensor.from_elements %12 : tensor<f64>
      %extracted_11 = tensor.extract %c[] : tensor<i64>
      %13 = quantum.insert %0[%extracted_11], %out_qubits : !quantum.reg, !quantum.bit
      %extracted_12 = tensor.extract %c_3[] : tensor<i64>
      %14 = quantum.insert %13[%extracted_12], %out_qubits_7 : !quantum.reg, !quantum.bit
      quantum.dealloc %14 : !quantum.reg
      quantum.device_release
      return %from_elements : tensor<f64>
    }
  }
  func.func @setup() {
    quantum.init
    return
  }
  func.func @teardown() {
    quantum.finalize
    return
  }
}

