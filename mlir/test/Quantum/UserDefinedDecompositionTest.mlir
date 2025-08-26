// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --user-defined-decomposition --split-input-file -verify-diagnostics %s | FileCheck %s

module @two_hadamards {
  func.func public @test_two_hadamards() -> tensor<4xf64> {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[CST_PI2:%.+]] = arith.constant 1.5707963267948966 : f64
    // CHECK: [[CST_PI:%.+]] = arith.constant 3.1415926535897931 : f64
    // CHECK: [[REG:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[QUBIT:%.+]] = quantum.extract [[REG]][ 0] : !quantum.reg -> !quantum.bit

    // CHECK: [[QUBIT1:%.+]] = quantum.custom "RZ"([[CST_PI]]) [[QUBIT]] : !quantum.bit
    // CHECK: [[QUBIT2:%.+]] = quantum.custom "RY"([[CST_PI2]]) [[QUBIT1]] : !quantum.bit
    // CHECK-NOT: quantum.custom "Hadamard"
    %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit

    // CHECK: [[QUBIT3:%.+]] = quantum.custom "RZ"([[CST_PI]]) [[QUBIT2]] : !quantum.bit
    // CHECK: [[QUBIT4:%.+]] = quantum.custom "RY"([[CST_PI2]]) [[QUBIT3]] : !quantum.bit
    // CHECK-NOT: quantum.custom "Hadamard"
    %out_qubits_0 = quantum.custom "Hadamard"() %out_qubits : !quantum.bit

    // CHECK: [[UPDATED_REG:%.+]] = quantum.insert [[REG]][ 0], [[QUBIT4]] : !quantum.reg, !quantum.bit
    %2 = quantum.insert %0[ 0], %out_qubits_0 : !quantum.reg, !quantum.bit
    %3 = quantum.compbasis qreg %2 : !quantum.obs
    %4 = quantum.probs %3 : tensor<4xf64>
    quantum.dealloc %2 : !quantum.reg
    return %4 : tensor<4xf64>
  }

  // Decomposition function should be applied and removed from the module
  // CHECK-NOT: func.func private @Hadamard_to_RY_decomp
  func.func private @Hadamard_to_RY_decomp(%arg0: !quantum.bit) -> !quantum.bit attributes {catalyst.decomposition, catalyst.decomposition.target_op = "Hadamard", llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 3.1415926535897931 : f64
    %cst_0 = arith.constant 1.5707963267948966 : f64
    %out_qubits = quantum.custom "RZ"(%cst) %arg0 : !quantum.bit
    %out_qubits_1 = quantum.custom "RY"(%cst_0) %out_qubits : !quantum.bit
    return %out_qubits_1 : !quantum.bit
  }
}

// -----

// Test single Hadamard decomposition
module @single_hadamard {
  func.func @test_single_hadamard() -> !quantum.bit {
      // CHECK: [[CST_PI2:%.+]] = arith.constant 1.5707963267948966 : f64
      // CHECK: [[CST_PI:%.+]] = arith.constant 3.1415926535897931 : f64
      // CHECK: [[REG:%.+]] = quantum.alloc( 1) : !quantum.reg
      // CHECK: [[QUBIT:%.+]] = quantum.extract [[REG]][ 0] : !quantum.reg -> !quantum.bit
      %0 = quantum.alloc( 1) : !quantum.reg
      %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit

      // CHECK: [[QUBIT1:%.+]] = quantum.custom "RZ"([[CST_PI]]) [[QUBIT]] : !quantum.bit
      // CHECK: [[QUBIT2:%.+]] = quantum.custom "RY"([[CST_PI2]]) [[QUBIT1]] : !quantum.bit
      // CHECK-NOT: quantum.custom "Hadamard"
      %2 = quantum.custom "Hadamard"() %1 : !quantum.bit

      // CHECK: return [[QUBIT2]]
      return %2 : !quantum.bit
  }

  // Decomposition function should be applied and removed from the module
  // CHECK-NOT: func.func private @Hadamard_to_RY_decomp
  func.func private @Hadamard_to_RY_decomp(%arg0: !quantum.bit) -> !quantum.bit attributes {catalyst.decomposition, catalyst.decomposition.target_op = "Hadamard", llvm.linkage = #llvm.linkage<internal>} {
      %cst = arith.constant 3.1415926535897931 : f64
      %cst_0 = arith.constant 1.5707963267948966 : f64
      %out_qubits = quantum.custom "RZ"(%cst) %arg0 : !quantum.bit
      %out_qubits_1 = quantum.custom "RY"(%cst_0) %out_qubits : !quantum.bit
      return %out_qubits_1 : !quantum.bit
  }
}

// -----
module @recursive {
  func.func public @test_recursive() -> tensor<4xf64> {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[CST_PI2:%.+]] = arith.constant 1.5707963267948966 : f64
    // CHECK: [[CST_PI:%.+]] = arith.constant 3.1415926535897931 : f64
    // CHECK: [[REG:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[QUBIT:%.+]] = quantum.extract [[REG]][ 0] : !quantum.reg -> !quantum.bit

    // CHECK: [[QUBIT1:%.+]] = quantum.custom "RZ"([[CST_PI]]) [[QUBIT]] : !quantum.bit
    // CHECK: [[QUBIT2:%.+]] = quantum.custom "RY"([[CST_PI2]]) [[QUBIT1]] : !quantum.bit
    // CHECK-NOT: quantum.custom "Hadamard"
    %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit

    // CHECK: [[QUBIT3:%.+]] = quantum.custom "RZ"([[CST_PI]]) [[QUBIT2]] : !quantum.bit
    // CHECK: [[QUBIT4:%.+]] = quantum.custom "RY"([[CST_PI2]]) [[QUBIT3]] : !quantum.bit
    // CHECK-NOT: quantum.custom "Hadamard"
    %out_qubits_0 = quantum.custom "Hadamard"() %out_qubits : !quantum.bit

    // CHECK: [[UPDATED_REG:%.+]] = quantum.insert [[REG]][ 0], [[QUBIT4]] : !quantum.reg, !quantum.bit
    %2 = quantum.insert %0[ 0], %out_qubits_0 : !quantum.reg, !quantum.bit
    %3 = quantum.compbasis qreg %2 : !quantum.obs
    %4 = quantum.probs %3 : tensor<4xf64>
    quantum.dealloc %2 : !quantum.reg
    return %4 : tensor<4xf64>
  }

  // Decomposition function should be applied and removed from the module
  // CHECK-NOT: func.func private @Hadamard_to_RY_decomp
  func.func private @Hadamard_to_RY_decomp(%arg0: !quantum.bit) -> !quantum.bit attributes {catalyst.decomposition, catalyst.decomposition.target_op = "Hadamard", llvm.linkage = #llvm.linkage<internal>} {
    %out_qubits_0 = quantum.custom "RZRY"() %arg0 : !quantum.bit
    return %out_qubits_0 : !quantum.bit
  }

  // Decomposition function should be applied and removed from the module
  // CHECK-NOT: func.func private @RZRY_decomp
  func.func private @RZRY_decomp(%arg0: !quantum.bit) -> !quantum.bit attributes {catalyst.decomposition, catalyst.decomposition.target_op = "RZRY", llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 3.1415926535897931 : f64
    %cst_0 = arith.constant 1.5707963267948966 : f64
    %out_qubits_1 = quantum.custom "RZ"(%cst) %arg0 : !quantum.bit
    %out_qubits_2 = quantum.custom "RY"(%cst_0) %out_qubits_1 : !quantum.bit
    return %out_qubits_2 : !quantum.bit
  }
}

// -----
module @recursive {
  func.func public @test_recursive() -> tensor<4xf64> {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[CST_PI2:%.+]] = arith.constant 1.5707963267948966 : f64
    // CHECK: [[CST_PI:%.+]] = arith.constant 3.1415926535897931 : f64
    // CHECK: [[REG:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[QUBIT:%.+]] = quantum.extract [[REG]][ 0] : !quantum.reg -> !quantum.bit

    // CHECK: [[QUBIT1:%.+]] = quantum.custom "RZ"([[CST_PI]]) [[QUBIT]] : !quantum.bit
    // CHECK: [[QUBIT2:%.+]] = quantum.custom "RY"([[CST_PI2]]) [[QUBIT1]] : !quantum.bit
    // CHECK-NOT: quantum.custom "Hadamard"
    %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit

    // CHECK: [[QUBIT3:%.+]] = quantum.custom "RZ"([[CST_PI]]) [[QUBIT2]] : !quantum.bit
    // CHECK: [[QUBIT4:%.+]] = quantum.custom "RY"([[CST_PI2]]) [[QUBIT3]] : !quantum.bit
    // CHECK-NOT: quantum.custom "Hadamard"
    %out_qubits_0 = quantum.custom "Hadamard"() %out_qubits : !quantum.bit

    // CHECK: [[UPDATED_REG:%.+]] = quantum.insert [[REG]][ 0], [[QUBIT4]] : !quantum.reg, !quantum.bit
    %2 = quantum.insert %0[ 0], %out_qubits_0 : !quantum.reg, !quantum.bit
    %3 = quantum.compbasis qreg %2 : !quantum.obs
    %4 = quantum.probs %3 : tensor<4xf64>
    quantum.dealloc %2 : !quantum.reg
    return %4 : tensor<4xf64>
  }

  // Decomposition function should be applied and removed from the module
  // CHECK-NOT: func.func private @Hadamard_to_RY_decomp
  func.func private @Hadamard_to_RY_decomp(%arg0: !quantum.bit) -> !quantum.bit attributes {catalyst.decomposition, catalyst.decomposition.target_op = "Hadamard", llvm.linkage = #llvm.linkage<internal>} {
    %out_qubits_0 = quantum.custom "RZRY"() %arg0 : !quantum.bit
    return %out_qubits_0 : !quantum.bit
  }

  // Decomposition function should be applied and removed from the module
  // CHECK-NOT: func.func private @RZRY_decomp
  func.func private @RZRY_decomp(%arg0: !quantum.bit) -> !quantum.bit attributes {catalyst.decomposition, catalyst.decomposition.target_op = "RZRY", llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 3.1415926535897931 : f64
    %cst_0 = arith.constant 1.5707963267948966 : f64
    %out_qubits_1 = quantum.custom "RZ"(%cst) %arg0 : !quantum.bit
    %out_qubits_2 = quantum.custom "RY"(%cst_0) %out_qubits_1 : !quantum.bit
    return %out_qubits_2 : !quantum.bit
  }
}

// -----

// Test parametric gates and wires
module @param_rxry {
  func.func public @test_param_rxry(%arg0: tensor<f64>, %arg1: tensor<i64>) -> tensor<2xf64> {
    %c0_i64 = arith.constant 0 : i64

    // CHECK: [[REG:%.+]] = quantum.alloc( 1) : !quantum.reg
    %0 = quantum.alloc( 1) : !quantum.reg

    // CHECK: [[WIRE:%.+]] = tensor.extract %arg1[] : tensor<i64>
    %extracted = tensor.extract %arg1[] : tensor<i64>

    // CHECK: [[QUBIT:%.+]] = quantum.extract [[REG]][[[WIRE]]] : !quantum.reg -> !quantum.bit
    %1 = quantum.extract %0[%extracted] : !quantum.reg -> !quantum.bit

    // CHECK: [[PARAM:%.+]] = tensor.extract %arg0[] : tensor<f64>
    %param_0 = tensor.extract %arg0[] : tensor<f64>

    // CHECK: [[QUBIT1:%.+]] = quantum.custom "RX"([[PARAM]]) [[QUBIT]] : !quantum.bit
    // CHECK: [[QUBIT2:%.+]] = quantum.custom "RY"([[PARAM]]) [[QUBIT1]] : !quantum.bit
    // CHECK-NOT: quantum.custom "ParametrizedRXRY"
    %out_qubits = quantum.custom "ParametrizedRXRY"(%param_0) %1 : !quantum.bit

    // CHECK: [[UPDATED_REG:%.+]] = quantum.insert [[REG]][ 0], [[QUBIT2]] : !quantum.reg, !quantum.bit
    %2 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit
    %3 = quantum.compbasis qreg %2 : !quantum.obs
    %4 = quantum.probs %3 : tensor<2xf64>
    quantum.dealloc %2 : !quantum.reg
    return %4 : tensor<2xf64>
  }

  // Decomposition function expects tensor<f64> while operation provides f64
  // CHECK-NOT: func.func private @ParametrizedRX_decomp
  func.func private @ParametrizedRXRY_decomp(%arg0: tensor<f64>, %arg1: !quantum.bit) -> !quantum.bit
      attributes {catalyst.decomposition, catalyst.decomposition.target_op = "ParametrizedRXRY", llvm.linkage = #llvm.linkage<internal>} {
    %extracted = tensor.extract %arg0[] : tensor<f64>
    %out_qubits = quantum.custom "RX"(%extracted) %arg1 : !quantum.bit
    %extracted_0 = tensor.extract %arg0[] : tensor<f64>
    %out_qubits_1 = quantum.custom "RY"(%extracted_0) %out_qubits : !quantum.bit
    return %out_qubits_1 : !quantum.bit
  }
}