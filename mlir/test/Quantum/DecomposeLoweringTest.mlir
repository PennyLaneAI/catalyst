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

// RUN: quantum-opt --decompose-lowering --split-input-file -verify-diagnostics %s | FileCheck %s

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
  func.func private @Hadamard_to_RY_decomp(%arg0: !quantum.bit) -> !quantum.bit attributes {catalyst.decomposition.target_op = "Hadamard", llvm.linkage = #llvm.linkage<internal>} {
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
  func.func private @Hadamard_to_RY_decomp(%arg0: !quantum.bit) -> !quantum.bit attributes {catalyst.decomposition.target_op = "Hadamard", llvm.linkage = #llvm.linkage<internal>} {
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
  func.func private @Hadamard_to_RY_decomp(%arg0: !quantum.bit) -> !quantum.bit attributes {catalyst.decomposition.target_op = "Hadamard", llvm.linkage = #llvm.linkage<internal>} {
    %out_qubits_0 = quantum.custom "RZRY"() %arg0 : !quantum.bit
    return %out_qubits_0 : !quantum.bit
  }

  // Decomposition function should be applied and removed from the module
  // CHECK-NOT: func.func private @RZRY_decomp
  func.func private @RZRY_decomp(%arg0: !quantum.bit) -> !quantum.bit attributes {catalyst.decomposition.target_op = "RZRY", llvm.linkage = #llvm.linkage<internal>} {
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
  func.func private @Hadamard_to_RY_decomp(%arg0: !quantum.bit) -> !quantum.bit attributes {catalyst.decomposition.target_op = "Hadamard", llvm.linkage = #llvm.linkage<internal>} {
    %out_qubits_0 = quantum.custom "RZRY"() %arg0 : !quantum.bit
    return %out_qubits_0 : !quantum.bit
  }

  // Decomposition function should be applied and removed from the module
  // CHECK-NOT: func.func private @RZRY_decomp
  func.func private @RZRY_decomp(%arg0: !quantum.bit) -> !quantum.bit attributes {catalyst.decomposition.target_op = "RZRY", llvm.linkage = #llvm.linkage<internal>} {
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
      attributes {catalyst.decomposition.target_op = "ParametrizedRXRY", llvm.linkage = #llvm.linkage<internal>} {
    %extracted = tensor.extract %arg0[] : tensor<f64>
    %out_qubits = quantum.custom "RX"(%extracted) %arg1 : !quantum.bit
    %extracted_0 = tensor.extract %arg0[] : tensor<f64>
    %out_qubits_1 = quantum.custom "RY"(%extracted_0) %out_qubits : !quantum.bit
    return %out_qubits_1 : !quantum.bit
  }
}
// -----

// Test recursive and qreg-based gate decomposition
module @qreg_base_circuit {
  func.func public @test_qreg_base_circuit() -> tensor<2xf64> {
      // CHECK: [[CST:%.+]] = arith.constant 1.000000e+00 : f64
      %cst = arith.constant 1.000000e+00 : f64

      // CHECK: [[CST_0:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      // CHECK: [[CST_1:%.+]] = arith.constant dense<0> : tensor<1xi64>
      // CHECK: [[CST_2:%.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
      // CHECK: [[REG:%.+]] = quantum.alloc( 1) : !quantum.reg
      %0 = quantum.alloc( 1) : !quantum.reg

      // CHECK: [[EXTRACT_QUBIT:%.+]] = quantum.extract [[REG]][ 0] : !quantum.reg -> !quantum.bit
      // CHECK: [[MRES:%.+]], [[OUT_QUBIT:%.+]] = quantum.measure [[EXTRACT_QUBIT]] : i1, !quantum.bit
      // CHECK: [[REG1:%.+]] = quantum.insert [[REG]][ 0], [[OUT_QUBIT]] : !quantum.reg, !quantum.bit
      // CHECK: [[COMPARE:%.+]] = stablehlo.compare  NE, [[CST_2]], [[CST_0]],  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      // CHECK: [[EXTRACTED:%.+]] = tensor.extract [[COMPARE]][] : tensor<i1>
      // CHECK: [[CONDITIONAL:%.+]] = scf.if [[EXTRACTED]] -> (!quantum.reg) {
      // CHECK:   [[SLICE1:%.+]] = stablehlo.slice [[CST_1]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      // CHECK:   [[RESHAPE1:%.+]] = stablehlo.reshape [[SLICE1]] : (tensor<1xi64>) -> tensor<i64>
      // CHECK:   [[EXTRACTED_3:%.+]] = tensor.extract [[RESHAPE1]][] : tensor<i64>
      // CHECK:   [[FROM_ELEMENTS:%.+]] = tensor.from_elements [[EXTRACTED_3]] : tensor<1xi64>
      // CHECK:   [[SLICE2:%.+]] = stablehlo.slice [[FROM_ELEMENTS]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      // CHECK:   [[RESHAPE2:%.+]] = stablehlo.reshape [[SLICE2]] : (tensor<1xi64>) -> tensor<i64>
      // CHECK:   [[EXTRACTED_4:%.+]] = tensor.extract [[RESHAPE2]][] : tensor<i64>
      // CHECK:   [[EXTRACT1:%.+]] = quantum.extract [[REG1]][[[EXTRACTED_4]]] : !quantum.reg -> !quantum.bit
      // CHECK:   [[RZ1:%.+]] = quantum.custom "RZ"([[CST]]) [[EXTRACT1]] : !quantum.bit
      // CHECK:   [[INSERT1:%.+]] = quantum.insert [[REG1]][[[EXTRACTED_4]]], [[RZ1]] : !quantum.reg, !quantum.bit
      // CHECK:   [[EXTRACT2:%.+]] = quantum.extract [[INSERT1]][[[EXTRACTED_3]]] : !quantum.reg -> !quantum.bit
      // CHECK:   [[INSERT2:%.+]] = quantum.insert [[REG1]][[[EXTRACTED_3]]], [[EXTRACT2]] : !quantum.reg, !quantum.bit
      // CHECK:   [[EXTRACT3:%.+]] = quantum.extract [[INSERT2]][[[EXTRACTED_4]]] : !quantum.reg -> !quantum.bit
      // CHECK:   [[RZ2:%.+]] = quantum.custom "RZ"([[CST]]) [[EXTRACT3]] : !quantum.bit
      // CHECK:   [[INSERT3:%.+]] = quantum.insert [[INSERT2]][[[EXTRACTED_4]]], [[RZ2]] : !quantum.reg, !quantum.bit
      // CHECK:   [[EXTRACT4:%.+]] = quantum.extract [[INSERT3]][[[EXTRACTED_3]]] : !quantum.reg -> !quantum.bit
      // CHECK:   [[INSERT4:%.+]] = quantum.insert [[INSERT2]][[[EXTRACTED_3]]], [[EXTRACT4]] : !quantum.reg, !quantum.bit
      // CHECK:   scf.yield [[INSERT4]] : !quantum.reg
      // CHECK: } else {
      // CHECK:   scf.yield [[REG1]] : !quantum.reg
      // CHECK: }
      // CHECK-NOT: quantum.custom "Test"
      %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "Test"(%cst) %1 : !quantum.bit
      %2 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit
      %3 = quantum.compbasis qreg %2 : !quantum.obs
      %4 = quantum.probs %3 : tensor<2xf64>

      quantum.dealloc %2 : !quantum.reg
      quantum.device_release
      return %4 : tensor<2xf64>
    }

    // Decomposition function should be applied and removed from the module
    // CHECK-NOT: func.func private @Test_rule_1
    func.func private @Test_rule_1(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<1xi64>) -> !quantum.reg {
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %10 = quantum.extract %arg0[ 0] : !quantum.reg -> !quantum.bit
      %mres, %out_qubit = quantum.measure %10 : i1, !quantum.bit
      %11 = quantum.insert %arg0[ 0], %out_qubit : !quantum.reg, !quantum.bit
      %0 = stablehlo.compare  NE, %arg1, %cst,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %extracted = tensor.extract %0[] : tensor<i1>
      %1 = scf.if %extracted -> (!quantum.reg) {
        %2 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
        %extracted_0 = tensor.extract %3[] : tensor<i64>
        %4 = quantum.extract %11[%extracted_0] : !quantum.reg -> !quantum.bit
        %extracted_1 = tensor.extract %arg1[] : tensor<f64>
        %out_qubits = quantum.custom "RzDecomp"(%extracted_1) %4 : !quantum.bit
        %5 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
        %6 = stablehlo.reshape %5 : (tensor<1xi64>) -> tensor<i64>
        %extracted_2 = tensor.extract %3[] : tensor<i64>
        %7 = quantum.insert %11[%extracted_2], %out_qubits : !quantum.reg, !quantum.bit
        %extracted_3 = tensor.extract %6[] : tensor<i64>
        %8 = quantum.extract %7[%extracted_3] : !quantum.reg -> !quantum.bit
        %extracted_4 = tensor.extract %arg1[] : tensor<f64>
        %out_qubits_5 = quantum.custom "RzDecomp"(%extracted_4) %8 : !quantum.bit
        %extracted_6 = tensor.extract %6[] : tensor<i64>
        %9 = quantum.insert %7[%extracted_6], %out_qubits_5 : !quantum.reg, !quantum.bit
        scf.yield %9 : !quantum.reg
      } else {
        scf.yield %11 : !quantum.reg
      }
      return %1 : !quantum.reg
    }

    // Decomposition function should be applied and removed from the module
    // CHECK-NOT: func.func private @RzDecomp_rule_1
    func.func private @RzDecomp_rule_1(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<1xi64>) -> !quantum.reg {
      %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      %extracted = tensor.extract %1[] : tensor<i64>
      %2 = quantum.extract %arg0[%extracted] : !quantum.reg -> !quantum.bit
      %extracted_0 = tensor.extract %arg1[] : tensor<f64>
      %out_qubits = quantum.custom "RZ"(%extracted_0) %2 : !quantum.bit
      %extracted_1 = tensor.extract %1[] : tensor<i64>
      %3 = quantum.insert %arg0[%extracted_1], %out_qubits : !quantum.reg, !quantum.bit
      return %3 : !quantum.reg
    }
}
