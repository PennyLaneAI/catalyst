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
  func.func public @test_two_hadamards() -> tensor<4xf64> attributes {quantum.node} {
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
  func.func private @Hadamard_to_RY_decomp(%arg0: !quantum.bit) -> !quantum.bit attributes {target_gate = "Hadamard", llvm.linkage = #llvm.linkage<internal>} {
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
  func.func private @Hadamard_to_RY_decomp(%arg0: !quantum.bit) -> !quantum.bit attributes {target_gate = "Hadamard", llvm.linkage = #llvm.linkage<internal>} {
      %cst = arith.constant 3.1415926535897931 : f64
      %cst_0 = arith.constant 1.5707963267948966 : f64
      %out_qubits = quantum.custom "RZ"(%cst) %arg0 : !quantum.bit
      %out_qubits_1 = quantum.custom "RY"(%cst_0) %out_qubits : !quantum.bit
      return %out_qubits_1 : !quantum.bit
  }
}

// -----
module @recursive {
  func.func public @test_recursive() -> tensor<4xf64> attributes {quantum.node} {
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
  func.func private @Hadamard_to_RY_decomp(%arg0: !quantum.bit) -> !quantum.bit attributes {target_gate = "Hadamard", llvm.linkage = #llvm.linkage<internal>} {
    %out_qubits_0 = quantum.custom "RZRY"() %arg0 : !quantum.bit
    return %out_qubits_0 : !quantum.bit
  }

  // Decomposition function should be applied and removed from the module
  // CHECK-NOT: func.func private @RZRY_decomp
  func.func private @RZRY_decomp(%arg0: !quantum.bit) -> !quantum.bit attributes {target_gate = "RZRY", llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 3.1415926535897931 : f64
    %cst_0 = arith.constant 1.5707963267948966 : f64
    %out_qubits_1 = quantum.custom "RZ"(%cst) %arg0 : !quantum.bit
    %out_qubits_2 = quantum.custom "RY"(%cst_0) %out_qubits_1 : !quantum.bit
    return %out_qubits_2 : !quantum.bit
  }
}

// -----
module @recursive {
  func.func public @test_recursive() -> tensor<4xf64> attributes {quantum.node} {
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
  func.func private @Hadamard_to_RY_decomp(%arg0: !quantum.bit) -> !quantum.bit attributes {target_gate = "Hadamard", llvm.linkage = #llvm.linkage<internal>} {
    %out_qubits_0 = quantum.custom "RZRY"() %arg0 : !quantum.bit
    return %out_qubits_0 : !quantum.bit
  }

  // Decomposition function should be applied and removed from the module
  // CHECK-NOT: func.func private @RZRY_decomp
  func.func private @RZRY_decomp(%arg0: !quantum.bit) -> !quantum.bit attributes {target_gate = "RZRY", llvm.linkage = #llvm.linkage<internal>} {
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
  func.func public @test_param_rxry(%arg0: tensor<f64>, %arg1: tensor<i64>) -> tensor<2xf64> attributes {quantum.node} {
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
      attributes {target_gate = "ParametrizedRXRY", llvm.linkage = #llvm.linkage<internal>} {
    %extracted = tensor.extract %arg0[] : tensor<f64>
    %out_qubits = quantum.custom "RX"(%extracted) %arg1 : !quantum.bit
    %extracted_0 = tensor.extract %arg0[] : tensor<f64>
    %out_qubits_1 = quantum.custom "RY"(%extracted_0) %out_qubits : !quantum.bit
    return %out_qubits_1 : !quantum.bit
  }
}
// -----

// Test parametric gates and wires
module @param_rxry_2 {
  func.func public @test_param_rxry_2(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<i64>) -> tensor<2xf64> attributes {quantum.node} {
    %c0_i64 = arith.constant 0 : i64

    // CHECK: [[REG:%.+]] = quantum.alloc( 1) : !quantum.reg
    %0 = quantum.alloc( 1) : !quantum.reg

    // CHECK: [[WIRE:%.+]] = tensor.extract %arg2[] : tensor<i64>
    %extracted = tensor.extract %arg2[] : tensor<i64>

    // CHECK: [[QUBIT:%.+]] = quantum.extract [[REG]][[[WIRE]]] : !quantum.reg -> !quantum.bit
    %1 = quantum.extract %0[%extracted] : !quantum.reg -> !quantum.bit

    // CHECK: [[PARAM_0:%.+]] = tensor.extract %arg0[] : tensor<f64>
    %param_0 = tensor.extract %arg0[] : tensor<f64>

    // CHECK: [[PARAM_1:%.+]] = tensor.extract %arg1[] : tensor<f64>
    %param_1 = tensor.extract %arg1[] : tensor<f64>

    // CHECK: [[QUBIT1:%.+]] = quantum.custom "RX"([[PARAM_0]]) [[QUBIT]] : !quantum.bit
    // CHECK: [[QUBIT2:%.+]] = quantum.custom "RY"([[PARAM_1]]) [[QUBIT1]] : !quantum.bit
    // CHECK-NOT: quantum.custom "ParametrizedRXRY"
    %out_qubits = quantum.custom "ParametrizedRXRY"(%param_0, %param_1) %1 : !quantum.bit

    // CHECK: [[UPDATED_REG:%.+]] = quantum.insert [[REG]][ 0], [[QUBIT2]] : !quantum.reg, !quantum.bit
    %2 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit
    %3 = quantum.compbasis qreg %2 : !quantum.obs
    %4 = quantum.probs %3 : tensor<2xf64>
    quantum.dealloc %2 : !quantum.reg
    return %4 : tensor<2xf64>
  }

  // Decomposition function expects tensor<f64> while operation provides f64
  // CHECK-NOT: func.func private @ParametrizedRX_decomp
  func.func private @ParametrizedRXRY_decomp(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: !quantum.bit) -> !quantum.bit
      attributes {target_gate = "ParametrizedRXRY", llvm.linkage = #llvm.linkage<internal>} {
    %extracted_param_0 = tensor.extract %arg0[] : tensor<f64>
    %out_qubits = quantum.custom "RX"(%extracted_param_0) %arg2 : !quantum.bit
    %extracted_param_1 = tensor.extract %arg1[] : tensor<f64>
    %out_qubits_1 = quantum.custom "RY"(%extracted_param_1) %out_qubits : !quantum.bit
    return %out_qubits_1 : !quantum.bit
  }
}
// -----

// Test recursive and qreg-based gate decomposition
module @qreg_base_circuit {
  func.func public @test_qreg_base_circuit() -> tensor<2xf64> attributes {quantum.node} {
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
    func.func private @Test_rule_1(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<1xi64>) -> !quantum.reg
        attributes {target_gate = "Test", llvm.linkage = #llvm.linkage<internal>} {
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
    func.func private @RzDecomp_rule_1(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<1xi64>) -> !quantum.reg
        attributes {target_gate = "RzDecomp", llvm.linkage = #llvm.linkage<internal>} {
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

// -----

module @multi_wire_cnot_decomposition {
  func.func public @test_cnot_decomposition() -> tensor<4xf64> attributes {quantum.node} {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    // CHECK: [[CST_PI:%.+]] = arith.constant 3.1415926535897931 : f64
    // CHECK: [[CST_PI2:%.+]] = arith.constant 1.5707963267948966 : f64
    // CHECK: [[WIRE_TENSOR:%.+]] = arith.constant dense<[0, 1]> : tensor<2xi64>
    // CHECK: [[REG:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[SLICE1:%.+]] = stablehlo.slice [[WIRE_TENSOR]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    // CHECK: [[RESHAPE1:%.+]] = stablehlo.reshape [[SLICE1]] : (tensor<1xi64>) -> tensor<i64>
    // CHECK: [[SLICE2:%.+]] = stablehlo.slice [[WIRE_TENSOR]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    // CHECK: [[RESHAPE2:%.+]] = stablehlo.reshape [[SLICE2]] : (tensor<1xi64>) -> tensor<i64>
    // CHECK: [[EXTRACTED:%.+]] = tensor.extract [[RESHAPE2]][] : tensor<i64>
    // CHECK: [[QUBIT1:%.+]] = quantum.extract [[REG]][[[EXTRACTED]]] : !quantum.reg -> !quantum.bit
    // CHECK: [[RZ1:%.+]] = quantum.custom "RZ"([[CST_PI]]) [[QUBIT1]] : !quantum.bit
    // CHECK: [[RY1:%.+]] = quantum.custom "RY"([[CST_PI2]]) [[RZ1]] : !quantum.bit
    // CHECK: [[EXTRACTED2:%.+]] = tensor.extract [[RESHAPE1]][] : tensor<i64>
    // CHECK: [[QUBIT0:%.+]] = quantum.extract [[REG]][[[EXTRACTED2]]] : !quantum.reg -> !quantum.bit
    // CHECK: [[CZ_RESULT:%.+]]:2 = quantum.custom "CZ"() [[QUBIT0]], [[RY1]] : !quantum.bit, !quantum.bit
    // CHECK: [[INSERT2:%.+]] = quantum.insert [[REG]][[[EXTRACTED2]]], [[CZ_RESULT]]#0 : !quantum.reg, !quantum.bit
    // CHECK: [[RZ2:%.+]] = quantum.custom "RZ"([[CST_PI]]) [[CZ_RESULT]]#1 : !quantum.bit
    // CHECK: [[RY2:%.+]] = quantum.custom "RY"([[CST_PI2]]) [[RZ2]] : !quantum.bit
    // CHECK: [[INSERT3:%.+]] = quantum.insert [[INSERT2]][[[EXTRACTED]]], [[RY2]] : !quantum.reg, !quantum.bit
    // CHECK: [[FINAL_QUBIT0:%.+]] = quantum.extract [[INSERT3]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[FINAL_QUBIT1:%.+]] = quantum.extract [[INSERT3]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK-NOT: quantum.custom "CNOT"
    %3, %4 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit

    // CHECK: [[FINAL_INSERT1:%.+]] = quantum.insert [[REG]][ 0], [[FINAL_QUBIT0]] : !quantum.reg, !quantum.bit
    // CHECK: [[FINAL_INSERT2:%.+]] = quantum.insert [[FINAL_INSERT1]][ 1], [[FINAL_QUBIT1]] : !quantum.reg, !quantum.bit
    %5 = quantum.insert %0[ 0], %3 : !quantum.reg, !quantum.bit
    %6 = quantum.insert %5[ 1], %4 : !quantum.reg, !quantum.bit
    %7 = quantum.compbasis qreg %6 : !quantum.obs
    %8 = quantum.probs %7 : tensor<4xf64>
    quantum.dealloc %6 : !quantum.reg
    return %8 : tensor<4xf64>
  }

  // Decomposition function should be applied and removed from the module
  // CHECK-NOT: func.func private @CNOT_rule_cz_rz_ry
  func.func private @CNOT_rule_cz_rz_ry(%arg0: !quantum.reg, %arg1: tensor<2xi64>) -> !quantum.reg attributes {target_gate = "CNOT", llvm.linkage = #llvm.linkage<internal>} {
    // CNOT decomposition: CNOT = (I ⊗ H) * CZ * (I ⊗ H)
    %cst = arith.constant 1.5707963267948966 : f64
    %cst_0 = arith.constant 3.1415926535897931 : f64

    // Extract wire indices from tensor
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %2 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>

    // Step 1: Apply H to target qubit (H = RZ(π) * RY(π/2))
    %extracted = tensor.extract %3[] : tensor<i64>
    %4 = quantum.extract %arg0[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "RZ"(%cst_0) %4 : !quantum.bit
    %out_qubits_1 = quantum.custom "RY"(%cst) %out_qubits : !quantum.bit
    %extracted_2 = tensor.extract %3[] : tensor<i64>
    %5 = quantum.insert %arg0[%extracted_2], %out_qubits_1 : !quantum.reg, !quantum.bit

    // Step 2: Apply CZ gate
    %extracted_3 = tensor.extract %1[] : tensor<i64>
    %6 = quantum.extract %5[%extracted_3] : !quantum.reg -> !quantum.bit
    %extracted_4 = tensor.extract %3[] : tensor<i64>
    %7 = quantum.extract %5[%extracted_4] : !quantum.reg -> !quantum.bit
    %out_qubits_5:2 = quantum.custom "CZ"() %6, %7 : !quantum.bit, !quantum.bit
    %extracted_6 = tensor.extract %1[] : tensor<i64>
    %8 = quantum.insert %5[%extracted_6], %out_qubits_5#0 : !quantum.reg, !quantum.bit
    %extracted_7 = tensor.extract %3[] : tensor<i64>
    %9 = quantum.insert %8[%extracted_7], %out_qubits_5#1 : !quantum.reg, !quantum.bit

    // Step 3: Apply H to target qubit again
    %extracted_8 = tensor.extract %3[] : tensor<i64>
    %10 = quantum.extract %9[%extracted_8] : !quantum.reg -> !quantum.bit
    %out_qubits_9 = quantum.custom "RZ"(%cst_0) %10 : !quantum.bit
    %out_qubits_10 = quantum.custom "RY"(%cst) %out_qubits_9 : !quantum.bit
    %extracted_11 = tensor.extract %3[] : tensor<i64>
    %11 = quantum.insert %9[%extracted_11], %out_qubits_10 : !quantum.reg, !quantum.bit

    return %11 : !quantum.reg
  }
}

// -----

module @cnot_alternative_decomposition {
  func.func public @test_cnot_alternative_decomposition() -> tensor<4xf64> attributes {quantum.node} {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    // CHECK: [[CST_PI:%.+]] = arith.constant 3.1415926535897931 : f64
    // CHECK: [[CST_PI2:%.+]] = arith.constant 1.5707963267948966 : f64
    // CHECK: [[REG:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[QUBIT0:%.+]] = quantum.extract [[REG]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[QUBIT1:%.+]] = quantum.extract [[REG]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[RZ1:%.+]] = quantum.custom "RZ"([[CST_PI]]) [[QUBIT1]] : !quantum.bit
    // CHECK: [[RY1:%.+]] = quantum.custom "RY"([[CST_PI2]]) [[RZ1]] : !quantum.bit
    // CHECK: [[CZ_RESULT:%.+]]:2 = quantum.custom "CZ"() [[QUBIT0]], [[RY1]] : !quantum.bit, !quantum.bit
    // CHECK: [[RZ2:%.+]] = quantum.custom "RZ"([[CST_PI]]) [[CZ_RESULT]]#1 : !quantum.bit
    // CHECK: [[RY2:%.+]] = quantum.custom "RY"([[CST_PI2]]) [[RZ2]] : !quantum.bit
    // CHECK-NOT: quantum.custom "CNOT"
    %3, %4 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit

    // CHECK: [[FINAL_INSERT1:%.+]] = quantum.insert [[REG]][ 0], [[CZ_RESULT]]#0 : !quantum.reg, !quantum.bit
    // CHECK: [[FINAL_INSERT2:%.+]] = quantum.insert [[FINAL_INSERT1]][ 1], [[RY2]] : !quantum.reg, !quantum.bit
    %5 = quantum.insert %0[ 0], %3 : !quantum.reg, !quantum.bit
    %6 = quantum.insert %5[ 1], %4 : !quantum.reg, !quantum.bit
    %7 = quantum.compbasis qreg %6 : !quantum.obs
    %8 = quantum.probs %7 : tensor<4xf64>
    quantum.dealloc %6 : !quantum.reg
    return %8 : tensor<4xf64>
  }

  // Decomposition function should be applied and removed from the module
  // CHECK-NOT: func.func private @CNOT_rule_h_cnot_h
  func.func private @CNOT_rule_h_cnot_h(%arg0: !quantum.bit, %arg1: !quantum.bit) -> (!quantum.bit, !quantum.bit) attributes {target_gate = "CNOT", llvm.linkage = #llvm.linkage<internal>} {
    // CNOT decomposition: CNOT = (I ⊗ H) * CZ * (I ⊗ H)
    %cst = arith.constant 1.5707963267948966 : f64
    %cst_0 = arith.constant 3.1415926535897931 : f64

    // Step 1: Apply H to target qubit (H = RZ(π) * RY(π/2))
    %out_qubits = quantum.custom "RZ"(%cst_0) %arg1 : !quantum.bit
    %out_qubits_1 = quantum.custom "RY"(%cst) %out_qubits : !quantum.bit

    // Step 2: Apply CZ gate
    %out_qubits_2:2 = quantum.custom "CZ"() %arg0, %out_qubits_1 : !quantum.bit, !quantum.bit

    // Step 3: Apply H to target qubit again
    %out_qubits_3 = quantum.custom "RZ"(%cst_0) %out_qubits_2#1 : !quantum.bit
    %out_qubits_4 = quantum.custom "RY"(%cst) %out_qubits_3 : !quantum.bit

    return %out_qubits_2#0, %out_qubits_4 : !quantum.bit, !quantum.bit
  }
}

// -----

module @mcm_example {
  func.func public @test_mcm_hadamard() -> tensor<2xf64> attributes {quantum.node} {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %mres, %out_qubit = quantum.measure %1 : i1, !quantum.bit
    %2 = quantum.insert %0[ 0], %out_qubit : !quantum.reg, !quantum.bit

    // CHECK: [[RZ_QUBIT:%.+]] = quantum.custom "RZ"([[CST_0:%.+]])
    // CHECK: [[RY_QUBIT:%.+]] = quantum.custom "RY"([[CST_1:%.+]]) [[RZ_QUBIT]] : !quantum.bit
    // CHECK: [[REG_1:%.+]] = quantum.insert [[REG:%.+]][[[EXTRACTED:%.+]]], [[RY_QUBIT]] : !quantum.reg, !quantum.bit
    // CHECK-NOT: quantum.custom "Hadamard"
    %3 = quantum.extract %2[ 0] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Hadamard"() %3 : !quantum.bit
    %4 = quantum.insert %2[ 0], %out_qubits : !quantum.reg, !quantum.bit

    %5 = quantum.compbasis qreg %4 : !quantum.obs
    %6 = quantum.probs %5 : tensor<2xf64>
    quantum.dealloc %4 : !quantum.reg
    return %6 : tensor<2xf64>
  }

  // Decomposition function should be applied and removed from the module
  // CHECK-NOT: func.func public @rz_ry
  func.func public @rz_ry(%arg0: !quantum.reg, %arg1: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "Hadamard"} {
    %cst = arith.constant 3.1415926535897931 : f64
    %cst_0 = arith.constant 1.5707963267948966 : f64
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = quantum.extract %arg0[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "RZ"(%cst_0) %2 : !quantum.bit
    %3 = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %extracted_1 = tensor.extract %1[] : tensor<i64>
    %5 = quantum.insert %arg0[%extracted_1], %out_qubits : !quantum.reg, !quantum.bit
    %extracted_2 = tensor.extract %4[] : tensor<i64>
    %6 = quantum.extract %5[%extracted_2] : !quantum.reg -> !quantum.bit
    %out_qubits_3 = quantum.custom "RY"(%cst) %6 : !quantum.bit
    %extracted_4 = tensor.extract %4[] : tensor<i64>
    %7 = quantum.insert %5[%extracted_4], %out_qubits_3 : !quantum.reg, !quantum.bit
    return %7 : !quantum.reg
  }
}

// -----

module @circuit_with_multirz {
  func.func public @test_with_multirz() -> tensor<4xf64> attributes {quantum.node} {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // CHECK: func.func public @test_with_multirz() -> tensor<4xf64>
    // CHECK: [[CST_RZ:%.+]] = arith.constant 5.000000e-01 : f64
    // CHECK: [[CST_PI2:%.+]] = arith.constant 1.5707963267948966 : f64
    // CHECK: [[CST_PI:%.+]] = arith.constant 3.1415926535897931 : f64
    // CHECK: [[REG:%.+]] = quantum.alloc( 2) : !quantum.reg

    // CHECK: [[QUBIT1:%.+]] = quantum.custom "RZ"([[CST_RZ]]) {{%.+}} : !quantum.bit
    // CHECK-NOT: quantum.multirz
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %extracted_2 = tensor.extract %cst[] : tensor<f64>
    %out_qubits = quantum.multirz(%extracted_2) %1 : !quantum.bit

    // CHECK: [[QUBIT3:%.+]] = quantum.custom "RZ"([[CST_PI]]) {{%.+}} : !quantum.bit
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

  // CHECK-NOT: func.func private @Hadamard_to_RY_decomp
  func.func private @Hadamard_to_RY_decomp(%arg0: !quantum.bit) -> !quantum.bit attributes {target_gate = "Hadamard", llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 3.1415926535897931 : f64
    %cst_0 = arith.constant 1.5707963267948966 : f64
    %out_qubits = quantum.custom "RZ"(%cst) %arg0 : !quantum.bit
    %out_qubits_1 = quantum.custom "RY"(%cst_0) %out_qubits : !quantum.bit
    return %out_qubits_1 : !quantum.bit
  }

  // CHECK-NOT: func.func private @_multi_rz_decomposition_wires_1
  func.func public @_multi_rz_decomposition_wires_1(%arg0: !quantum.reg, %arg1: tensor<1xf64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "MultiRZ"} {
    %0 = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = quantum.extract %arg0[%extracted] : !quantum.reg -> !quantum.bit
    %c0 = arith.constant 0 : index
    %extracted_0 = tensor.extract %arg1[%c0] : tensor<1xf64>
    %out_qubits = quantum.custom "RZ"(%extracted_0) %2 : !quantum.bit
    %extracted_1 = tensor.extract %1[] : tensor<i64>
    %3 = quantum.insert %arg0[%extracted_1], %out_qubits : !quantum.reg, !quantum.bit
    return %3 : !quantum.reg
  }
}
