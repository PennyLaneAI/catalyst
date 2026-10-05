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

// RUN: quantum-opt --decompose-lowering --split-input-file -verify-diagnostics %s | FileCheck %s --check-prefixes=ALL,CALL
// RUN: quantum-opt --decompose-lowering=inline-rule-body --split-input-file -verify-diagnostics %s | FileCheck %s --check-prefixes=ALL,INLINE

// ALL-LABEL: module @two_hadamards
module @two_hadamards {
  func.func public @test_two_hadamards() -> tensor<4xf64> attributes {quantum.node} {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // INLINE: [[CST_PI2:%.+]] = arith.constant 1.5707963267948966 : f64
    // INLINE: [[CST_PI:%.+]] = arith.constant 3.1415926535897931 : f64
    // ALL: [[REG:%.+]] = quantum.alloc( 2) : !quantum.reg
    // ALL: [[QUBIT:%.+]] = quantum.extract [[REG]][ 0] : !quantum.reg -> !quantum.bit

    // CALL: [[QUBIT2:%.+]] = call @Hadamard_to_RY_decomp_0([[QUBIT]]) : (!quantum.bit) -> !quantum.bit
    // INLINE: [[QUBIT1:%.+]] = quantum.custom "RZ"([[CST_PI]]) [[QUBIT]] : !quantum.bit
    // INLINE: [[QUBIT2:%.+]] = quantum.custom "RY"([[CST_PI2]]) [[QUBIT1]] : !quantum.bit
    // ALL-NOT: quantum.custom "Hadamard"
    %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit

    // CALL: [[QUBIT4:%.+]] = call @Hadamard_to_RY_decomp_0([[QUBIT2]]) : (!quantum.bit) -> !quantum.bit
    // INLINE: [[QUBIT3:%.+]] = quantum.custom "RZ"([[CST_PI]]) [[QUBIT2]] : !quantum.bit
    // INLINE: [[QUBIT4:%.+]] = quantum.custom "RY"([[CST_PI2]]) [[QUBIT3]] : !quantum.bit
    // ALL-NOT: quantum.custom "Hadamard"
    %out_qubits_0 = quantum.custom "Hadamard"() %out_qubits : !quantum.bit

    // ALL: [[UPDATED_REG:%.+]] = quantum.insert [[REG]][ 0], [[QUBIT4]] : !quantum.reg, !quantum.bit
    %2 = quantum.insert %0[ 0], %out_qubits_0 : !quantum.reg, !quantum.bit
    %3 = quantum.compbasis qreg %2 : !quantum.obs
    %4 = quantum.probs %3 : tensor<4xf64>
    quantum.dealloc %2 : !quantum.reg
    return %4 : tensor<4xf64>
  }

  // Decomposition function should be retained for future passes
  // ALL: func.func private @Hadamard_to_RY_decomp
  func.func private @Hadamard_to_RY_decomp(%arg0: !quantum.bit) -> !quantum.bit attributes {target_gate = "Hadamard", llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 3.1415926535897931 : f64
    %cst_0 = arith.constant 1.5707963267948966 : f64
    %out_qubits = quantum.custom "RZ"(%cst) %arg0 : !quantum.bit
    %out_qubits_1 = quantum.custom "RY"(%cst_0) %out_qubits : !quantum.bit
    return %out_qubits_1 : !quantum.bit
  }

  // Call mode should have a clone to the rule function
  // CALL: func.func private @Hadamard_to_RY_decomp_0
}

// -----

// Test single Hadamard decomposition

// ALL-LABEL: module @single_hadamard
module @single_hadamard {
  func.func @test_single_hadamard() -> tensor<2xf64> attributes {quantum.node} {
      // INLINE: [[CST_PI2:%.+]] = arith.constant 1.5707963267948966 : f64
      // INLINE: [[CST_PI:%.+]] = arith.constant 3.1415926535897931 : f64
      // ALL: [[REG:%.+]] = quantum.alloc( 1) : !quantum.reg
      // ALL: [[QUBIT:%.+]] = quantum.extract [[REG]][ 0] : !quantum.reg -> !quantum.bit
      %0 = quantum.alloc( 1) : !quantum.reg
      %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit

      // CALL: [[QUBIT2:%.+]] = call @Hadamard_to_RY_decomp_0([[QUBIT]]) : (!quantum.bit) -> !quantum.bit
      // INLINE: [[QUBIT1:%.+]] = quantum.custom "RZ"([[CST_PI]]) [[QUBIT]] : !quantum.bit
      // INLINE: [[QUBIT2:%.+]] = quantum.custom "RY"([[CST_PI2]]) [[QUBIT1]] : !quantum.bit
      // ALL-NOT: quantum.custom "Hadamard"
      %out_qubits_0 = quantum.custom "Hadamard"() %1 : !quantum.bit

      // ALL: [[UPDATED_REG:%.+]] = quantum.insert [[REG]][ 0], [[QUBIT2]] : !quantum.reg, !quantum.bit
      %2 = quantum.insert %0[ 0], %out_qubits_0 : !quantum.reg, !quantum.bit
      %3 = quantum.compbasis qreg %2 : !quantum.obs
      %4 = quantum.probs %3 : tensor<2xf64>
      quantum.dealloc %2 : !quantum.reg
      return %4 : tensor<2xf64>
  }

  // Decomposition function should be retained for future passes
  // ALL: func.func private @Hadamard_to_RY_decomp
  func.func private @Hadamard_to_RY_decomp(%arg0: !quantum.bit) -> !quantum.bit attributes {target_gate = "Hadamard", llvm.linkage = #llvm.linkage<internal>} {
      %cst = arith.constant 3.1415926535897931 : f64
      %cst_0 = arith.constant 1.5707963267948966 : f64
      %out_qubits = quantum.custom "RZ"(%cst) %arg0 : !quantum.bit
      %out_qubits_1 = quantum.custom "RY"(%cst_0) %out_qubits : !quantum.bit
      return %out_qubits_1 : !quantum.bit
  }

  // Call mode should have a clone to the rule function
  // CALL: func.func private @Hadamard_to_RY_decomp_0
}

// -----

// ALL-LABEL: module @recursive
module @recursive {
  func.func public @test_recursive() -> tensor<4xf64> attributes {quantum.node} {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // INLINE: [[CST_PI2:%.+]] = arith.constant 1.5707963267948966 : f64
    // INLINE: [[CST_PI:%.+]] = arith.constant 3.1415926535897931 : f64
    // ALL: [[REG:%.+]] = quantum.alloc( 2) : !quantum.reg
    // ALL: [[QUBIT:%.+]] = quantum.extract [[REG]][ 0] : !quantum.reg -> !quantum.bit

    // CALL: [[QUBIT2:%.+]] = call @Hadamard_to_RY_decomp_0([[QUBIT]]) : (!quantum.bit) -> !quantum.bit
    // INLINE: [[QUBIT1:%.+]] = quantum.custom "RZ"([[CST_PI]]) [[QUBIT]] : !quantum.bit
    // INLINE: [[QUBIT2:%.+]] = quantum.custom "RY"([[CST_PI2]]) [[QUBIT1]] : !quantum.bit
    // ALL-NOT: quantum.custom "Hadamard"
    %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit

    // CALL: [[QUBIT4:%.+]] = call @Hadamard_to_RY_decomp_0([[QUBIT2]]) : (!quantum.bit) -> !quantum.bit
    // INLINE: [[QUBIT3:%.+]] = quantum.custom "RZ"([[CST_PI]]) [[QUBIT2]] : !quantum.bit
    // INLINE: [[QUBIT4:%.+]] = quantum.custom "RY"([[CST_PI2]]) [[QUBIT3]] : !quantum.bit
    // ALL-NOT: quantum.custom "Hadamard"
    %out_qubits_0 = quantum.custom "Hadamard"() %out_qubits : !quantum.bit

    // ALL: [[UPDATED_REG:%.+]] = quantum.insert [[REG]][ 0], [[QUBIT4]] : !quantum.reg, !quantum.bit
    %2 = quantum.insert %0[ 0], %out_qubits_0 : !quantum.reg, !quantum.bit
    %3 = quantum.compbasis qreg %2 : !quantum.obs
    %4 = quantum.probs %3 : tensor<4xf64>
    quantum.dealloc %2 : !quantum.reg
    return %4 : tensor<4xf64>
  }

  // Decomposition function should be retained for future passes
  // ALL: func.func private @Hadamard_to_RY_decomp
  func.func private @Hadamard_to_RY_decomp(%arg0: !quantum.bit) -> !quantum.bit attributes {target_gate = "Hadamard", llvm.linkage = #llvm.linkage<internal>} {
    %out_qubits_0 = quantum.custom "RZRY"() %arg0 : !quantum.bit
    return %out_qubits_0 : !quantum.bit
  }

  // Decomposition function should be retained for future passes
  // ALL: func.func private @RZRY_decomp
  func.func private @RZRY_decomp(%arg0: !quantum.bit) -> !quantum.bit attributes {target_gate = "RZRY", llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 3.1415926535897931 : f64
    %cst_0 = arith.constant 1.5707963267948966 : f64
    %out_qubits_1 = quantum.custom "RZ"(%cst) %arg0 : !quantum.bit
    %out_qubits_2 = quantum.custom "RY"(%cst_0) %out_qubits_1 : !quantum.bit
    return %out_qubits_2 : !quantum.bit
  }

  // CALL: func.func private @Hadamard_to_RY_decomp_0(%arg0: !quantum.bit)
  // CALL:   [[OQ:%.+]] = call @RZRY_decomp_1(%arg0) : (!quantum.bit) -> !quantum.bit
  // CALL:   return [[OQ]] : !quantum.bit
  // CALL: }
  // CALL: func.func private @RZRY_decomp_1(%arg0: !quantum.bit) -> !quantum.bit attributes {llvm.linkage = #llvm.linkage<internal>} {
  // CALL:   [[cst0:%.+]] = arith.constant 3.1415926535897931 : f64
  // CALL:   [[cst1:%.+]] = arith.constant 1.5707963267948966 : f64
  // CALL:   [[RZ:%.+]] = quantum.custom "RZ"([[cst0]]) %arg0 : !quantum.bit
  // CALL:   [[RY:%.+]] = quantum.custom "RY"([[cst1]]) [[RZ]] : !quantum.bit
  // CALL:   return [[RY]] : !quantum.bit
  // CALL: }
}

// -----

// Test parametric gates and wires

// ALL-LABEL: module @param_rxry
module @param_rxry {
  func.func public @test_param_rxry(%arg0: tensor<f64>, %arg1: tensor<i64>) -> tensor<2xf64> attributes {quantum.node} {
    %c0_i64 = arith.constant 0 : i64

    // ALL: [[REG:%.+]] = quantum.alloc( 1) : !quantum.reg
    %0 = quantum.alloc( 1) : !quantum.reg

    // ALL: [[WIRE:%.+]] = tensor.extract %arg1[] : tensor<i64>
    // ALL: [[PARAM:%.+]] = tensor.extract %arg0[] : tensor<f64>
    // CALL: [[FROMELEMENTS:%.+]] = tensor.from_elements [[PARAM]] : tensor<f64>
    %extracted = tensor.extract %arg1[] : tensor<i64>
    %param_0 = tensor.extract %arg0[] : tensor<f64>

    // ALL: [[QUBIT:%.+]] = quantum.extract [[REG]][[[WIRE]]] : !quantum.reg -> !quantum.bit
    %1 = quantum.extract %0[%extracted] : !quantum.reg -> !quantum.bit

    // CALL: [[QUBIT2:%.+]] = call @ParametrizedRXRY_decomp_0([[FROMELEMENTS]], [[QUBIT]]) : (tensor<f64>, !quantum.bit) -> !quantum.bit
    // INLINE: [[QUBIT1:%.+]] = quantum.custom "RX"([[PARAM]]) [[QUBIT]] : !quantum.bit
    // INLINE: [[QUBIT2:%.+]] = quantum.custom "RY"([[PARAM]]) [[QUBIT1]] : !quantum.bit
    // ALL-NOT: quantum.custom "ParametrizedRXRY"
    %out_qubits = quantum.custom "ParametrizedRXRY"(%param_0) %1 : !quantum.bit

    // ALL: [[UPDATED_REG:%.+]] = quantum.insert [[REG]][[[WIRE]]], [[QUBIT2]] : !quantum.reg, !quantum.bit
    %2 = quantum.insert %0[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    %3 = quantum.compbasis qreg %2 : !quantum.obs
    %4 = quantum.probs %3 : tensor<2xf64>
    quantum.dealloc %2 : !quantum.reg
    return %4 : tensor<2xf64>
  }

  // Decomposition function expects tensor<f64> while operation provides f64
  // ALL: func.func private @ParametrizedRXRY_decomp
  func.func private @ParametrizedRXRY_decomp(%arg0: tensor<f64>, %arg1: !quantum.bit) -> !quantum.bit
      attributes {target_gate = "ParametrizedRXRY", llvm.linkage = #llvm.linkage<internal>} {
    %extracted = tensor.extract %arg0[] : tensor<f64>
    %out_qubits = quantum.custom "RX"(%extracted) %arg1 : !quantum.bit
    %extracted_0 = tensor.extract %arg0[] : tensor<f64>
    %out_qubits_1 = quantum.custom "RY"(%extracted_0) %out_qubits : !quantum.bit
    return %out_qubits_1 : !quantum.bit
  }

  // Call mode should have a clone to the rule function
  // CALL: func.func private @ParametrizedRXRY_decomp_0
}

// -----

// Test recursive and qreg-based gate decomposition

// ALL-LABEL: module @qreg_base_circuit
module @qreg_base_circuit {
  func.func public @test_qreg_base_circuit() -> tensor<2xf64> attributes {quantum.node} {
      // INLINE-DAG: [[cmp_0:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      // INLINE-DAG: [[test_angle:%.+]] = arith.constant 1.000000e+00 : f64
      // ALL-DAG: [[index_tensor:%.+]] = arith.constant dense<0> : tensor<1xi64>
      // ALL-DAG: [[cmp_1:%.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
      // ALL: [[reg0:%.+]] = quantum.alloc( 1) : !quantum.reg
      %cst = arith.constant 1.000000e+00 : f64
      %0 = quantum.alloc( 1) : !quantum.reg

      // CALL: [[out_qreg:%.+]] = call @Test_rule_1_0([[cmp_1]], [[index_tensor]], [[reg0]]) : (tensor<f64>, tensor<1xi64>, !quantum.reg) -> !quantum.reg
      // INLINE: [[q0:%.+]] = quantum.extract [[reg0]][ 0] : !quantum.reg -> !quantum.bit
      // INLINE: [[meas:%.+]], [[q1:%.+]] = quantum.measure [[q0]] : i1, !quantum.bit
      // INLINE: [[reg1:%.+]] = quantum.insert [[reg0]][ 0], [[q1]] : !quantum.reg, !quantum.bit
      // INLINE: [[cmp:%.+]] = stablehlo.compare  NE, [[cmp_1]], [[cmp_0]],  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      // INLINE: [[cond:%.+]] = tensor.extract [[cmp]][] : tensor<i1>
      // INLINE: [[out_qreg:%.+]] = scf.if [[cond]] -> (!quantum.reg) {
      // INLINE:   [[slice0:%.+]] = stablehlo.slice [[index_tensor]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      // INLINE:   [[reshape0:%.+]] = stablehlo.reshape [[slice0]] : (tensor<1xi64>) -> tensor<i64>
      // INLINE:   [[index0:%.+]] = tensor.extract [[reshape0]][] : tensor<i64>
      // INLINE:   [[fromelements0:%.+]] = tensor.from_elements [[index0]] : tensor<1xi64>
      // INLINE:   [[slice1:%.+]] = stablehlo.slice [[fromelements0]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      // INLINE:   [[reshape1:%.+]] = stablehlo.reshape [[slice1]] : (tensor<1xi64>) -> tensor<i64>
      // INLINE:   [[index1:%.+]] = tensor.extract [[reshape1]][] : tensor<i64>
      // INLINE:   [[q2:%.+]] = quantum.extract [[reg1]][[[index1]]] : !quantum.reg -> !quantum.bit
      // INLINE:   [[q3:%.+]] = quantum.custom "RZ"([[test_angle]]) [[q2]] : !quantum.bit
      // INLINE:   [[reg2:%.+]] = quantum.insert [[reg1]][[[index1]]], [[q3]] : !quantum.reg, !quantum.bit
      // INLINE:   [[slice3:%.+]] = stablehlo.slice [[index_tensor]] [0:1] : (tensor<1xi64>) -> tensor<1xi64>
      // INLINE:   [[reshape3:%.+]] = stablehlo.reshape [[slice3]] : (tensor<1xi64>) -> tensor<i64>
      // INLINE:   [[extract6:%.+]] = tensor.extract [[reshape3]][]
      // INLINE:   [[fromelements1:%.+]] = tensor.from_elements [[extract6]]
      // INLINE:   [[slice4:%.+]] = stablehlo.slice [[fromelements1]] [0:1]
      // INLINE:   [[reshape4:%.+]] = stablehlo.reshape [[slice4]]
      // INLINE:   [[index4:%.+]] = tensor.extract [[reshape4]][]
      // INLINE:   [[q5:%.+]] = quantum.extract [[reg2]][[[index4]]] : !quantum.reg -> !quantum.bit
      // INLINE:   [[q6:%.+]] = quantum.custom "RZ"([[test_angle]]) [[q5]] : !quantum.bit
      // INLINE:   [[out:%.+]] = quantum.insert [[reg2]][[[index4]]], [[q6]] : !quantum.reg, !quantum.bit
      // INLINE:   scf.yield [[out]] : !quantum.reg
      // INLINE: } else {
      // INLINE:   scf.yield [[reg1]] : !quantum.reg
      // INLINE: }
      // ALL-NOT: quantum.custom "Test"
      %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "Test"(%cst) %1 : !quantum.bit
      %2 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit
      %3 = quantum.compbasis qreg %2 : !quantum.obs
      %4 = quantum.probs %3 : tensor<2xf64>

      // ALL: quantum.dealloc [[out_qreg]] : !quantum.reg
      quantum.dealloc %2 : !quantum.reg
      quantum.device_release
      return %4 : tensor<2xf64>
    }

    // Decomposition function should be retained for future passes
    // ALL: func.func private @Test_rule_1
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

    // Decomposition function should be retained for future passes
    // ALL: func.func private @RzDecomp_rule_1
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

  // CALL: func.func private @Test_rule_1_0
  // CALL: func.func private @RzDecomp_rule_1_1
}

// -----

// ALL-LABEL: module @multi_wire_cnot_decomposition
module @multi_wire_cnot_decomposition {
  func.func public @test_cnot_decomposition() -> tensor<4xf64> attributes {quantum.node} {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    // INLINE: [[CST_PI:%.+]] = arith.constant 3.1415926535897931 : f64
    // INLINE: [[CST_PI2:%.+]] = arith.constant 1.5707963267948966 : f64
    // ALL: [[WIRE_TENSOR:%.+]] = arith.constant dense<[0, 1]> : tensor<2xi64>
    // ALL: [[REG:%.+]] = quantum.alloc( 2) : !quantum.reg

    // CALL: [[out_qreg:%.+]] = call @CNOT_rule_cz_rz_ry_0([[WIRE_TENSOR]], [[REG]]) : (tensor<2xi64>, !quantum.reg) -> !quantum.reg
    // INLINE: [[SLICE1:%.+]] = stablehlo.slice [[WIRE_TENSOR]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    // INLINE: [[RESHAPE1:%.+]] = stablehlo.reshape [[SLICE1]] : (tensor<1xi64>) -> tensor<i64>
    // INLINE: [[SLICE2:%.+]] = stablehlo.slice [[WIRE_TENSOR]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    // INLINE: [[RESHAPE2:%.+]] = stablehlo.reshape [[SLICE2]] : (tensor<1xi64>) -> tensor<i64>
    // INLINE: [[EXTRACTED:%.+]] = tensor.extract [[RESHAPE2]][] : tensor<i64>
    // INLINE: [[QUBIT1:%.+]] = quantum.extract [[REG]][[[EXTRACTED]]] : !quantum.reg -> !quantum.bit
    // INLINE: [[RZ1:%.+]] = quantum.custom "RZ"([[CST_PI]]) [[QUBIT1]] : !quantum.bit
    // INLINE: [[RY1:%.+]] = quantum.custom "RY"([[CST_PI2]]) [[RZ1]] : !quantum.bit
    // INLINE: [[INSERT_TARGET:%.+]] = quantum.insert [[REG]][[[EXTRACTED]]], [[RY1]] : !quantum.reg, !quantum.bit
    // INLINE: [[EXTRACTED2:%.+]] = tensor.extract [[RESHAPE1]][] : tensor<i64>
    // INLINE: [[index2:%.+]] = tensor.extract [[RESHAPE2]][]
    // INLINE: [[QUBIT0:%.+]] = quantum.extract [[INSERT_TARGET]][[[EXTRACTED2]]] : !quantum.reg -> !quantum.bit
    // INLINE: [[QUBIT1_UPDATED:%.+]] = quantum.extract [[INSERT_TARGET]][[[index2]]] : !quantum.reg -> !quantum.bit
    // INLINE: [[CZ_RESULT:%.+]]:2 = quantum.custom "CZ"() [[QUBIT0]], [[QUBIT1_UPDATED]] : !quantum.bit, !quantum.bit
    // INLINE: [[INSERT2:%.+]] = quantum.insert [[INSERT_TARGET]][[[EXTRACTED2]]], [[CZ_RESULT]]#0 : !quantum.reg, !quantum.bit
    // INLINE: [[INSERT_CZ1:%.+]] = quantum.insert [[INSERT2]][[[index2]]], [[CZ_RESULT]]#1 : !quantum.reg, !quantum.bit
    // INLINE: [[index5:%.+]] = tensor.extract [[RESHAPE2]][]
    // INLINE: [[TARGET_AFTER_CZ:%.+]] = quantum.extract [[INSERT_CZ1]][[[index5]]] : !quantum.reg -> !quantum.bit
    // INLINE: [[RZ2:%.+]] = quantum.custom "RZ"([[CST_PI]]) [[TARGET_AFTER_CZ]] : !quantum.bit
    // INLINE: [[RY2:%.+]] = quantum.custom "RY"([[CST_PI2]]) [[RZ2]] : !quantum.bit
    // INLINE: [[out_qreg:%.+]] = quantum.insert [[INSERT_CZ1]][[[index5]]], [[RY2]] : !quantum.reg, !quantum.bit
    // ALL-NOT: quantum.custom "CNOT"
    %3, %4 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit
    %5 = quantum.insert %0[ 0], %3 : !quantum.reg, !quantum.bit
    %6 = quantum.insert %5[ 1], %4 : !quantum.reg, !quantum.bit

    // ALL: quantum.compbasis qreg [[out_qreg]]
    %7 = quantum.compbasis qreg %6 : !quantum.obs
    %8 = quantum.probs %7 : tensor<4xf64>
    quantum.dealloc %6 : !quantum.reg
    return %8 : tensor<4xf64>
  }

  // Decomposition function should be retained for future passes
  // ALL: func.func private @CNOT_rule_cz_rz_ry
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

  // CALL: func.func private @CNOT_rule_cz_rz_ry_0
}

// -----

// ALL-LABEL: module @cnot_alternative_decomposition
module @cnot_alternative_decomposition {
  func.func public @test_cnot_alternative_decomposition() -> tensor<4xf64> attributes {quantum.node} {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    // INLINE: [[CST_PI:%.+]] = arith.constant 3.1415926535897931 : f64
    // INLINE: [[CST_PI2:%.+]] = arith.constant 1.5707963267948966 : f64
    // ALL: [[REG:%.+]] = quantum.alloc( 2) : !quantum.reg

    // CALL: [[QUBIT1:%.+]] = quantum.extract [[REG]][ 1] : !quantum.reg -> !quantum.bit
    // CALL: [[QUBIT0:%.+]] = quantum.extract [[REG]][ 0] : !quantum.reg -> !quantum.bit
    // CALL: [[OQ:%.+]]:2 = call @CNOT_rule_h_cnot_h_0([[QUBIT1]], [[QUBIT0]]) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
    // CALL: [[FINAL_INSERT1:%.+]] = quantum.insert [[REG]][ 1], [[OQ]]#0 : !quantum.reg, !quantum.bit
    // CALL: [[FINAL_INSERT2:%.+]] = quantum.insert [[FINAL_INSERT1]][ 0], [[OQ]]#1 : !quantum.reg, !quantum.bit

    // INLINE: [[QUBIT1:%.+]] = quantum.extract [[REG]][ 1] : !quantum.reg -> !quantum.bit
    // INLINE: [[RZ1:%.+]] = quantum.custom "RZ"([[CST_PI]]) [[QUBIT1]] : !quantum.bit
    // INLINE: [[RY1:%.+]] = quantum.custom "RY"([[CST_PI2]]) [[RZ1]] : !quantum.bit
    // INLINE: [[QUBIT0:%.+]] = quantum.extract [[REG]][ 0] : !quantum.reg -> !quantum.bit
    // INLINE: [[CZ_RESULT:%.+]]:2 = quantum.custom "CZ"() [[QUBIT0]], [[RY1]] : !quantum.bit, !quantum.bit
    // INLINE: [[FINAL_INSERT1:%.+]] = quantum.insert [[REG]][ 0], [[CZ_RESULT]]#0 : !quantum.reg, !quantum.bit
    // INLINE: [[RZ2:%.+]] = quantum.custom "RZ"([[CST_PI]]) [[CZ_RESULT]]#1 : !quantum.bit
    // INLINE: [[RY2:%.+]] = quantum.custom "RY"([[CST_PI2]]) [[RZ2]] : !quantum.bit
    // INLINE: [[FINAL_INSERT2:%.+]] = quantum.insert [[FINAL_INSERT1]][ 1], [[RY2]] : !quantum.reg, !quantum.bit
    // ALL-NOT: quantum.custom "CNOT"
    %3, %4 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit
    %5 = quantum.insert %0[ 0], %3 : !quantum.reg, !quantum.bit
    %6 = quantum.insert %5[ 1], %4 : !quantum.reg, !quantum.bit

    // ALL: quantum.compbasis qreg [[FINAL_INSERT2]]
    %7 = quantum.compbasis qreg %6 : !quantum.obs
    %8 = quantum.probs %7 : tensor<4xf64>
    quantum.dealloc %6 : !quantum.reg
    return %8 : tensor<4xf64>
  }

  // Decomposition function should be retained for future passes
  // ALL: func.func private @CNOT_rule_h_cnot_h
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

  // CALL: func.func private @CNOT_rule_h_cnot_h_0(%arg0: !quantum.bit, %arg1: !quantum.bit)
  // CALL:   [[CST_PI2:%.+]] = arith.constant 1.5707963267948966 : f64
  // CALL:   [[CST_PI:%.+]] = arith.constant 3.1415926535897931 : f64
  // CALL:   [[RZ:%.+]] = quantum.custom "RZ"([[CST_PI]]) %arg0 : !quantum.bit
  // CALL:   [[RY:%.+]] = quantum.custom "RY"([[CST_PI2]]) [[RZ]] : !quantum.bit
  // CALL:   [[CZ:%.+]]:2 = quantum.custom "CZ"() %arg1, [[RY]] : !quantum.bit, !quantum.bit
  // CALL:   [[RZ:%.+]] = quantum.custom "RZ"([[CST_PI]]) [[CZ]]#1 : !quantum.bit
  // CALL:   [[RY:%.+]] = quantum.custom "RY"([[CST_PI2]]) [[RZ]] : !quantum.bit
  // CALL:   return [[RY]], [[CZ]]#0 : !quantum.bit, !quantum.bit
  // CALL: }
}

// -----

// ALL-LABEL: module @mcm_example
module @mcm_example {
  func.func public @test_mcm_hadamard() -> tensor<2xf64> attributes {quantum.node} {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %mres, %out_qubit = quantum.measure %1 : i1, !quantum.bit
    %2 = quantum.insert %0[ 0], %out_qubit : !quantum.reg, !quantum.bit

    // CALL: call @rz_ry_0
    // INLINE: quantum.custom "RZ"
    // INLINE: quantum.custom "RY"

    // ALL-NOT: quantum.custom "Hadamard"
    %3 = quantum.extract %2[ 0] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Hadamard"() %3 : !quantum.bit
    %4 = quantum.insert %2[ 0], %out_qubits : !quantum.reg, !quantum.bit

    %5 = quantum.compbasis qreg %4 : !quantum.obs
    %6 = quantum.probs %5 : tensor<2xf64>
    quantum.dealloc %4 : !quantum.reg
    return %6 : tensor<2xf64>
  }

  // Decomposition function should be retained for future passes
  // ALL: func.func private @rz_ry
  func.func private @rz_ry(%arg0: !quantum.reg, %arg1: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "Hadamard"} {
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

  // CALL: func.func private @rz_ry_0
}

// -----

// ALL-LABEL: module @circuit_with_multirz
module @circuit_with_multirz {
  func.func public @test_with_multirz() -> tensor<4xf64> attributes {quantum.node} {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // INLINE-DAG: [[CST_PI2:%.+]] = arith.constant 1.5707963267948966 : f64
    // INLINE-DAG: [[CST_PI:%.+]] = arith.constant 3.1415926535897931 : f64
    // INLINE-DAG: [[CST_RZ:%.+]] = arith.constant 5.000000e-01 : f64
    // CALL-DAG: [[WIRE:%.+]] = arith.constant dense<0> : tensor<1xi64>
    // CALL-DAG: [[PARAM:%.+]] = arith.constant dense<5.000000e-01> : tensor<1xf64>
    // ALL: [[REG:%.+]] = quantum.alloc( 2) : !quantum.reg

    // CALL: [[REG_RZ:%.+]] = call @_multi_rz_decomposition_wires_1_1([[PARAM]], [[WIRE]], [[REG]]) : (tensor<1xf64>, tensor<1xi64>, !quantum.reg) -> !quantum.reg
    // INLINE: [[QUBIT1:%.+]] = quantum.custom "RZ"([[CST_RZ]]) {{%.+}} : !quantum.bit
    // INLINE: [[REG_RZ:%.+]] = quantum.insert [[REG]][{{%.+}}], [[QUBIT1]] : !quantum.reg, !quantum.bit
    // ALL-NOT: quantum.multirz
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %extracted_2 = tensor.extract %cst[] : tensor<f64>
    %out_qubits = quantum.multirz(%extracted_2) %1 : !quantum.bit

    // CALL: [[QUBIT:%.+]] = quantum.extract [[REG_RZ]][ 0] : !quantum.reg -> !quantum.bit
    // CALL: [[QUBIT4:%.+]] = call @Hadamard_to_RY_decomp_0([[QUBIT]]) : (!quantum.bit) -> !quantum.bit
    // INLINE: [[QUBIT3:%.+]] = quantum.custom "RZ"([[CST_PI]]) {{%.+}} : !quantum.bit
    // INLINE: [[QUBIT4:%.+]] = quantum.custom "RY"([[CST_PI2]]) [[QUBIT3]] : !quantum.bit
    // ALL-NOT: quantum.custom "Hadamard"
    %out_qubits_0 = quantum.custom "Hadamard"() %out_qubits : !quantum.bit

    // ALL: [[UPDATED_REG:%.+]] = quantum.insert [[REG_RZ]][ 0], [[QUBIT4]] : !quantum.reg, !quantum.bit
    %2 = quantum.insert %0[ 0], %out_qubits_0 : !quantum.reg, !quantum.bit
    %3 = quantum.compbasis qreg %2 : !quantum.obs
    %4 = quantum.probs %3 : tensor<4xf64>
    quantum.dealloc %2 : !quantum.reg
    return %4 : tensor<4xf64>
  }

  // ALL: func.func private @Hadamard_to_RY_decomp
  func.func private @Hadamard_to_RY_decomp(%arg0: !quantum.bit) -> !quantum.bit attributes {target_gate = "Hadamard", llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 3.1415926535897931 : f64
    %cst_0 = arith.constant 1.5707963267948966 : f64
    %out_qubits = quantum.custom "RZ"(%cst) %arg0 : !quantum.bit
    %out_qubits_1 = quantum.custom "RY"(%cst_0) %out_qubits : !quantum.bit
    return %out_qubits_1 : !quantum.bit
  }

  // ALL: func.func private @_multi_rz_decomposition_wires_1
  func.func private @_multi_rz_decomposition_wires_1(%arg0: !quantum.reg, %arg1: tensor<1xf64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "MultiRZ"} {
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

  // CALL: func.func private @Hadamard_to_RY_decomp_0
  // CALL: func.func private @_multi_rz_decomposition_wires_1_1
}

// -----

// ALL-LABEL: module @circuit_with_operator_op
module @circuit_with_operator_op {
  func.func public @test_with_operator(%arg0: f64) attributes {quantum.node} {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // CALL: call @_my_dummy_decomp_0
    // INLINE: quantum.custom "RZ"
    // ALL-NOT: quantum.operator
    %out_qubits_0 = quantum.operator "DummyOp"(%arg0: f64) qubits(%1) static_data = {metadata = "word"} param_map = {arg = [0]} qubit_map = {wires = [0]}
    %2 = quantum.insert %0[ 0], %out_qubits_0 : !quantum.reg, !quantum.bit
    quantum.dealloc %2 : !quantum.reg
    return
  }

  // ALL-LABEL: func.func private @_my_dummy_decomp
  func.func private @_my_dummy_decomp(%arg0: !quantum.reg, %arg1: tensor<1xf64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes
      {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "DummyOp{arg:[f64]}{wires:1}{metadata = \22word\22}"} {
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

  // CALL: func.func private @_my_dummy_decomp_0
}

// -----

// ALL-LABEL: module @qreg_at_not_first_arg
module @qreg_at_not_first_arg {
  func.func public @test_qreg_at_not_first_arg() attributes {quantum.node} {
    // ALL: [[wire_tensor:%.+]] = arith.constant dense<[0, 1]> : tensor<2xi64>
    // ALL: [[reg:%.+]] = quantum.alloc( 2) : !quantum.reg

    // CALL: [[out_qreg:%.+]] = call @my_cnot_0([[wire_tensor]], [[reg]]) : (tensor<2xi64>, !quantum.reg) -> !quantum.reg
    // INLINE: [[zero:%.+]] = stablehlo.slice [[wire_tensor]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    // INLINE: [[zero_i64:%.+]] = stablehlo.reshape [[zero]] : (tensor<1xi64>) -> tensor<i64>
    // INLINE: [[one:%.+]] = stablehlo.slice [[wire_tensor]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    // INLINE: [[one_i64:%.+]] = stablehlo.reshape [[one]] : (tensor<1xi64>) -> tensor<i64>
    // INLINE: [[zero:%.+]] = tensor.extract [[zero_i64]][] : tensor<i64>
    // INLINE: [[one:%.+]] = tensor.extract [[one_i64]][] : tensor<i64>
    // INLINE: [[q0:%.+]] = quantum.extract [[reg]][[[zero]]] : !quantum.reg -> !quantum.bit
    // INLINE: [[q1:%.+]] = quantum.extract [[reg]][[[one]]] : !quantum.reg -> !quantum.bit
    // INLINE: [[out_qubits:%.+]]:2 = quantum.custom "CZ"() [[q0]], [[q1]] : !quantum.bit, !quantum.bit
    // INLINE: [[insert0:%.+]] = quantum.insert [[reg]][[[zero]]], [[out_qubits]]#0 : !quantum.reg, !quantum.bit
    // INLINE: [[out_qreg:%.+]] = quantum.insert [[insert0]][[[one]]], [[out_qubits]]#1 : !quantum.reg, !quantum.bit
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %out_qubits_0:2 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit
    %3 = quantum.insert %0[ 0], %out_qubits_0#0 : !quantum.reg, !quantum.bit
    %4 = quantum.insert %3[ 1], %out_qubits_0#1 : !quantum.reg, !quantum.bit

    // ALL: quantum.dealloc [[out_qreg]] : !quantum.reg
    quantum.dealloc %4 : !quantum.reg
    return
  }

  // ALL: func.func private @my_cnot
  func.func private @my_cnot(%arg0: tensor<2xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {target_gate = "CNOT"} {
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %2 = stablehlo.slice %arg0 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
    %extracted_0 = tensor.extract %1[] : tensor<i64>
    %extracted_1 = tensor.extract %3[] : tensor<i64>
    %4 = quantum.extract %arg1[%extracted_0] : !quantum.reg -> !quantum.bit
    %5 = quantum.extract %arg1[%extracted_1] : !quantum.reg -> !quantum.bit
    %out_qubits:2 = quantum.custom "CZ"() %4, %5 : !quantum.bit, !quantum.bit
    %6 = quantum.insert %arg1[%extracted_0], %out_qubits#0 : !quantum.reg, !quantum.bit
    %7 = quantum.insert %6[%extracted_1], %out_qubits#1 : !quantum.reg, !quantum.bit
    return %7 : !quantum.reg
  }

  // CALL: func.func private @my_cnot_0
}

// -----

// ALL-LABEL: module @test_paulirot
module @test_paulirot {
    func.func @test() attributes {quantum.node} {
        %pi = arith.constant 3.1 : f64
        %inreg = quantum.alloc( 3) : !quantum.reg
        %q0 = quantum.extract %inreg[ 0] : !quantum.reg -> !quantum.bit
        %q1 = quantum.extract %inreg[ 1] : !quantum.reg -> !quantum.bit
        %q2 = quantum.extract %inreg[ 2] : !quantum.reg -> !quantum.bit
        // ALL-NOT: quantum.paulirot
        %out:3 = quantum.paulirot ["Z", "X", "Y"](%pi) %q0, %q1, %q2 : !quantum.bit, !quantum.bit, !quantum.bit

        // CALL: call @my_paulirot_decomp_0
        // INLINE-DAG: Hadamard
        // INLINE-DAG: RX
        // INLINE: multirz
        // INLINE-DAG: Hadamard
        // INLINE-DAG: RX
        %reg2 = quantum.insert %inreg[ 0], %out#0 : !quantum.reg, !quantum.bit
        %reg3 = quantum.insert %reg2[ 1], %out#1 : !quantum.reg, !quantum.bit
        %reg4 = quantum.insert %reg3[ 2], %out#2 : !quantum.reg, !quantum.bit
        quantum.dealloc %reg4 : !quantum.reg
        return
    }

    // ALL: my_paulirot_decomp
    func.func private @my_paulirot_decomp(%inreg : !quantum.reg, %angle_tensor : tensor<f64>, %q_tensor : tensor<3xi64>) -> !quantum.reg attributes {target_gate = "PauliRot{theta:[f64]}{wires:3}{pauli_word = \22ZXY\22}"} {
        %pi_by_2 = arith.constant 1.57 : f64
        %m_pi_by_2 = arith.constant -1.57 : f64
        %angle = tensor.extract %angle_tensor[] : tensor<f64>

        %q0_slice = stablehlo.slice %q_tensor [0:1] : (tensor<3xi64>) -> tensor<1xi64>
        %q1_slice = stablehlo.slice %q_tensor [1:2] : (tensor<3xi64>) -> tensor<1xi64>
        %q2_slice = stablehlo.slice %q_tensor [2:3] : (tensor<3xi64>) -> tensor<1xi64>

        %q0_tensor = stablehlo.reshape %q0_slice : (tensor<1xi64>) -> tensor<i64>
        %q1_tensor = stablehlo.reshape %q1_slice : (tensor<1xi64>) -> tensor<i64>
        %q2_tensor = stablehlo.reshape %q2_slice : (tensor<1xi64>) -> tensor<i64>

        %q0_index = tensor.extract %q0_tensor[] : tensor<i64>
        %q1_index = tensor.extract %q1_tensor[] : tensor<i64>
        %q2_index = tensor.extract %q2_tensor[] : tensor<i64>

        %q0 = quantum.extract %inreg[%q0_index] : !quantum.reg -> !quantum.bit
        %q1 = quantum.extract %inreg[%q1_index] : !quantum.reg -> !quantum.bit
        %q2 = quantum.extract %inreg[%q2_index] : !quantum.reg -> !quantum.bit

        %h1_out = quantum.custom "Hadamard"() %q0 : !quantum.bit
        %rx1_out = quantum.custom "RX"(%pi_by_2) %q1 : !quantum.bit
        %mrz_out:3 = quantum.multirz(%angle) %h1_out, %rx1_out, %q2 : !quantum.bit, !quantum.bit, !quantum.bit

        %h2_out = quantum.custom "Hadamard"() %mrz_out#0 : !quantum.bit
        %rx2_out = quantum.custom "RX"(%m_pi_by_2) %mrz_out#1 : !quantum.bit

        %reg2 = quantum.insert %inreg[0], %h2_out : !quantum.reg, !quantum.bit
        %reg3 = quantum.insert %reg2[1], %rx2_out : !quantum.reg, !quantum.bit
        %outreg = quantum.insert %reg3[2], %mrz_out#2 : !quantum.reg, !quantum.bit

        return %outreg : !quantum.reg
    }

    // CALL: func.func private @my_paulirot_decomp_0
}

// -----

// ALL-LABEL: module @null_decomp_rule
module @null_decomp_rule{
  func.func public @test_null_decomp_rule() attributes {quantum.node} {
    // ALL: [[reg:%.+]] = quantum.alloc( 1)
    // ALL: [[q0:%.+]] = quantum.extract [[reg]][ 0]
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit

    // ALL-NOT: PauliX
    // ALL: quantum.custom "Hadamard"() [[q0]] : !quantum.bit
    %2 = quantum.custom "PauliX"() %1 : !quantum.bit
    %3 = quantum.custom "Hadamard"() %2 : !quantum.bit
    %4 = quantum.insert %0[ 0], %3 : !quantum.reg, !quantum.bit
    quantum.dealloc %4 : !quantum.reg
    return
  }

  // ALL: func.func private @null_decomp
  func.func private @null_decomp() attributes {target_gate = "PauliX"} {
    return
  }
}

// -----

// ALL-LABEL: module @different_qreg_values
module @different_qreg_values{
  func.func public @circuit() attributes {quantum.node} {
    // ALL: [[wire_tensor:%.+]] = arith.constant dense<[2, 1]> : tensor<2xi64>
    // ALL: [[reg:%.+]] = quantum.alloc( 3) : !quantum.reg
    // ALL: [[q0:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    // ALL: [[H:%.+]] = quantum.custom "Hadamard"() [[q0]] : !quantum.bit
    // ALL: [[H_insert:%.+]] = quantum.insert [[reg]][ 0], [[H]] : !quantum.reg, !quantum.bit
    %0 = quantum.alloc( 3) : !quantum.reg
    %1 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Hadamard"() %2 : !quantum.bit
    %3 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit

    // CALL: [[out_qreg:%.+]] = call @my_cnot_0([[wire_tensor]], [[H_insert]]) : (tensor<2xi64>, !quantum.reg) -> !quantum.reg
    // INLINE: [[two:%.+]] = stablehlo.slice [[wire_tensor]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    // INLINE: [[two_i64:%.+]] = stablehlo.reshape [[two]] : (tensor<1xi64>) -> tensor<i64>
    // INLINE: [[one:%.+]] = stablehlo.slice [[wire_tensor]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    // INLINE: [[one_i64:%.+]] = stablehlo.reshape [[one]] : (tensor<1xi64>) -> tensor<i64>
    // INLINE: [[two:%.+]] = tensor.extract [[two_i64]][] : tensor<i64>
    // INLINE: [[one:%.+]] = tensor.extract [[one_i64]][] : tensor<i64>
    // INLINE: [[q2:%.+]] = quantum.extract [[H_insert]][[[two]]] : !quantum.reg -> !quantum.bit
    // INLINE: [[q1:%.+]] = quantum.extract [[H_insert]][[[one]]] : !quantum.reg -> !quantum.bit
    // INLINE: [[CZ:%.+]]:2 = quantum.custom "CZ"() [[q2]], [[q1]] : !quantum.bit, !quantum.bit
    // INLINE: [[insert2:%.+]] = quantum.insert [[H_insert]][[[two]]], [[CZ]]#0 : !quantum.reg, !quantum.bit
    // INLINE: [[out_qreg:%.+]] = quantum.insert [[insert2]][[[one]]], [[CZ]]#1 : !quantum.reg, !quantum.bit
    %4 = quantum.extract %3[ 2] : !quantum.reg -> !quantum.bit
    %out_qubits_0:2 = quantum.custom "CNOT"() %4, %1 : !quantum.bit, !quantum.bit
    %5 = quantum.insert %3[ 2], %out_qubits_0#0 : !quantum.reg, !quantum.bit
    %6 = quantum.insert %5[ 1], %out_qubits_0#1 : !quantum.reg, !quantum.bit

    // ALL: quantum.dealloc [[out_qreg]] : !quantum.reg
    quantum.dealloc %6 : !quantum.reg
    return
  }

  // ALL: func.func private @my_cnot
  func.func private @my_cnot(%arg0: tensor<2xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {target_gate = "CNOT"} {
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %2 = stablehlo.slice %arg0 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
    %extracted_0 = tensor.extract %1[] : tensor<i64>
    %extracted_1 = tensor.extract %3[] : tensor<i64>
    %4 = quantum.extract %arg1[%extracted_0] : !quantum.reg -> !quantum.bit
    %5 = quantum.extract %arg1[%extracted_1] : !quantum.reg -> !quantum.bit
    %out_qubits:2 = quantum.custom "CZ"() %4, %5 : !quantum.bit, !quantum.bit
    %6 = quantum.insert %arg1[%extracted_0], %out_qubits#0 : !quantum.reg, !quantum.bit
    %7 = quantum.insert %6[%extracted_1], %out_qubits#1 : !quantum.reg, !quantum.bit
    return %7 : !quantum.reg
  }

  // CALL: func.func private @my_cnot_0
}

// -----

// ALL-LABEL: module @test_if
module @test_if {
  func.func @circuit() {
    %reg = quantum.alloc( 2) : !quantum.reg
    %in = quantum.extract %reg[0]  : !quantum.reg -> !quantum.bit

    %init_index = arith.constant 1 : index
    %true = arith.constant 1 : i1
    %limit = arith.constant 10 : index

    %out = scf.if %true -> !quantum.bit {
      // ALL-NOT: "T"
      // CALL: call @"__builtin__t_phaseshift_T{}{wires:1}{}_0"
      // INLINE: "PhaseShift"
      %if_out = quantum.custom "T"() %in : !quantum.bit
      scf.yield %if_out : !quantum.bit
    } else {
      scf.yield %in : !quantum.bit
    }

    // ALL-NOT: "T"
    // CALL: call @"__builtin__t_phaseshift_T{}{wires:1}{}_0"
    // INLINE: "PhaseShift"
    %post_out = quantum.custom "T"() %out : !quantum.bit
    %out_qreg = quantum.insert %reg[0], %post_out : !quantum.reg, !quantum.bit
    quantum.dealloc %out_qreg : !quantum.reg

    return
  }

  // ALL-LABEL: func.func private @"__builtin__t_phaseshift_T{}{wires:1}{}"
  func.func private @"__builtin__t_phaseshift_T{}{wires:1}{}"(%arg0: tensor<1xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PhaseShift{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "T{}{wires:1}{}"} {
    %cst = arith.constant 0.78539816339744828 : f64
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = quantum.extract %arg1[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "PhaseShift"(%cst) %2 : !quantum.bit
    %3 = quantum.insert %arg1[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %3 : !quantum.reg
  }

  // CALL: func.func private @"__builtin__t_phaseshift_T{}{wires:1}{}_0"
}

// -----

// ALL-LABEL: module @test_for_loop
module @test_for_loop {
  func.func @circuit() {
    %reg = quantum.alloc( 2) : !quantum.reg
    %in = quantum.extract %reg[0]  : !quantum.reg -> !quantum.bit

    %cond = arith.constant 1 : i1

    %start = arith.constant 0 : index
    %end = arith.constant 10 : index
    %step = arith.constant 1 : index

    %rout = scf.for %iter = %start to %end step %step iter_args(%for_in = %in) -> (!quantum.bit) {
      // ALL-NOT: "T"
      // CALL: call @"__builtin__t_phaseshift_T{}{wires:1}{}_0"
      // INLINE: "PhaseShift"
      %out = quantum.custom "T"() %for_in : !quantum.bit
      scf.yield %out : !quantum.bit
    }

    // ALL-NOT: "T"
    // CALL: call @"__builtin__t_phaseshift_T{}{wires:1}{}_0"
    // INLINE: "PhaseShift"
    %post_out = quantum.custom "T"() %rout : !quantum.bit
    %out_qreg = quantum.insert %reg[0], %post_out : !quantum.reg, !quantum.bit
    quantum.dealloc %out_qreg : !quantum.reg

    return
  }

  // ALL-LABEL: func.func private @"__builtin__t_phaseshift_T{}{wires:1}{}"
  func.func private @"__builtin__t_phaseshift_T{}{wires:1}{}"(%arg0: tensor<1xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>,  target_gate = "T{}{wires:1}{}"} {
    %cst = arith.constant 0.78539816339744828 : f64
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = quantum.extract %arg1[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "PhaseShift"(%cst) %2 : !quantum.bit
    %3 = quantum.insert %arg1[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %3 : !quantum.reg
  }

  // CALL: func.func private @"__builtin__t_phaseshift_T{}{wires:1}{}_0"
}

// -----

// ALL-LABEL: module @test_while_loop
module @test_while_loop {
  func.func @circuit() {
    %reg = quantum.alloc( 2) : !quantum.reg
    %in = quantum.extract %reg[0]  : !quantum.reg -> !quantum.bit

    %init_index = arith.constant 1 : index
    %true = arith.constant 1 : i1
    %limit = arith.constant 10 : index

    %out_index, %rout = scf.while (%before_index = %init_index, %before_qubit = %in) : (index, !quantum.bit) -> (index, !quantum.bit) {
      // ALL-NOT: "T"
      // CALL: call @"__builtin__t_phaseshift_T{}{wires:1}{}_0"
      // INLINE: "PhaseShift"
      %out = quantum.custom "T"() %before_qubit: !quantum.bit
      %increment = arith.constant 1 : index
      %updated_index = index.add %before_index, %increment
      %condition = index.cmp ult (%updated_index, %limit)
      scf.condition(%condition) %updated_index, %out: index, !quantum.bit
    } do {
      ^bb0(%after_index: index, %after_qubit : !quantum.bit):
        // ALL-NOT: "T"
        // CALL: call @"__builtin__t_phaseshift_T{}{wires:1}{}_0"
        // INLINE: "PhaseShift"
        %after_out = quantum.custom "T"() %after_qubit : !quantum.bit
        scf.yield %after_index, %after_out: index, !quantum.bit
    }

    // ALL-NOT: "T"
    // CALL: call @"__builtin__t_phaseshift_T{}{wires:1}{}_0"
    // INLINE: "PhaseShift"
    %post_out = quantum.custom "T"() %rout : !quantum.bit
    %out_qreg = quantum.insert %reg[0], %post_out : !quantum.reg, !quantum.bit
    quantum.dealloc %out_qreg : !quantum.reg

    return
  }

  // ALL-LABEL: func.func private @"__builtin__t_phaseshift_T{}{wires:1}{}"
  func.func private @"__builtin__t_phaseshift_T{}{wires:1}{}"(%arg0: tensor<1xi64>, %arg1: !quantum.reg) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, resources = {operations = {"PhaseShift{0:[f64]}{wires:1}{}" = 1 : i64}}, target_gate = "T{}{wires:1}{}"} {
    %cst = arith.constant 0.78539816339744828 : f64
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = quantum.extract %arg1[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "PhaseShift"(%cst) %2 : !quantum.bit
    %3 = quantum.insert %arg1[%extracted], %out_qubits : !quantum.reg, !quantum.bit
    return %3 : !quantum.reg
  }

  // CALL: func.func private @"__builtin__t_phaseshift_T{}{wires:1}{}_0"
}
