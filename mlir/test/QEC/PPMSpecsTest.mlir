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

// RUN: quantum-opt --ppm-specs --split-input-file -verify-diagnostics %s | FileCheck %s

//CHECK: {
//CHECK:     "test_no_ppr_ppm": {
//CHECK:         "num_logical_qubits": 2
//CHECK:     }
//CHECK: }
func.func public @test_no_ppr_ppm() {
    %c0_i64 = arith.constant 0 : i64
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "S"() %1 : !quantum.bit
    %2 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %out_qubits_0 = quantum.custom "Hadamard"() %2 : !quantum.bit
    %out_qubits_1 = quantum.custom "T"() %out_qubits_0 : !quantum.bit
    %out_qubits_2:2 = quantum.custom "CNOT"() %out_qubits_1, %out_qubits : !quantum.bit, !quantum.bit
    %3 = quantum.insert %0[ 0], %out_qubits_2#0 : !quantum.reg, !quantum.bit
    %4 = quantum.insert %3[ 1], %out_qubits_2#1 : !quantum.reg, !quantum.bit
    quantum.dealloc %4 : !quantum.reg
    quantum.device_release
    return
}

// -----

//CHECK: {
//CHECK:     "test_to_ppr": {
//CHECK:         "max_weight_pi4": 2,
//CHECK:         "max_weight_pi8": 1,
//CHECK:         "num_logical_qubits": 2,
//CHECK:         "num_pi4_gates": 7,
//CHECK:         "num_pi8_gates": 1
//CHECK:     }
//CHECK: }
func.func public @test_to_ppr() {
    %c0_i64 = arith.constant 0 : i64
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %2 = qec.ppr ["Z"](4) %1 : !quantum.bit
    %3 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %4 = qec.ppr ["Z"](4) %3 : !quantum.bit
    %5 = qec.ppr ["X"](4) %4 : !quantum.bit
    %6 = qec.ppr ["Z"](4) %5 : !quantum.bit
    %7 = qec.ppr ["Z"](8) %6 : !quantum.bit
    %8:2 = qec.ppr ["Z", "X"](4) %7, %2 : !quantum.bit, !quantum.bit
    %9 = qec.ppr ["Z"](-4) %8#0 : !quantum.bit
    %10 = qec.ppr ["X"](-4) %8#1 : !quantum.bit
    %11 = quantum.insert %0[ 0], %9 : !quantum.reg, !quantum.bit
    %12 = quantum.insert %11[ 1], %10 : !quantum.reg, !quantum.bit
    quantum.dealloc %12 : !quantum.reg
    quantum.device_release
    return
}

// -----

//CHECK: {
//CHECK:     "test_commute_ppr": {
//CHECK:         "max_weight_pi4": 2,
//CHECK:         "max_weight_pi8": 1,
//CHECK:         "num_logical_qubits": 2,
//CHECK:         "num_of_ppm": 2,
//CHECK:         "num_pi4_gates": 7,
//CHECK:         "num_pi8_gates": 1
//CHECK:     }
//CHECK: }
func.func public @test_commute_ppr() {
    %c0_i64 = arith.constant 0 : i64
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %2 = qec.ppr ["Z"](4) %1 : !quantum.bit
    %3 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %4 = qec.ppr ["X"](8) %3 : !quantum.bit
    quantum.device_release
    %5 = qec.ppr ["Z"](4) %4 : !quantum.bit
    %6 = qec.ppr ["X"](4) %5 : !quantum.bit
    %7 = qec.ppr ["Z"](4) %6 : !quantum.bit
    %8:2 = qec.ppr ["Z", "X"](4) %7, %2 : !quantum.bit, !quantum.bit
    %9 = qec.ppr ["Z"](-4) %8#0 : !quantum.bit
    %10 = qec.ppr ["X"](-4) %8#1 : !quantum.bit
    %mres, %out_qubits = qec.ppm ["Z"] %9 : !quantum.bit
    %from_elements = tensor.from_elements %mres : tensor<i1>
    %mres_0, %out_qubits_1 = qec.ppm ["Z"] %10 : !quantum.bit
    %from_elements_2 = tensor.from_elements %mres_0 : tensor<i1>
    %11 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit
    %12 = quantum.insert %11[ 1], %out_qubits_1 : !quantum.reg, !quantum.bit
    quantum.dealloc %12 : !quantum.reg
    return
}

// -----

//CHECK: {
//CHECK:     "test_merge_ppr_ppm": {
//CHECK:         "num_logical_qubits": 2,
//CHECK:         "num_of_ppm": 2
//CHECK:     }
//CHECK: }
func.func public @test_merge_ppr_ppm() {
    %c0_i64 = arith.constant 0 : i64
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %mres, %out_qubits:2 = qec.ppm ["Z", "X"] %2, %1 : !quantum.bit, !quantum.bit
    %from_elements = tensor.from_elements %mres : tensor<i1>
    quantum.device_release
    %mres_0, %out_qubits_1 = qec.ppm ["X"] %out_qubits#1 : !quantum.bit
    %from_elements_2 = tensor.from_elements %mres_0 : tensor<i1>
    %3 = quantum.insert %0[ 0], %out_qubits_1 : !quantum.reg, !quantum.bit
    %4 = quantum.insert %3[ 1], %out_qubits#0 : !quantum.reg, !quantum.bit
    quantum.dealloc %4 : !quantum.reg
    return
}

// -----

//CHECK: {
//CHECK:     "test_ppr_to_ppm": {
//CHECK:         "max_weight_pi2": 2,
//CHECK:         "num_logical_qubits": 2,
//CHECK:         "num_of_ppm": 19,
//CHECK:         "num_pi2_gates": 8
//CHECK:     }
//CHECK: }
func.func public @test_ppr_to_ppm() {
    %c0_i64 = arith.constant 0 : i64
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %2 = quantum.alloc_qb : !quantum.bit
    %mres, %out_qubits:2 = qec.ppm ["Z", "Y"] %1, %2 : !quantum.bit, !quantum.bit
    %mres_0, %out_qubits_1 = qec.ppm ["X"] %out_qubits#1 : !quantum.bit
    %3 = arith.xori %mres, %mres_0 : i1
    %4 = qec.ppr ["Z"](2) %out_qubits#0 cond(%3) : !quantum.bit
    quantum.dealloc_qb %out_qubits_1 : !quantum.bit
    %5 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %6 = quantum.alloc_qb : !quantum.bit
    %mres_2, %out_qubits_3:2 = qec.ppm ["Z", "Y"] %5, %6 : !quantum.bit, !quantum.bit
    %mres_4, %out_qubits_5 = qec.ppm ["X"] %out_qubits_3#1 : !quantum.bit
    %7 = arith.xori %mres_2, %mres_4 : i1
    %8 = qec.ppr ["Z"](2) %out_qubits_3#0 cond(%7) : !quantum.bit
    quantum.dealloc_qb %out_qubits_5 : !quantum.bit
    %9 = quantum.alloc_qb : !quantum.bit
    %mres_6, %out_qubits_7:2 = qec.ppm ["X", "Y"] %8, %9 : !quantum.bit, !quantum.bit
    %mres_8, %out_qubits_9 = qec.ppm ["X"] %out_qubits_7#1 : !quantum.bit
    %10 = arith.xori %mres_6, %mres_8 : i1
    %11 = qec.ppr ["X"](2) %out_qubits_7#0 cond(%10) : !quantum.bit
    quantum.dealloc_qb %out_qubits_9 : !quantum.bit
    %12 = quantum.alloc_qb : !quantum.bit
    %mres_10, %out_qubits_11:2 = qec.ppm ["Z", "Y"] %11, %12 : !quantum.bit, !quantum.bit
    %mres_12, %out_qubits_13 = qec.ppm ["X"] %out_qubits_11#1 : !quantum.bit
    %13 = arith.xori %mres_10, %mres_12 : i1
    %14 = qec.ppr ["Z"](2) %out_qubits_11#0 cond(%13) : !quantum.bit
    quantum.dealloc_qb %out_qubits_13 : !quantum.bit
    %15 = quantum.alloc_qb : !quantum.bit
    %16 = qec.fabricate  magic : !quantum.bit
    %mres_14, %out_qubits_15:2 = qec.ppm ["Z", "Z"] %14, %16 : !quantum.bit, !quantum.bit
    %mres_16, %out_qubits_17:2 = qec.ppm ["Z", "Y"] %out_qubits_15#1, %15 : !quantum.bit, !quantum.bit
    %mres_18, %out_qubits_19 = qec.ppm ["X"] %out_qubits_17#0 : !quantum.bit
    %mres_20, %out_qubits_21 = qec.select.ppm(%mres_14, ["X"], ["Z"]) %out_qubits_17#1 : !quantum.bit
    %17 = arith.xori %mres_16, %mres_18 : i1
    %18 = qec.ppr ["Z"](2) %out_qubits_15#0 cond(%17) : !quantum.bit
    quantum.dealloc_qb %out_qubits_21 : !quantum.bit
    quantum.dealloc_qb %out_qubits_19 : !quantum.bit
    %19 = quantum.alloc_qb : !quantum.bit
    %mres_22, %out_qubits_23:3 = qec.ppm ["Z", "X", "Y"] %18, %4, %19 : !quantum.bit, !quantum.bit, !quantum.bit
    %mres_24, %out_qubits_25 = qec.ppm ["X"] %out_qubits_23#2 : !quantum.bit
    %20 = arith.xori %mres_22, %mres_24 : i1
    %21:2 = qec.ppr ["Z", "X"](2) %out_qubits_23#0, %out_qubits_23#1 cond(%20) : !quantum.bit, !quantum.bit
    quantum.dealloc_qb %out_qubits_25 : !quantum.bit
    %22 = quantum.alloc_qb : !quantum.bit
    %mres_26, %out_qubits_27:2 = qec.ppm ["Z", "Y"] %21#0, %22 : !quantum.bit, !quantum.bit
    %mres_28, %out_qubits_29 = qec.ppm ["X"] %out_qubits_27#1 : !quantum.bit
    %23 = arith.xori %mres_26, %mres_28 : i1
    %24 = qec.ppr ["Z"](2) %out_qubits_27#0 cond(%23) : !quantum.bit
    quantum.dealloc_qb %out_qubits_29 : !quantum.bit
    %25 = quantum.alloc_qb : !quantum.bit
    %mres_30, %out_qubits_31:2 = qec.ppm ["X", "Y"] %21#1, %25 : !quantum.bit, !quantum.bit
    %mres_32, %out_qubits_33 = qec.ppm ["X"] %out_qubits_31#1 : !quantum.bit
    %26 = arith.xori %mres_30, %mres_32 : i1
    %27 = qec.ppr ["X"](2) %out_qubits_31#0 cond(%26) : !quantum.bit
    quantum.dealloc_qb %out_qubits_33 : !quantum.bit
    %mres_34, %out_qubits_35 = qec.ppm ["Z"] %24 : !quantum.bit
    %from_elements = tensor.from_elements %mres_34 : tensor<i1>
    %mres_36, %out_qubits_37 = qec.ppm ["Z"] %27 : !quantum.bit
    %from_elements_38 = tensor.from_elements %mres_36 : tensor<i1>
    %28 = quantum.insert %0[ 0], %out_qubits_35 : !quantum.reg, !quantum.bit
    %29 = quantum.insert %28[ 1], %out_qubits_37 : !quantum.reg, !quantum.bit
    quantum.dealloc %29 : !quantum.reg
    quantum.device_release
    return
}

// -----

//CHECK: {
//CHECK:     "test_ppm_compilation_1": {
//CHECK:         "max_weight_pi2": 2,
//CHECK:         "num_logical_qubits": 2,
//CHECK:         "num_of_ppm": 7,
//CHECK:         "num_pi2_gates": 2
//CHECK:     },
//CHECK:     "test_ppm_compilation_2": {
//CHECK:         "num_logical_qubits": 2
//CHECK:     }
//CHECK: }
func.func public @test_ppm_compilation_1() {
    %c0_i64 = arith.constant 0 : i64
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit
    %4 = quantum.extract %0[ 3] : !quantum.reg -> !quantum.bit
    %5 = quantum.extract %0[ 4] : !quantum.reg -> !quantum.bit
    %6 = quantum.extract %0[ 5] : !quantum.reg -> !quantum.bit
    %7 = quantum.alloc_qb : !quantum.bit
    %8 = qec.fabricate  magic : !quantum.bit
    %mres, %out_qubits:2 = qec.ppm ["X", "Z"] %1, %8 : !quantum.bit, !quantum.bit
    %mres_0, %out_qubits_1:2 = qec.ppm ["Z", "Y"] %out_qubits#1, %7 : !quantum.bit, !quantum.bit
    %mres_2, %out_qubits_3 = qec.ppm ["X"] %out_qubits_1#0 : !quantum.bit
    %mres_4, %out_qubits_5 = qec.select.ppm(%mres, ["X"], ["Z"]) %out_qubits_1#1 : !quantum.bit
    %9 = arith.xori %mres_0, %mres_2 : i1
    %10 = qec.ppr ["X"](2) %out_qubits#0 cond(%9) : !quantum.bit
    quantum.dealloc_qb %out_qubits_5 : !quantum.bit
    quantum.dealloc_qb %out_qubits_3 : !quantum.bit
    quantum.device_release
    %11 = quantum.alloc_qb : !quantum.bit
    %12 = qec.fabricate  magic : !quantum.bit
    %mres_6, %out_qubits_7:3 = qec.ppm ["X", "Z", "Z"] %10, %2, %12 : !quantum.bit, !quantum.bit, !quantum.bit
    %mres_8, %out_qubits_9:2 = qec.ppm ["Z", "Y"] %out_qubits_7#2, %11 : !quantum.bit, !quantum.bit
    %mres_10, %out_qubits_11 = qec.ppm ["X"] %out_qubits_9#0 : !quantum.bit
    %mres_12, %out_qubits_13 = qec.select.ppm(%mres_6, ["X"], ["Z"]) %out_qubits_9#1 : !quantum.bit
    %13 = arith.xori %mres_8, %mres_10 : i1
    %14:2 = qec.ppr ["X", "Z"](2) %out_qubits_7#0, %out_qubits_7#1 cond(%13) : !quantum.bit, !quantum.bit
    quantum.dealloc_qb %out_qubits_13 : !quantum.bit
    quantum.dealloc_qb %out_qubits_11 : !quantum.bit
    %mres_14, %out_qubits_15 = qec.ppm ["X"] %14#0 : !quantum.bit
    %from_elements = tensor.from_elements %mres_14 : tensor<i1>
    %15 = quantum.insert %0[ 0], %out_qubits_15 : !quantum.reg, !quantum.bit
    %16 = quantum.insert %15[ 1], %14#1 : !quantum.reg, !quantum.bit
    %17 = quantum.insert %16[ 2], %3 : !quantum.reg, !quantum.bit
    %18 = quantum.insert %17[ 3], %4 : !quantum.reg, !quantum.bit
    %19 = quantum.insert %18[ 4], %5 : !quantum.reg, !quantum.bit
    %20 = quantum.insert %19[ 5], %6 : !quantum.reg, !quantum.bit
    quantum.dealloc %20 : !quantum.reg
    return 
}  
  
func.func public @test_ppm_compilation_2() {
    %c0_i64 = arith.constant 0 : i64
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit
    %4 = quantum.extract %0[ 3] : !quantum.reg -> !quantum.bit
    %5 = quantum.extract %0[ 4] : !quantum.reg -> !quantum.bit
    %6 = quantum.extract %0[ 5] : !quantum.reg -> !quantum.bit
    quantum.device_release
    %7 = quantum.insert %0[ 0], %1 : !quantum.reg, !quantum.bit
    %8 = quantum.insert %7[ 1], %2 : !quantum.reg, !quantum.bit
    %9 = quantum.insert %8[ 2], %3 : !quantum.reg, !quantum.bit
    %10 = quantum.insert %9[ 3], %4 : !quantum.reg, !quantum.bit
    %11 = quantum.insert %10[ 4], %5 : !quantum.reg, !quantum.bit
    %12 = quantum.insert %11[ 5], %6 : !quantum.reg, !quantum.bit
    quantum.dealloc %12 : !quantum.reg
    return
}

// -----

//CHECK: {
//CHECK:     "game_of_surface_code": {
//CHECK:         "max_weight_pi4": 2,
//CHECK:         "max_weight_pi8": 4,
//CHECK:         "num_of_ppm": 4,
//CHECK:         "num_pi4_gates": 17,
//CHECK:         "num_pi8_gates": 4
//CHECK:     }
//CHECK: }
func.func public @game_of_surface_code(%arg0: !quantum.bit, %arg1: !quantum.bit, %arg2: !quantum.bit, %arg3: !quantum.bit) {

    %0 = qec.ppr ["Z"](8) %arg0 : !quantum.bit
    %1 = qec.ppr ["Y"](-8) %arg3 : !quantum.bit
    %2:2 = qec.ppr ["Y", "X"](8) %arg2, %arg1 : !quantum.bit, !quantum.bit
    %3:4 = qec.ppr ["Z", "Z", "Y", "Z"](-8) %2#0, %2#1, %1, %0 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
    %4 = qec.ppr ["X"](-4) %3#2 : !quantum.bit
    %5:2 = qec.ppr ["Z", "X"](4) %3#0, %3#1 : !quantum.bit, !quantum.bit
    %6 = qec.ppr ["X"](-4) %5#1 : !quantum.bit
    %7:2 = qec.ppr ["Z", "X"](4) %6, %3#3 : !quantum.bit, !quantum.bit
    %8 = qec.ppr ["Z"](-4) %7#0 : !quantum.bit
    %9 = qec.ppr ["Z"](4) %8 : !quantum.bit
    %10 = qec.ppr ["X"](4) %9 : !quantum.bit
    %11 = qec.ppr ["X"](-4) %7#1 : !quantum.bit
    %12:2 = qec.ppr ["Z", "X"](4) %4, %11 : !quantum.bit, !quantum.bit
    %13 = qec.ppr ["Z"](-4) %12#0 : !quantum.bit
    %14 = qec.ppr ["Z"](4) %13 : !quantum.bit
    %15 = qec.ppr ["X"](4) %14 : !quantum.bit
    %16 = qec.ppr ["X"](-4) %12#1 : !quantum.bit
    %17 = qec.ppr ["X"](-4) %16 : !quantum.bit
    %18 = qec.ppr ["Z"](-4) %5#0 : !quantum.bit
    %19 = qec.ppr ["X"](4) %18 : !quantum.bit
    %20 = qec.ppr ["X"](4) %19 : !quantum.bit

    %mres_4, %out_qubit_4 = qec.ppm ["Z"] %15 : !quantum.bit
    %mres_3, %out_qubit_3 = qec.ppm ["Z"] %20 : !quantum.bit
    %mres_2, %out_qubit_2 = qec.ppm ["Z"] %10 : !quantum.bit
    %mres_1, %out_qubit_1 = qec.ppm ["Z"] %17 : !quantum.bit
    return
}

// -----

//CHECK: {
//CHECK:     "static_for_loop": {
//CHECK:         "max_weight_pi4": 1,
//CHECK:         "num_of_ppm": 5,
//CHECK:         "num_pi4_gates": 5
//CHECK:     }
//CHECK: }
func.func public @static_for_loop(%arg0: !quantum.bit) {
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %q = scf.for %iter = %c0 to %c5 step %c1 iter_args(%arg1 = %arg0) -> (!quantum.bit) {
      %out_qubits = qec.ppr ["Z"](4) %arg1 : !quantum.bit
      %mres, %out_qubits_1 = qec.ppm ["Z"] %out_qubits : !quantum.bit
      scf.yield %out_qubits_1 : !quantum.bit
    }

    return
}

// -----

//CHECK: {
//CHECK:     "static_for_loop_bigstep": {
//CHECK:         "max_weight_pi4": 1,
//CHECK:         "num_of_ppm": 3,
//CHECK:         "num_pi4_gates": 3
//CHECK:     }
//CHECK: }
func.func public @static_for_loop_bigstep(%arg0: !quantum.bit) {
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    // COM: should be 3 iterations (0,2,4)

    %q = scf.for %iter = %c0 to %c5 step %c2 iter_args(%arg1 = %arg0) -> (!quantum.bit) {
      %out_qubits = qec.ppr ["Z"](4) %arg1 : !quantum.bit
      %mres, %out_qubits_1 = qec.ppm ["Z"] %out_qubits : !quantum.bit
      scf.yield %out_qubits_1 : !quantum.bit
    }

    return
}

// -----

//CHECK: {
//CHECK:     "static_for_loop_nested": {
//CHECK:         "max_weight_pi4": 1,
//CHECK:         "max_weight_pi8": 1,
//CHECK:         "num_of_ppm": 30,
//CHECK:         "num_pi4_gates": 30,
//CHECK:         "num_pi8_gates": 6
//CHECK:     }
//CHECK: }
func.func public @static_for_loop_nested(%arg0: !quantum.bit) {
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %q = scf.for %iter = %c0 to %c6 step %c1 iter_args(%arg1 = %arg0) -> (!quantum.bit) {

        %q_inner = scf.for %iter_inner = %c0 to %c5 step %c1 iter_args(%arg1_inner = %arg1) -> (!quantum.bit) {
          %out_qubits_inner = qec.ppr ["Z"](4) %arg1_inner : !quantum.bit
          %mres, %out_qubits_inner_1 = qec.ppm ["Z"] %out_qubits_inner : !quantum.bit
          scf.yield %out_qubits_inner_1 : !quantum.bit
        }

        %out_qubits = qec.ppr ["Z"](8) %q_inner : !quantum.bit
        scf.yield %out_qubits : !quantum.bit
    }

    return
}

// -----

func.func public @dynamic_for_loop_error(%arg0: !quantum.bit, %c: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %q = scf.for %iter = %c0 to %c step %c1 iter_args(%arg1 = %arg0) -> (!quantum.bit) {
      %out_qubits = qec.ppr ["Z"](4) %arg1 : !quantum.bit
      // expected-error@above {{PPM statistics is not available when there are dynamically sized for loops.}}
      %mres, %out_qubits_1 = qec.ppm ["Z"] %out_qubits : !quantum.bit
      scf.yield %out_qubits_1 : !quantum.bit
    }

    return
}

// -----

func.func public @cond_error(%arg0: !quantum.bit, %b: i1) {
    %out_qubits = scf.if %b -> !quantum.bit {
        %out_qubits_t = qec.ppr ["Z"](4) %arg0 : !quantum.bit
        // expected-error@above {{PPM statistics is not available when there are conditionals or while loops.}}
        scf.yield %out_qubits_t : !quantum.bit
    } else {
        scf.yield %arg0 : !quantum.bit
    }

    return
}

// -----

func.func public @while_error(%arg0: !quantum.bit, %b: i1) {

    %q = scf.while (%in_qubit = %arg0) : (!quantum.bit) -> (!quantum.bit) {
        scf.condition(%b) %in_qubit: !quantum.bit
    } do {
        ^bb0(%in_qubit: !quantum.bit):
        %out_qubits = qec.ppr ["Z"](4) %in_qubit : !quantum.bit
        // expected-error@above {{PPM statistics is not available when there are conditionals or while loops.}}
        scf.yield %out_qubits : !quantum.bit
    }

    return
}
