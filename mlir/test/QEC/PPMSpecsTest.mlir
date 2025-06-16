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

// RUN: quantum-opt --ppm-specs --split-input-file -verify-diagnostics %s > %t.ppm

func.func @game_of_surface_code() -> (tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>) {
    %0 = quantum.alloc( 4) : !quantum.reg
    %1 = quantum.extract %0[ 3] : !quantum.reg -> !quantum.bit
    %2 = quantum.alloc_qb : !quantum.bit
    %3 = qec.fabricate  magic_conj : !quantum.bit
    %mres, %out_qubits:2 = qec.ppm ["X", "Z"] %1, %3 : !quantum.bit, !quantum.bit
    %mres_0, %out_qubits_1:2 = qec.ppm ["Z", "Y"] %out_qubits#1, %2 : !quantum.bit, !quantum.bit
    %mres_2, %out_qubits_3 = qec.ppm ["X"] %out_qubits_1#0 : !quantum.bit
    %mres_4, %out_qubits_5 = qec.select.ppm(%mres, ["X"], ["Z"]) %out_qubits_1#1 : !quantum.bit
    %4 = arith.xori %mres_0, %mres_2 : i1
    %5 = qec.ppr ["X"](2) %out_qubits#0 cond(%4) : !quantum.bit
    quantum.dealloc_qb %out_qubits_5 : !quantum.bit
    quantum.dealloc_qb %out_qubits_3 : !quantum.bit
    %6 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %7 = quantum.alloc_qb : !quantum.bit
    %8 = qec.fabricate  magic : !quantum.bit
    %mres_6, %out_qubits_7:2 = qec.ppm ["Z", "Z"] %6, %8 : !quantum.bit, !quantum.bit
    %mres_8, %out_qubits_9:2 = qec.ppm ["Z", "Y"] %out_qubits_7#1, %7 : !quantum.bit, !quantum.bit
    %mres_10, %out_qubits_11 = qec.ppm ["X"] %out_qubits_9#0 : !quantum.bit
    %mres_12, %out_qubits_13 = qec.select.ppm(%mres_6, ["X"], ["Z"]) %out_qubits_9#1 : !quantum.bit
    %9 = arith.xori %mres_8, %mres_10 : i1
    %10 = qec.ppr ["Z"](2) %out_qubits_7#0 cond(%9) : !quantum.bit
    quantum.dealloc_qb %out_qubits_13 : !quantum.bit
    quantum.dealloc_qb %out_qubits_11 : !quantum.bit
    %11 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit
    %12 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %13 = quantum.alloc_qb : !quantum.bit
    %14 = qec.fabricate  magic : !quantum.bit
    %mres_14, %out_qubits_15:3 = qec.ppm ["X", "X", "Z"] %11, %12, %14 : !quantum.bit, !quantum.bit, !quantum.bit
    %mres_16, %out_qubits_17:2 = qec.ppm ["Z", "Y"] %out_qubits_15#2, %13 : !quantum.bit, !quantum.bit
    %mres_18, %out_qubits_19 = qec.ppm ["X"] %out_qubits_17#0 : !quantum.bit
    %mres_20, %out_qubits_21 = qec.select.ppm(%mres_14, ["X"], ["Z"]) %out_qubits_17#1 : !quantum.bit
    %15 = arith.xori %mres_16, %mres_18 : i1
    %16:2 = qec.ppr ["X", "X"](2) %out_qubits_15#0, %out_qubits_15#1 cond(%15) : !quantum.bit, !quantum.bit
    quantum.dealloc_qb %out_qubits_21 : !quantum.bit
    quantum.dealloc_qb %out_qubits_19 : !quantum.bit
    quantum.device_release
    %17 = quantum.alloc_qb : !quantum.bit
    %18 = qec.fabricate  magic : !quantum.bit
    %mres_22, %out_qubits_23:2 = qec.ppm ["Z", "Z"] %16#0, %18 : !quantum.bit, !quantum.bit
    %mres_24, %out_qubits_25:2 = qec.ppm ["Z", "Y"] %out_qubits_23#1, %17 : !quantum.bit, !quantum.bit
    %mres_26, %out_qubits_27 = qec.ppm ["X"] %out_qubits_25#0 : !quantum.bit
    %mres_28, %out_qubits_29 = qec.select.ppm(%mres_22, ["X"], ["Z"]) %out_qubits_25#1 : !quantum.bit
    %19 = arith.xori %mres_24, %mres_26 : i1
    %20 = qec.ppr ["Z"](2) %out_qubits_23#0 cond(%19) : !quantum.bit
    quantum.dealloc_qb %out_qubits_29 : !quantum.bit
    quantum.dealloc_qb %out_qubits_27 : !quantum.bit
    %21 = quantum.alloc_qb : !quantum.bit
    %22 = qec.fabricate  magic : !quantum.bit
    %mres_30, %out_qubits_31:3 = qec.ppm ["X", "X", "Z"] %20, %16#1, %22 : !quantum.bit, !quantum.bit, !quantum.bit
    %mres_32, %out_qubits_33:2 = qec.ppm ["Z", "Y"] %out_qubits_31#2, %21 : !quantum.bit, !quantum.bit
    %mres_34, %out_qubits_35 = qec.ppm ["X"] %out_qubits_33#0 : !quantum.bit
    %mres_36, %out_qubits_37 = qec.select.ppm(%mres_30, ["X"], ["Z"]) %out_qubits_33#1 : !quantum.bit
    %23 = arith.xori %mres_32, %mres_34 : i1
    %24:2 = qec.ppr ["X", "X"](2) %out_qubits_31#0, %out_qubits_31#1 cond(%23) : !quantum.bit, !quantum.bit
    quantum.dealloc_qb %out_qubits_37 : !quantum.bit
    quantum.dealloc_qb %out_qubits_35 : !quantum.bit
    %25 = quantum.alloc_qb : !quantum.bit
    %26 = qec.fabricate  magic_conj : !quantum.bit
    %mres_38, %out_qubits_39:4 = qec.ppm ["Z", "Y", "X", "Z"] %24#0, %24#1, %10, %26 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
    %mres_40, %out_qubits_41:2 = qec.ppm ["Z", "Y"] %out_qubits_39#3, %25 : !quantum.bit, !quantum.bit
    %mres_42, %out_qubits_43 = qec.ppm ["X"] %out_qubits_41#0 : !quantum.bit
    %mres_44, %out_qubits_45 = qec.select.ppm(%mres_38, ["X"], ["Z"]) %out_qubits_41#1 : !quantum.bit
    %27 = arith.xori %mres_40, %mres_42 : i1
    %28:3 = qec.ppr ["Z", "Y", "X"](2) %out_qubits_39#0, %out_qubits_39#1, %out_qubits_39#2 cond(%27) : !quantum.bit, !quantum.bit, !quantum.bit
    quantum.dealloc_qb %out_qubits_45 : !quantum.bit
    quantum.dealloc_qb %out_qubits_43 : !quantum.bit
    %29 = quantum.alloc_qb : !quantum.bit
    %30 = qec.fabricate  magic_conj : !quantum.bit
    %mres_46, %out_qubits_47:3 = qec.ppm ["Y", "X", "Z"] %5, %28#2, %30 : !quantum.bit, !quantum.bit, !quantum.bit
    %mres_48, %out_qubits_49:2 = qec.ppm ["Z", "Y"] %out_qubits_47#2, %29 : !quantum.bit, !quantum.bit
    %mres_50, %out_qubits_51 = qec.ppm ["X"] %out_qubits_49#0 : !quantum.bit
    %mres_52, %out_qubits_53 = qec.select.ppm(%mres_46, ["X"], ["Z"]) %out_qubits_49#1 : !quantum.bit
    %31 = arith.xori %mres_48, %mres_50 : i1
    %32:2 = qec.ppr ["Y", "X"](2) %out_qubits_47#0, %out_qubits_47#1 cond(%31) : !quantum.bit, !quantum.bit
    quantum.dealloc_qb %out_qubits_53 : !quantum.bit
    quantum.dealloc_qb %out_qubits_51 : !quantum.bit
    %33 = quantum.alloc_qb : !quantum.bit
    %34 = qec.fabricate  magic : !quantum.bit
    %mres_54, %out_qubits_55:5 = qec.ppm ["Z", "Z", "Z", "Z", "Z"] %32#0, %28#0, %28#1, %32#1, %34 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
    %mres_56, %out_qubits_57:2 = qec.ppm ["Z", "Y"] %out_qubits_55#4, %33 : !quantum.bit, !quantum.bit
    %mres_58, %out_qubits_59 = qec.ppm ["X"] %out_qubits_57#0 : !quantum.bit
    %mres_60, %out_qubits_61 = qec.select.ppm(%mres_54, ["X"], ["Z"]) %out_qubits_57#1 : !quantum.bit
    %35 = arith.xori %mres_56, %mres_58 : i1
    %36:4 = qec.ppr ["Z", "Z", "Z", "Z"](2) %out_qubits_55#0, %out_qubits_55#1, %out_qubits_55#2, %out_qubits_55#3 cond(%35) : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
    quantum.dealloc_qb %out_qubits_61 : !quantum.bit
    quantum.dealloc_qb %out_qubits_59 : !quantum.bit
    %37 = quantum.alloc_qb : !quantum.bit
    %38 = qec.fabricate  magic_conj : !quantum.bit
    %mres_62, %out_qubits_63:2 = qec.ppm ["X", "Z"] %36#3, %38 : !quantum.bit, !quantum.bit
    %mres_64, %out_qubits_65:2 = qec.ppm ["Z", "Y"] %out_qubits_63#1, %37 : !quantum.bit, !quantum.bit
    %mres_66, %out_qubits_67 = qec.ppm ["X"] %out_qubits_65#0 : !quantum.bit
    %mres_68, %out_qubits_69 = qec.select.ppm(%mres_62, ["X"], ["Z"]) %out_qubits_65#1 : !quantum.bit
    %39 = arith.xori %mres_64, %mres_66 : i1
    %40 = qec.ppr ["X"](2) %out_qubits_63#0 cond(%39) : !quantum.bit
    quantum.dealloc_qb %out_qubits_69 : !quantum.bit
    quantum.dealloc_qb %out_qubits_67 : !quantum.bit
    %mres_70, %out_qubits_71 = qec.ppm ["Z"] %36#0 : !quantum.bit
    %from_elements = tensor.from_elements %mres_70 : tensor<i1>
    %mres_72, %out_qubits_73:4 = qec.ppm ["Z", "Z", "Z", "Z"] %36#1, %36#2, %40, %out_qubits_71 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
    %from_elements_74 = tensor.from_elements %mres_72 : tensor<i1>
    %mres_75, %out_qubits_76 = qec.ppm ["Z"] %out_qubits_73#0 : !quantum.bit
    %from_elements_77 = tensor.from_elements %mres_75 : tensor<i1>
    %mres_78, %out_qubits_79:2 = qec.ppm ["Z", "Z"] %out_qubits_76, %out_qubits_73#1 : !quantum.bit, !quantum.bit
    %from_elements_80 = tensor.from_elements %mres_78 : tensor<i1>
    %41 = quantum.insert %0[ 0], %out_qubits_73#2 : !quantum.reg, !quantum.bit
    %42 = quantum.insert %41[ 2], %out_qubits_79#0 : !quantum.reg, !quantum.bit
    %43 = quantum.insert %42[ 1], %out_qubits_79#1 : !quantum.reg, !quantum.bit
    %44 = quantum.insert %43[ 3], %out_qubits_73#3 : !quantum.reg, !quantum.bit
    quantum.dealloc %44 : !quantum.reg
    return %from_elements_74, %from_elements_80, %from_elements_77, %from_elements : tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>
    // CHECK:      "game_of_surface_code": {
    // CHECK:         "max_weight_pi2": 4,
    // CHECK:         "num_logical_qubits": 4,
    // CHECK:         "num_of_ppm": 31,
    // CHECK:         "num_pi2_gates": 9
}


// -----
