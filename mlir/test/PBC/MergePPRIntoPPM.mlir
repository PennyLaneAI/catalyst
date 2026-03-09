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

// RUN: quantum-opt --merge-ppr-ppm --split-input-file -verify-diagnostics %s | FileCheck %s
// RUN: quantum-opt --merge-ppr-ppm="max-pauli-size=3" --split-input-file -verify-diagnostics %s | FileCheck %s --check-prefixes=CHECK-MPS

// Merge tests, pi/4 rotations.
//
// The PPR can be removed after the commute, because the PPM is a terminal measurement and has no operations applied to those qubits.

func.func public @merge_ppr_ppm_pi_4_rotation_test_1(%q1: !quantum.bit) -> tensor<i1> {

    // CHECK-NOT: pbc.ppr ["X"]
    // CHECK:     pbc.ppm ["X"]
    // CHECK-NOT: pbc.ppr ["X"]
    %0 = pbc.ppr ["X"](4) %q1: !quantum.bit
    %m, %out_qubits = pbc.ppm ["X"] %0 : i1, !quantum.bit
    %from_elements = tensor.from_elements %m : tensor<i1>
    return %from_elements : tensor<i1>
}

func.func public @merge_ppr_ppm_pi_4_rotation_test_2() -> tensor<i1> {

    // CHECK: pbc.ppm ["Y"]
    // CHECK-NOT: pbc.ppr ["X"]
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = pbc.ppr ["X"](-4) %1 : !quantum.bit
    %m, %out_qubits = pbc.ppm ["Z"](-1) %2 : i1, !quantum.bit
    %from_elements = tensor.from_elements %m : tensor<i1>
    %3 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit
    quantum.dealloc %3 : !quantum.reg
    return %from_elements : tensor<i1>
}

func.func public @merge_ppr_ppm_pi_4_rotation_test_3(%q1: !quantum.bit) -> tensor<i1> {

    // CHECK: pbc.ppm ["Y"]
    // CHECK-NOT: pbc.ppr ["X"]
    %0 = pbc.ppr ["X"](-4) %q1 : !quantum.bit
    %m, %out_qubits = pbc.ppm ["Z"](-1) %0 : i1, !quantum.bit
    %from_elements = tensor.from_elements %m : tensor<i1>
    return %from_elements : tensor<i1>
}

// Commute tests, pi/4 rotations.
//
// The PPR cannot be removed after the commute, because it operates on qubits that are returned by the function.

func.func public @commute_ppr_ppm_pi_4_rotation_test_1(%q1: !quantum.bit) -> (tensor<i1>, !quantum.bit) {

    // CHECK: pbc.ppm ["Z"](-1)
    // CHECK: pbc.ppr ["X"](4)
    %0 = pbc.ppr ["X"](4) %q1: !quantum.bit
    %m, %out_qubits = pbc.ppm ["Y"] %0 : i1, !quantum.bit
    %from_elements = tensor.from_elements %m : tensor<i1>
    return %from_elements, %out_qubits : tensor<i1>, !quantum.bit
}

func.func public @commute_ppr_ppm_pi_4_rotation_test_2(%q1: !quantum.bit) -> (tensor<i1>, !quantum.bit) {

    // CHECK: pbc.ppm ["Y"]
    // CHECK: pbc.ppr ["X"](-4)
    %0 = pbc.ppr ["X"](-4) %q1: !quantum.bit
    %m, %out_qubits = pbc.ppm ["Z"](-1) %0 : i1, !quantum.bit
    %from_elements = tensor.from_elements %m : tensor<i1>
    return %from_elements, %out_qubits : tensor<i1>, !quantum.bit
}

func.func public @commute_ppr_ppm_pi_4_rotation_test_3(%q1: !quantum.bit, %q2: !quantum.bit) -> (tensor<i1>, !quantum.bit) {

    // CHECK:     [[q0:%.+]] = pbc.ppr ["X"](8) %arg0 : !quantum.bit
    // CHECK-NOT: ["X"](4)
    // CHECK:     [[m1:%.+]], [[o1:%.+]]:2 = pbc.ppm ["X", "X"] %arg1, [[q0]]
    // CHECK-NOT: ["X"](4)
    // CHECK:     %from_elements = tensor.from_elements [[m1]] : tensor<i1>
    // CHECK:     return %from_elements, [[o1]]#1 : tensor<i1>, !quantum.bit
    %0 = pbc.ppr ["X"](8) %q1: !quantum.bit
    %1 = pbc.ppr ["X"](4) %q2: !quantum.bit
    %m, %out_qubits:2 = pbc.ppm ["X", "X"] %0, %1 : i1, !quantum.bit, !quantum.bit
    %from_elements = tensor.from_elements %m : tensor<i1>
    return %from_elements, %out_qubits#0 : tensor<i1>, !quantum.bit
}

func.func public @commute_ppr_ppm_pi_4_rotation_test_4(%q1: !quantum.bit, %q2: !quantum.bit) -> (tensor<i1>, !quantum.bit) {

    // CHECK:     [[q0:%.+]] = pbc.ppr ["X"](8) %arg0 : !quantum.bit
    // CHECK-NOT: pbc.ppr ["Y"](4)
    // CHECK:     [[m1:%.+]], [[o1:%.+]]:2 = pbc.ppm ["Z", "X"] %arg1, [[q0]]
    // CHECK-NOT: pbc.ppr ["Y"](4)
    // CHECK:     %from_elements = tensor.from_elements [[m1]] : tensor<i1>
    // CHECK:     return %from_elements, [[o1]]#1 : tensor<i1>, !quantum.bit
    %0 = pbc.ppr ["X"](8) %q1: !quantum.bit
    %1 = pbc.ppr ["Y"](4) %q2: !quantum.bit
    %m, %out_qubits:2 = pbc.ppm ["X", "X"] %0, %1 : i1, !quantum.bit, !quantum.bit
    %from_elements = tensor.from_elements %m : tensor<i1>
    return %from_elements, %out_qubits#0 : tensor<i1>, !quantum.bit
}

func.func public @commute_ppr_ppm_pi_4_rotation_test_5(%q1: !quantum.bit, %q2: !quantum.bit) -> (tensor<i1>, !quantum.bit) {

    // CHECK: [[q1:%.+]]:2 = pbc.ppr ["X", "X"](8) %arg0, %arg1
    // CHECK: [[m1:%.+]], [[o1:%.+]]:2 = pbc.ppm ["X", "Z"] [[q1]]#0, [[q1]]#1
    // CHECK: [[q2:%.+]]:2 = pbc.ppr ["Y", "X"](4) [[o1]]#0, [[o1]]#1
    // CHECK: [[f1:%.+]] = tensor.from_elements [[m1]] : tensor<i1>
    // CHECK: return [[f1]], [[q2]]#0 : tensor<i1>, !quantum.bit
    %0:2 = pbc.ppr ["X", "X"](8) %q1, %q2 : !quantum.bit, !quantum.bit
    %1:2 = pbc.ppr ["Y", "X"](4) %0#0, %0#1 : !quantum.bit, !quantum.bit
    %m, %out_qubits:2 = pbc.ppm ["X", "Z"] %1#0, %1#1 : i1, !quantum.bit, !quantum.bit
    %from_elements = tensor.from_elements %m : tensor<i1>
    return %from_elements, %out_qubits#0 : tensor<i1>, !quantum.bit
}

// Merge tests, pi/2 rotations.
//
// The PPR can be removed after the commute, because the PPM is a terminal measurement and has no operations applied to those qubits.

func.func public @merge_ppr_ppm_pi_2_rotation_test_1(%q1: !quantum.bit) -> tensor<i1> {

    // CHECK-NOT: pbc.ppr ["X"]
    // CHECK:     pbc.ppm ["X"]
    // CHECK-NOT: pbc.ppr ["X"]
    %0 = pbc.ppr ["X"](2) %q1: !quantum.bit
    %m, %out_qubits = pbc.ppm ["X"] %0 : i1, !quantum.bit
    %from_elements = tensor.from_elements %m : tensor<i1>
    return %from_elements : tensor<i1>
}

func.func public @merge_ppr_ppm_pi_2_rotation_test_2() -> tensor<i1> {

    // CHECK: pbc.ppm ["Z"]
    // CHECK-NOT: pbc.ppr ["X"]
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = pbc.ppr ["X"](-2) %1 : !quantum.bit
    %m, %out_qubits = pbc.ppm ["Z"](-1) %2 : i1, !quantum.bit
    %from_elements = tensor.from_elements %m : tensor<i1>
    %3 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit
    quantum.dealloc %3 : !quantum.reg
    return %from_elements : tensor<i1>
}

func.func public @merge_ppr_ppm_pi_2_rotation_test_3(%q1: !quantum.bit) -> tensor<i1> {

    // CHECK: pbc.ppm ["Z"]
    // CHECK-NOT: pbc.ppr ["X"]
    %0 = pbc.ppr ["X"](-2) %q1 : !quantum.bit
    %m, %out_qubits = pbc.ppm ["Z"](-1) %0 : i1, !quantum.bit
    %from_elements = tensor.from_elements %m : tensor<i1>
    return %from_elements : tensor<i1>
}

// Commute tests, pi/2 rotations.
//
// The PPR cannot be removed after the commute, because it operates on qubits that are returned by the function.

func.func public @commute_ppr_ppm_pi_2_rotation_test_1(%q1: !quantum.bit) -> (tensor<i1>, !quantum.bit) {

    // CHECK: pbc.ppm ["Y"](-1)
    // CHECK: pbc.ppr ["X"](2)
    %0 = pbc.ppr ["X"](2) %q1: !quantum.bit
    %m, %out_qubits = pbc.ppm ["Y"] %0 : i1, !quantum.bit
    %from_elements = tensor.from_elements %m : tensor<i1>
    return %from_elements, %out_qubits : tensor<i1>, !quantum.bit
}

func.func public @commute_ppr_ppm_pi_2_rotation_test_2(%q1: !quantum.bit) -> (tensor<i1>, !quantum.bit) {

    // CHECK: pbc.ppm ["Z"] %
    // CHECK: pbc.ppr ["X"](-2)
    %0 = pbc.ppr ["X"](-2) %q1: !quantum.bit
    %m, %out_qubits = pbc.ppm ["Z"](-1) %0 : i1, !quantum.bit
    %from_elements = tensor.from_elements %m : tensor<i1>
    return %from_elements, %out_qubits : tensor<i1>, !quantum.bit
}

func.func public @merge_ppr_ppm_different_size_test(%q1: !quantum.bit, %q2: !quantum.bit, %q3: !quantum.bit) -> (tensor<i1>, !quantum.bit) {
    // CHECK: pbc.ppm ["Z"](-1)
    %0:3 = pbc.ppr ["X", "Y", "Z"](2) %q1, %q2, %q3 : !quantum.bit, !quantum.bit, !quantum.bit
    %mres, %out_qubits = pbc.ppm ["Z"] %0#0 : i1, !quantum.bit
    %from_elements = tensor.from_elements %mres : tensor<i1>
    return %from_elements, %out_qubits : tensor<i1>, !quantum.bit
}

func.func public @game_of_surface_code(%arg0: !quantum.bit, %arg1: !quantum.bit, %arg2: !quantum.bit, %arg3: !quantum.bit) {

    // q1
    // CHECK: [[q0:%.+]] = pbc.ppr ["Z"](8) %arg0
    // q4
    // CHECK: [[q1:%.+]] = pbc.ppr ["Y"](-8) %arg3
    // q3, q2
    // CHECK: [[q2:%.+]]:2 = pbc.ppr ["Y", "X"](8) %arg2, %arg1

    // q3, q2, q4, q1
    // CHECK: [[q3:%.+]]:4 = pbc.ppr ["Z", "Z", "Y", "Z"](-8) [[q2]]#0, [[q2]]#1, [[q1]], [[q0]]

    // q3, q2, q1, q4
    // CHECK: [[m1:%.+]], [[o1:%.+]]:4 = pbc.ppm ["Z", "Z", "Y", "Y"] [[q3]]#0, [[q3]]#1, [[q3]]#3, [[q3]]#2

    // q2, q1
    // CHECK: [[m2:%.+]], [[o2:%.+]]:2 = pbc.ppm ["X", "X"] [[o1]]#1, [[o1]]#2
    // q3
    // CHECK: [[m3:%.+]], [[o3:%.+]] = pbc.ppm ["Z"](-1) [[o1]]#0

    // q1, q4
    // CHECK: [[m4:%.+]], [[o4:%.+]]:2 = pbc.ppm ["X", "X"] [[o2]]#1, [[o1]]#3

    // Because the Pauli size is limited to 3, the two operations below are not commuted.
    // CHECK-MPS: pbc.ppr ["Z", "X"](4)
    // CHECK-MPS: pbc.ppm ["Y", "Y", "Y"](-1)
    // CHECK-MPS-NOT: pbc.ppm ["Z", "Z", "Y", "Y"]

    %0 = pbc.ppr ["Z"](8) %arg0 : !quantum.bit
    %1 = pbc.ppr ["Y"](-8) %arg3 : !quantum.bit
    %2:2 = pbc.ppr ["Y", "X"](8) %arg2, %arg1 : !quantum.bit, !quantum.bit
    %3:4 = pbc.ppr ["Z", "Z", "Y", "Z"](-8) %2#0, %2#1, %1, %0 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
    %4 = pbc.ppr ["X"](-4) %3#2 : !quantum.bit
    %5:2 = pbc.ppr ["Z", "X"](4) %3#0, %3#1 : !quantum.bit, !quantum.bit
    %6 = pbc.ppr ["X"](-4) %5#1 : !quantum.bit
    %7:2 = pbc.ppr ["Z", "X"](4) %6, %3#3 : !quantum.bit, !quantum.bit
    %8 = pbc.ppr ["Z"](-4) %7#0 : !quantum.bit
    %9 = pbc.ppr ["Z"](4) %8 : !quantum.bit
    %10 = pbc.ppr ["X"](4) %9 : !quantum.bit
    %11 = pbc.ppr ["X"](-4) %7#1 : !quantum.bit
    %12:2 = pbc.ppr ["Z", "X"](4) %4, %11 : !quantum.bit, !quantum.bit
    %13 = pbc.ppr ["Z"](-4) %12#0 : !quantum.bit
    %14 = pbc.ppr ["Z"](4) %13 : !quantum.bit
    %15 = pbc.ppr ["X"](4) %14 : !quantum.bit
    %16 = pbc.ppr ["X"](-4) %12#1 : !quantum.bit
    %17 = pbc.ppr ["X"](-4) %16 : !quantum.bit
    %18 = pbc.ppr ["Z"](-4) %5#0 : !quantum.bit
    %19 = pbc.ppr ["X"](4) %18 : !quantum.bit
    %20 = pbc.ppr ["X"](4) %19 : !quantum.bit

    %mres_4, %out_qubit_4 = pbc.ppm ["Z"] %15 : i1, !quantum.bit
    %mres_3, %out_qubit_3 = pbc.ppm ["Z"] %20 : i1, !quantum.bit
    %mres_2, %out_qubit_2 = pbc.ppm ["Z"] %10 : i1, !quantum.bit
    %mres_1, %out_qubit_1 = pbc.ppm ["Z"] %17 : i1, !quantum.bit
    return
}

func.func public @circuit_transformed_0() -> (tensor<i1>, tensor<i1>) {

    // CHECK: quantum.device
    // CHECK: [[qreg:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[q0:%.+]] = quantum.extract [[qreg]][ 0]
    // CHECK: [[q1:%.+]] = quantum.extract [[qreg]][ 1]
    // CHECK: [[q2:%.+]]:2 = pbc.ppr ["X", "Z"](8) [[q0]], [[q1]]
    // CHECK: [[m1:%.+]], [[o1:%.+]]:2 = pbc.ppm ["X", "Z"] [[q2]]#0, [[q2]]#1
    // CHECK: [[m2:%.+]], [[o2:%.+]] = pbc.ppm ["X"] [[o1]]#0 : i1, !quantum.bit
    // CHECK: [[f1:%.+]] = tensor.from_elements [[m2]] : tensor<i1>
    // CHECK: [[q3:%.+]] = quantum.insert [[qreg]][ 0], [[o2]]
    // CHECK: [[f2:%.+]] = tensor.from_elements [[m1]] : tensor<i1>
    // CHECK: [[q4:%.+]] = quantum.insert [[q3]][ 1], [[o1]]#1
    // CHECK: quantum.dealloc [[q4]] : !quantum.reg
    // CHECK: quantum.device_release
    // CHECK: return [[f1]], [[f2]] : tensor<i1>, tensor<i1>


    %c1_i64 = arith.constant 1 : i64
    quantum.device shots(%c1_i64) ["../runtime/build/lib/librtd_null_qubit.dylib", "NullQubit", ""]
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3:2 = pbc.ppr ["X", "Z"](8) %1, %2 : !quantum.bit, !quantum.bit
    %4 = pbc.ppr ["Z"](4) %3#0 : !quantum.bit
    %5 = pbc.ppr ["X"](4) %4 : !quantum.bit
    %6 = pbc.ppr ["Z"](4) %5 : !quantum.bit
    %7:2 = pbc.ppr ["Z", "X"](4) %6, %3#1 : !quantum.bit, !quantum.bit
    %8 = pbc.ppr ["Z"](-4) %7#0 : !quantum.bit
    %mres, %out_qubits = pbc.ppm ["Z"] %8 : i1, !quantum.bit
    %from_elements = tensor.from_elements %mres : tensor<i1>
    %9 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit
    %10 = pbc.ppr ["X"](-4) %7#1 : !quantum.bit
    %mres_0, %out_qubits_1 = pbc.ppm ["Z"] %10 : i1, !quantum.bit
    %from_elements_2 = tensor.from_elements %mres_0 : tensor<i1>
    %11 = quantum.insert %9[ 1], %out_qubits_1 : !quantum.reg, !quantum.bit
    quantum.dealloc %11 : !quantum.reg
    quantum.device_release
    return %from_elements, %from_elements_2 : tensor<i1>, tensor<i1>
}
