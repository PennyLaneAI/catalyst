// Copyright 2024-2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --pass-pipeline="builtin.module(merge-rotations)" --split-input-file -verify-diagnostics %s | FileCheck %s

func.func @test_merge_rotations(%arg0: f64, %arg1: f64) -> !quantum.bit {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[reg:%.+]] = quantum.alloc( 1) : !quantum.reg
    // CHECK: [[qubit:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[sum:%.+]] = arith.addf %arg0, %arg1 : f64
    // CHECK: [[ret:%.+]] = quantum.custom "RX"([[sum]]) [[qubit]] : !quantum.bit
    // CHECK-NOT: quantum.custom "RX"
    %2 = quantum.custom "RX"(%arg0) %1 : !quantum.bit
    %3 = quantum.custom "RX"(%arg1) %2 : !quantum.bit
    // CHECK: return [[ret]]
    return %3 : !quantum.bit
}

// -----

func.func @test_merge_rotations(%arg0: f64, %arg1: f64) -> !quantum.bit {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[reg:%.+]] = quantum.alloc( 1) : !quantum.reg
    // CHECK: [[qubit:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[sum:%.+]] = arith.addf %arg0, %arg1 : f64
    // CHECK: [[ret:%.+]] = quantum.custom "PhaseShift"([[sum]]) [[qubit]] : !quantum.bit
    // CHECK-NOT: quantum.custom "PhaseShift"
    %2 = quantum.custom "PhaseShift"(%arg0) %1 : !quantum.bit
    %3 = quantum.custom "PhaseShift"(%arg1) %2 : !quantum.bit
    // CHECK: return [[ret]]
    return %3 : !quantum.bit
}

// -----

func.func @test_merge_rotations(%arg0: f64, %arg1: f64, %arg2: f64) -> !quantum.bit {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit

    // CHECK: [[reg:%.+]] = quantum.alloc( 1) : !quantum.reg
    // CHECK: [[qubit:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[sum1:%.+]] = arith.addf %arg1, %arg2 : f64
    // CHECK: [[sum2:%.+]] = arith.addf %arg0, [[sum1]] : f64
    // CHECK: [[ret:%.+]] = quantum.custom "RX"([[sum2]]) [[qubit]] : !quantum.bit
    // CHECK-NOT: quantum.custom "RX"
    %2 = quantum.custom "RX"(%arg0) %1 : !quantum.bit
    %3 = quantum.custom "RX"(%arg1) %2 : !quantum.bit
    %4 = quantum.custom "RX"(%arg2) %3 : !quantum.bit
    // CHECK: return [[ret]]
    return %4 : !quantum.bit
}

// -----

func.func @test_merge_rotations(%arg0: f64, %arg1: f64, %arg2: f64) -> (!quantum.bit, !quantum.bit) {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    // CHECK: [[reg:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[qubit1:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit2:%.+]] = quantum.extract [[reg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[sum1:%.+]] = arith.addf %arg1, %arg2 : f64
    // CHECK: [[sum2:%.+]] = arith.addf %arg0, [[sum1]] : f64
    // CHECK: [[ret:%.+]]:2 = quantum.custom "CRX"([[sum2]]) [[qubit1]], [[qubit2]] : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "CRX"
    %3:2 = quantum.custom "CRX"(%arg0) %1, %2: !quantum.bit, !quantum.bit
    %4:2 = quantum.custom "CRX"(%arg1) %3#0, %3#1 : !quantum.bit, !quantum.bit
    %5:2 = quantum.custom "CRX"(%arg2) %4#0, %4#1 : !quantum.bit, !quantum.bit
    // CHECK: return [[ret]]#0, [[ret]]#1
    return %5#0, %5#1 : !quantum.bit, !quantum.bit
}

// -----

func.func @test_merge_rotations(%arg0: f64, %arg1: f64, %arg2: f64) -> (!quantum.bit, !quantum.bit) {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    // CHECK: [[reg:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[qubit1:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit2:%.+]] = quantum.extract [[reg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[sum:%.+]] = arith.addf %arg0, %arg1 : f64
    // CHECK: [[qubits3:%.+]]:2 = quantum.custom "CRY"([[sum]]) [[qubit1]], [[qubit2]] : !quantum.bit, !quantum.bit
    // CHECK: [[ret:%.+]]:2 = quantum.custom "CRY"(%arg2) [[qubits3]]#1, [[qubits3]]#0 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "CRY"
    %3:2 = quantum.custom "CRY"(%arg0) %1, %2: !quantum.bit, !quantum.bit
    %4:2 = quantum.custom "CRY"(%arg1) %3#0, %3#1 : !quantum.bit, !quantum.bit
    %5:2 = quantum.custom "CRY"(%arg2) %4#1, %4#0 : !quantum.bit, !quantum.bit
    // CHECK: return [[ret]]#0, [[ret]]#1
    return %5#0, %5#1 : !quantum.bit, !quantum.bit
}

// -----

func.func @test_merge_rotations(%arg0: f64, %arg1: f64, %arg2: f64) -> (!quantum.bit, !quantum.bit) {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    // CHECK: [[reg:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[qubit1:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit2:%.+]] = quantum.extract [[reg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[sum1:%.+]] = arith.addf %arg1, %arg2 : f64
    // CHECK: [[sum2:%.+]] = arith.addf %arg0, [[sum1]] : f64
    // CHECK: [[ret:%.+]]:2 = quantum.custom "ControlledPhaseShift"([[sum2]]) [[qubit1]], [[qubit2]] : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "CRX"
    %3:2 = quantum.custom "ControlledPhaseShift"(%arg0) %1, %2: !quantum.bit, !quantum.bit
    %4:2 = quantum.custom "ControlledPhaseShift"(%arg1) %3#0, %3#1 : !quantum.bit, !quantum.bit
    %5:2 = quantum.custom "ControlledPhaseShift"(%arg2) %4#0, %4#1 : !quantum.bit, !quantum.bit
    // CHECK: return [[ret]]#0, [[ret]]#1
    return %5#0, %5#1 : !quantum.bit, !quantum.bit
}

// -----

func.func @test_merge_rotations(%arg0: f64, %arg1: f64, %arg2: f64) -> (!quantum.bit, !quantum.bit) {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    // CHECK: [[reg:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[qubit1:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit2:%.+]] = quantum.extract [[reg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[sum:%.+]] = arith.addf %arg0, %arg1 : f64
    // CHECK: [[qubits3:%.+]]:2 = quantum.custom "ControlledPhaseShift"([[sum]]) [[qubit1]], [[qubit2]] : !quantum.bit, !quantum.bit
    // CHECK: [[ret:%.+]]:2 = quantum.custom "ControlledPhaseShift"(%arg2) [[qubits3]]#1, [[qubits3]]#0 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "ControlledPhaseShift"
    %3:2 = quantum.custom "ControlledPhaseShift"(%arg0) %1, %2: !quantum.bit, !quantum.bit
    %4:2 = quantum.custom "ControlledPhaseShift"(%arg1) %3#0, %3#1 : !quantum.bit, !quantum.bit
    %5:2 = quantum.custom "ControlledPhaseShift"(%arg2) %4#1, %4#0 : !quantum.bit, !quantum.bit
    // CHECK: return [[ret]]#0, [[ret]]#1
    return %5#0, %5#1 : !quantum.bit, !quantum.bit
}

// -----

// Arbitrary rotations, general case: three rotations, two merges
//
// Rot(arg0, arg1, arg2)
// Rot(arg1, arg2, arg0)
// Rot(arg2, arg0, arg1)
//
// Expected output: a single Rot

func.func @test_merge_rotations(%arg0: f64, %arg1: f64, %arg2: f64) -> !quantum.bit {
    // CHECK: [[twoConst:%.+]] = arith.constant 2.000000e+00 : f64
    // CHECK: [[halfConst:%.+]] = arith.constant 5.000000e-01 : f64
    // CHECK: [[reg:%.+]] = quantum.alloc( 1) : !quantum.reg
    // CHECK: [[qubit:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    %twoConst = arith.constant 2.000000e+00 : f64
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit

    // First merge
    //
    // CHECK: [[cF_1:%.+]] = math.sqrt {{.+}} : f64
    // CHECK: [[acosCF_1:%.+]] = math.acos [[cF_1]] : f64
    // CHECK: [[thetaF_1:%.+]] = arith.mulf [[acosCF_1]], [[twoConst]] : f64
    // CHECK: [[atan_1:%.+]] = math.atan {{.*}} : f64
    // CHECK: [[alphaF_1:%.+]] = arith.negf [[atan_1]] : f64
    // CHECK: [[atan_2:%.+]] = math.atan {{.*}} : f64
    // CHECK: [[betaF_1:%.+]] = arith.negf [[atan_2]] : f64

    // Second merge
    //
    // CHECK: [[cF_2:%.+]] = math.sqrt {{.+}} : f64
    // CHECK: [[acosCF_2:%.+]] = math.acos [[cF_2]] : f64
    // CHECK: [[thetaF_2:%.+]] = arith.mulf [[acosCF_2]], [[twoConst]] : f64
    // CHECK: [[atan_3:%.+]] = math.atan {{.*}} : f64
    // CHECK: [[alphaF_2:%.+]] = arith.negf [[atan_3]] : f64
    // CHECK: [[atan_4:%.+]] = math.atan {{.*}} : f64
    // CHECK: [[betaF_2:%.+]] = arith.negf [[atan_4]] : f64
    // CHECK: [[phiF_2:%.+]] = arith.addf [[alphaF_2]], [[betaF_2]] : f64
    // CHECK: [[omegaF_2:%.+]] = arith.subf [[alphaF_2]], [[betaF_2]] : f64

    // CHECK: [[ret:%.+]] = quantum.custom "Rot"([[phiF_2]], [[thetaF_2]], [[omegaF_2]]) [[qubit]] : !quantum.bit
    // CHECK-NOT: quantum.custom "Rot"
    %2 = quantum.custom "Rot"(%arg0, %arg1, %arg2) %1 : !quantum.bit
    %3 = quantum.custom "Rot"(%arg1, %arg2, %arg0) %2 : !quantum.bit
    %4 = quantum.custom "Rot"(%arg2, %arg0, %arg1) %3 : !quantum.bit

    // CHECK: return [[ret]]
    return %4 : !quantum.bit
}

// -----

// Arbitrary controlled rotations, general case: three rotations, two merges
//
// CRot(arg0, arg1, arg2)
// CRot(arg1, arg2, arg0)
// CRot(arg2, arg0, arg1)
//
// Expected output: a single CRot

func.func @test_merge_rotations(%arg0: f64, %arg1: f64, %arg2: f64) -> (!quantum.bit, !quantum.bit) {
    // CHECK: [[twoConst:%.+]] = arith.constant 2.000000e+00 : f64
    // CHECK: [[halfConst:%.+]] = arith.constant 5.000000e-01 : f64
    // CHECK: [[reg:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[qubit1:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit2:%.+]] = quantum.extract [[reg]][ 1] : !quantum.reg -> !quantum.bit
    %twoConst = arith.constant 2.000000e+00 : f64
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    // First merge
    //
    // CHECK: [[cF_1:%.+]] = math.sqrt {{.+}} : f64
    // CHECK: [[acosCF_1:%.+]] = math.acos [[cF_1]] : f64
    // CHECK: [[thetaF_1:%.+]] = arith.mulf [[acosCF_1]], [[twoConst]] : f64
    // CHECK: [[atan_1:%.+]] = math.atan {{.*}} : f64
    // CHECK: [[alphaF_1:%.+]] = arith.negf [[atan_1]] : f64
    // CHECK: [[atan_2:%.+]] = math.atan {{.*}} : f64
    // CHECK: [[betaF_1:%.+]] = arith.negf [[atan_2]] : f64

    // Second merge
    //
    // CHECK: [[cF_2:%.+]] = math.sqrt {{.+}} : f64
    // CHECK: [[acosCF_2:%.+]] = math.acos [[cF_2]] : f64
    // CHECK: [[thetaF_2:%.+]] = arith.mulf [[acosCF_2]], [[twoConst]] : f64
    // CHECK: [[atan_3:%.+]] = math.atan {{.*}} : f64
    // CHECK: [[alphaF_2:%.+]] = arith.negf [[atan_3]] : f64
    // CHECK: [[atan_4:%.+]] = math.atan {{.*}} : f64
    // CHECK: [[betaF_2:%.+]] = arith.negf [[atan_4]] : f64
    // CHECK: [[phiF_2:%.+]] = arith.addf [[alphaF_2]], [[betaF_2]] : f64
    // CHECK: [[omegaF_2:%.+]] = arith.subf [[alphaF_2]], [[betaF_2]] : f64

    // CHECK: [[ret:%.+]]:2 = quantum.custom "CRot"([[phiF_2]], [[thetaF_2]], [[omegaF_2]]) [[qubit1]], [[qubit2]] : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "CRot"
    %3:2 = quantum.custom "CRot"(%arg0, %arg1, %arg2) %1, %2 : !quantum.bit, !quantum.bit
    %4:2 = quantum.custom "CRot"(%arg1, %arg2, %arg0) %3#0, %3#1 : !quantum.bit, !quantum.bit
    %5:2 = quantum.custom "CRot"(%arg2, %arg0, %arg1) %4#0, %4#1 : !quantum.bit, !quantum.bit

    // CHECK: return [[ret]]#0, [[ret]]#1
    return %5#0, %5#1 : !quantum.bit, !quantum.bit
}

// -----

// Arbitrary rotations, special case 1: ω1 = ϕ2 = 0.0
//
// Rot(arg0, arg1, 0.0) <--- parent params ϕ1, θ1, and ω1
// Rot(0.0, arg1, arg2) <--- params ϕ2, θ2, and ω2
//
// Expected output: a single Rot { ϕF = ϕ1 = arg0; θF = θ1 + θ2 = 2 * arg1; ωF = ω2 = arg2; }

func.func @test_merge_rotations(%arg0: f64, %arg1: f64, %arg2: f64) -> !quantum.bit {
    %zeroConst = arith.constant 0.000000e+00 : f64

    // CHECK: [[reg:%.+]] = quantum.alloc( 1) : !quantum.reg
    // CHECK: [[qubit:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit

    // CHECK: [[arg1PlusArg1:%.+]] = arith.addf %arg1, %arg1 : f64
    // CHECK: [[ret:%.+]] = quantum.custom "Rot"(%arg0, [[arg1PlusArg1]], %arg2) [[qubit]] : !quantum.bit
    // CHECK-NOT: quantum.custom "Rot"
    %2 = quantum.custom "Rot"(%arg0, %arg1, %zeroConst) %1 : !quantum.bit
    %3 = quantum.custom "Rot"(%zeroConst, %arg1, %arg2) %2 : !quantum.bit

    // CHECK: return [[ret]]
    return %3 : !quantum.bit
}

// -----

// Special case 2a: θ1 = θ2 = 0.0
//
// Rot(arg0, 0.0, arg2) <--- parent params ϕ1, θ1, and ω1
// Rot(arg0, 0.0, arg2) <--- params ϕ2, θ2, and ω2
//
// Expected output: a single Rot { ϕF = ϕ1 + ϕ2 + ω1 + ω2 = arg0 + arg0 + arg1 + arg1; θF = 0.0; ωF = 0.0; }

func.func @test_merge_rotations(%arg0: f64, %arg1: f64) -> !quantum.bit {
    // CHECK: [[zeroConst:%.+]] = arith.constant 0.000000e+00 : f64
    // CHECK: [[reg:%.+]] = quantum.alloc( 1) : !quantum.reg
    // CHECK: [[qubit:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    %zeroConst = arith.constant 0.000000e+00 : f64
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit

    // CHECK-DAG: [[arg0PlusArg0:%.+]] = arith.addf %arg0, %arg0 : f64
    // CHECK-DAG: [[arg1PlusArg1:%.+]] = arith.addf %arg1, %arg1 : f64
    // CHECK: [[arg0sPlusArg1s:%.+]] = arith.addf [[arg0PlusArg0]], [[arg1PlusArg1]] : f64
    // CHECK: [[ret:%.+]] = quantum.custom "Rot"([[arg0sPlusArg1s]], [[zeroConst]], [[zeroConst]]) [[qubit]] : !quantum.bit
    // CHECK-NOT: quantum.custom "Rot"
    %2 = quantum.custom "Rot"(%arg0, %zeroConst, %arg1) %1 : !quantum.bit
    %3 = quantum.custom "Rot"(%arg0, %zeroConst, %arg1) %2 : !quantum.bit

    // CHECK: return [[ret]]
    return %3 : !quantum.bit
}

// -----

// Special case 2b: θ1 = 0.0
//
// Rot(arg0, 0.0, arg2) <--- parent params ϕ1, θ1, and ω1
// Rot(arg0, arg1, arg2) <--- params ϕ2, θ2, and ω2
//
// Expected output: a single Rot { ϕF = ϕ1 + ϕ2 + ω1 = arg0 + arg0 + arg1; θF = θ2 = arg1; ωF = ω2 = arg2; }

func.func @test_merge_rotations(%arg0: f64, %arg1: f64, %arg2: f64) -> !quantum.bit {
    %zeroConst = arith.constant 0.000000e+00 : f64

    // CHECK: [[reg:%.+]] = quantum.alloc( 1) : !quantum.reg
    // CHECK: [[qubit:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit

    // CHECK: [[arg0PlusArg0:%.+]] = arith.addf %arg0, %arg0 : f64
    // CHECK: [[arg0sPlusArg1:%.+]] = arith.addf [[arg0PlusArg0]], %arg1 : f64
    // CHECK: [[ret:%.+]] = quantum.custom "Rot"([[arg0sPlusArg1]], %arg1, %arg2) [[qubit]] : !quantum.bit
    // CHECK-NOT: quantum.custom "Rot"
    %2 = quantum.custom "Rot"(%arg0, %zeroConst, %arg1) %1 : !quantum.bit
    %3 = quantum.custom "Rot"(%arg0, %arg1, %arg2) %2 : !quantum.bit

    // CHECK: return [[ret]]
    return %3 : !quantum.bit
}

// -----

// Special case 2c: θ2 = 0.0
//
// Rot(arg0, arg1, arg2) <--- parent params ϕ1, θ1, and ω1
// Rot(arg0, 0.0, arg2) <--- params ϕ2, θ2, and ω2
//
// Expected output: a single Rot { ϕF = ϕ1 = arg0; θF = θ1 = arg1; ωF = ω1 + ω2 + ϕ2 = arg2 + arg2 + arg0; }

func.func @test_merge_rotations(%arg0: f64, %arg1: f64, %arg2: f64) -> !quantum.bit {
    %zeroConst = arith.constant 0.000000e+00 : f64

    // CHECK: [[reg:%.+]] = quantum.alloc( 1) : !quantum.reg
    // CHECK: [[qubit:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit

    // CHECK: [[arg2PlusArg2:%.+]] = arith.addf %arg2, %arg2 : f64
    // CHECK: [[arg2sPlusArg0:%.+]] = arith.addf [[arg2PlusArg2]], %arg0 : f64
    // CHECK: [[ret:%.+]] = quantum.custom "Rot"(%arg0, %arg1, [[arg2sPlusArg0]]) [[qubit]] : !quantum.bit
    // CHECK-NOT: quantum.custom "Rot"
    %2 = quantum.custom "Rot"(%arg0, %arg1, %arg2) %1 : !quantum.bit
    %3 = quantum.custom "Rot"(%arg0, %zeroConst, %arg2) %2 : !quantum.bit

    // CHECK: return [[ret]]
    return %3 : !quantum.bit
}

// -----

func.func @test_merge_rotations(%arg0: f64, %arg1: f64, %arg2: f64) -> (!quantum.bit, !quantum.bit) {
    // CHECK: [[reg:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[qubit1:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit2:%.+]] = quantum.extract [[reg]][ 1] : !quantum.reg -> !quantum.bit
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    // CHECK: [[theta1:%.+]] = arith.addf %arg1, %arg2 : f64
    // CHECK: [[theta2:%.+]] = arith.addf %arg0, [[theta1]] : f64
    // CHECK: [[ret:%.+]]:2 = quantum.multirz([[theta2]]) [[qubit1]], [[qubit2]] : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.multirz
    %3:2 = quantum.multirz (%arg0) %1, %2 : !quantum.bit, !quantum.bit
    %4:2 = quantum.multirz (%arg1) %3#0, %3#1 : !quantum.bit, !quantum.bit
    %5:2 = quantum.multirz (%arg2) %4#0, %4#1 : !quantum.bit, !quantum.bit
    // CHECK: return [[ret]]#0, [[ret]]#1
    return %5#0, %5#1 : !quantum.bit, !quantum.bit
}

// -----

func.func @test_merge_rotations(%arg0: f64, %arg1: f64, %arg2: f64) -> (!quantum.bit, !quantum.bit) {
    // CHECK: [[reg:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[qubit1:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit2:%.+]] = quantum.extract [[reg]][ 1] : !quantum.reg -> !quantum.bit
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit
    // CHECK-NOT: arith.addf
    // CHECK: quantum.multirz
    // CHECK: quantum.multirz
    // CHECK-NOT: quantum.multirz
    %4:2 = quantum.multirz (%arg0) %1, %2 : !quantum.bit, !quantum.bit
    %5:2 = quantum.multirz (%arg1) %4#0, %3 : !quantum.bit, !quantum.bit
    return %5#0, %5#1 : !quantum.bit, !quantum.bit
}

// -----

func.func @test_merge_rotations(%arg0: f64, %arg1: f64, %arg2: f64) -> (!quantum.bit, !quantum.bit) {
    // CHECK: [[reg:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[qubit1:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit2:%.+]] = quantum.extract [[reg]][ 1] : !quantum.reg -> !quantum.bit
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit
    // CHECK-NOT: arith.addf
    // CHECK: quantum.multirz
    // CHECK: quantum.multirz
    // CHECK-NOT: quantum.multirz
    %4:2 = quantum.multirz (%arg0) %1, %2 : !quantum.bit, !quantum.bit
    %5 = quantum.multirz (%arg1) %4#0 : !quantum.bit
    return %5, %4#1 : !quantum.bit, !quantum.bit
}

// -----

func.func @test_merge_rotations(%arg0: f64) -> (!quantum.bit, !quantum.bit, !quantum.bit) {
    // CHECK: [[true:%.+]] = llvm.mlir.constant
    // CHECK: [[false:%.+]] = llvm.mlir.constant
    %true = llvm.mlir.constant (1 : i1) :i1
    %false = llvm.mlir.constant (0 : i1) :i1

    // CHECK: quantum.alloc
    // CHECK: [[qubit0:%.+]] = quantum.extract {{.+}}[ 0]
    // CHECK: [[qubit1:%.+]] = quantum.extract {{.+}}[ 1]
    // CHECK: [[qubit2:%.+]] = quantum.extract {{.+}}[ 2]
    %reg = quantum.alloc( 4) : !quantum.reg
    %0 = quantum.extract %reg[ 0] : !quantum.reg -> !quantum.bit
    %1 = quantum.extract %reg[ 1] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %reg[ 2] : !quantum.reg -> !quantum.bit

    // CHECK:    [[angle:%.+]] = arith.addf %arg0, %arg0 : f64
    // CHECK:    [[ret:%.+]], [[ctrlret:%.+]]:2 = quantum.custom "RX"([[angle]]) [[qubit0]] ctrls([[qubit1]], [[qubit2]]) ctrlvals([[true]], [[false]]) : !quantum.bit ctrls !quantum.bit, !quantum.bit
    %out_qubits, %out_ctrl_qubits:2 = quantum.custom "RX"(%arg0) %0 ctrls(%1, %2) ctrlvals(%true, %false) : !quantum.bit ctrls !quantum.bit, !quantum.bit
    %out_qubits_1, %out_ctrl_qubits_1:2 = quantum.custom "RX"(%arg0) %out_qubits ctrls(%out_ctrl_qubits#0, %out_ctrl_qubits#1) ctrlvals(%true, %false) : !quantum.bit ctrls !quantum.bit, !quantum.bit

    // CHECK:  return [[ret]], [[ctrlret]]#0, [[ctrlret]]#1
    return %out_qubits_1, %out_ctrl_qubits_1#0, %out_ctrl_qubits_1#1 : !quantum.bit, !quantum.bit, !quantum.bit
}

// -----

func.func @test_merge_rotations(%arg0: f64, %arg1: f64) -> !quantum.bit {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[reg:%.+]] = quantum.alloc( 1) : !quantum.reg
    // CHECK: [[qubit:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[arg0neg:%.+]] = arith.negf %arg0 : f64
    // CHECK: [[arg1neg:%.+]] = arith.negf %arg1 : f64
    // CHECK: [[add:%.+]] = arith.addf [[arg0neg]], [[arg1neg]] : f64
    // CHECK: [[ret:%.+]] = quantum.custom "RX"([[add]]) [[qubit]] : !quantum.bit
    %2 = quantum.custom "RX"(%arg0) %1 {adjoint}: !quantum.bit
    %3 = quantum.custom "RX"(%arg1) %2 {adjoint}: !quantum.bit

    // CHECK:  return [[ret]]
    return %3 : !quantum.bit
}

// -----

func.func @test_merge_rotations(%arg0: f64, %arg1: f64, %arg2: f64) -> (!quantum.bit, !quantum.bit) {
    // CHECK: [[reg:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[qubit1:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit2:%.+]] = quantum.extract [[reg]][ 1] : !quantum.reg -> !quantum.bit
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[arg0neg:%.+]] = arith.negf %arg0 : f64
    // CHECK: [[arg1neg:%.+]] = arith.negf %arg1 : f64
    // CHECK: [[add:%.+]] = arith.addf [[arg0neg]], [[arg1neg]] : f64
    // CHECK: [[ret:%.+]]:2 = quantum.multirz([[add]]) [[qubit1]], [[qubit2]] : !quantum.bit, !quantum.bit
    %3:2 = quantum.multirz (%arg0) %1, %2 {adjoint}: !quantum.bit, !quantum.bit
    %4:2 = quantum.multirz (%arg1) %3#0, %3#1 {adjoint}: !quantum.bit, !quantum.bit
    // CHECK: return [[ret]]#0, [[ret]]#1
    return %4#0, %4#1 : !quantum.bit, !quantum.bit
}

// -----

func.func @test_merge_rotations(%arg0: f64) -> !quantum.bit {
    // CHECK: [[cst:%.+]] = arith.constant 1.000000e-01 : f64
    // CHECK: [[reg:%.+]] = quantum.alloc( 1) : !quantum.reg
    // CHECK: [[qubit:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit

    // CHECK: [[x2:%.+]] = arith.mulf %arg0, %arg0 : f64
    %x2 = arith.mulf %arg0, %arg0 : f64

    // CHECK: [[sum1:%.+]] = arith.addf [[x2]], [[cst]] : f64
    // CHECK: [[sum2:%.+]] = arith.addf %arg0, [[sum1]] : f64
    // CHECK: [[ret:%.+]] = quantum.custom "RX"([[sum2]]) [[qubit]] : !quantum.bit
    %cst = arith.constant 1.000000e-01 : f64
    %2 = quantum.custom "RX"(%arg0) %1 : !quantum.bit
    %3 = quantum.custom "RX"(%cst) %2 : !quantum.bit
    %4 = quantum.custom "RX"(%x2) %3 : !quantum.bit

    // CHECK: return [[ret]]
    return %4 : !quantum.bit
}

// -----

func.func @test_merge_rotations(%arg0: f64, %arg1: i1, %arg2: i1) -> (!quantum.bit, !quantum.bit, !quantum.bit) {
    // CHECK: [[cst:%.+]] = arith.constant 1.000000e-01 : f64
    // CHECK: [[reg:%.+]] = quantum.alloc( 3) : !quantum.reg
    // CHECK: [[qubit:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[ctrl0:%.+]] = quantum.extract [[reg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[ctrl1:%.+]] = quantum.extract [[reg]][ 2] : !quantum.reg -> !quantum.bit
    %reg = quantum.alloc( 3) : !quantum.reg
    %0 = quantum.extract %reg[ 0] : !quantum.reg -> !quantum.bit
    %1 = quantum.extract %reg[ 1] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %reg[ 2] : !quantum.reg -> !quantum.bit

    // CHECK: [[x2:%.+]] = arith.mulf %arg0, %arg0 : f64
    %x2 = arith.mulf %arg0, %arg0 : f64

    // CHECK: [[sum1:%.+]] = arith.addf [[x2]], [[cst]] : f64
    // CHECK: [[sum2:%.+]] = arith.addf %arg0, [[sum1]] : f64
    // CHECK: [[ret:%.+]], [[ctrl_ret:%.+]]:2 = quantum.custom "RX"([[sum2]]) [[qubit]] ctrls([[ctrl0]], [[ctrl1]]) ctrlvals(%arg1, %arg2) : !quantum.bit ctrls !quantum.bit, !quantum.bit
    %cst = arith.constant 1.000000e-01 : f64
    %out_0, %ctrl_out_0:2 = quantum.custom "RX"(%arg0) %0 ctrls(%1, %2) ctrlvals(%arg1, %arg2) : !quantum.bit ctrls !quantum.bit, !quantum.bit
    %out_1, %ctrl_out_1:2 = quantum.custom "RX"(%cst) %out_0 ctrls(%ctrl_out_0#0, %ctrl_out_0#1) ctrlvals(%arg1, %arg2) : !quantum.bit ctrls !quantum.bit, !quantum.bit
    %out_2, %ctrl_out_2:2 = quantum.custom "RX"(%x2) %out_1 ctrls(%ctrl_out_1#0, %ctrl_out_1#1) ctrlvals(%arg1, %arg2) : !quantum.bit ctrls !quantum.bit, !quantum.bit

    // CHECK: return [[ret]], [[ctrl_ret]]#0, [[ctrl_ret]]#1
    return %out_2, %ctrl_out_2#0, %ctrl_out_2#1 : !quantum.bit, !quantum.bit, !quantum.bit
}

// -----

func.func @test_loop_boundary_rotation(%q0: !quantum.bit, %q1: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
    %stop = arith.constant 10 : index
    %one = arith.constant 1 : index
    %theta = arith.constant 0.2 : f64
    %beta = arith.constant 0.3 : f64

    // Quantum circuit:           // Expected output:
    // RX(beta) Q0                // RX(beta+theta) Q0
    //                            // 
    // for _ in range(n)          // for _ in range(n):
    //    RX(theta) Q0      ->    //
    //    X Q1                    //    X Q1
    //    CNOT Q0, Q1             //    CNOT Q0, Q1
    //    RX(theta) Q0            //    RX(theta+theta) Q0
    //    X Q1                    //    X Q1
    //                            // RX(-theta) Q0

    // CHECK-LABEL: func.func @test_loop_boundary_rotation(
    // CHECK-SAME: [[q0:%.+]]: !quantum.bit,
    // CHECK-SAME: [[q1:%.+]]: !quantum.bit
    // CHECK: [[cst:%.+]] = {{.*}} -2.000000e-01 : f64
    // CHECK: [[cst_0:%.+]] = {{.*}} 4.000000e-01 : f64
    // CHECK: [[cst_1:%.+]] = {{.*}} 5.000000e-01 : f64

    // CHECK: [[qubit_0:%.+]] = {{.*}} "RX"([[cst_1]]) [[q0]]
    %q_00 = quantum.custom "RX"(%beta) %q0 : !quantum.bit

    // CHECK: [[scf:%.+]]:2 = scf.for {{.*}} iter_args([[arg0:%.+]] = [[qubit_0]], [[arg1:%.+]] = [[q1]])
    %scf:2 = scf.for %i = %one to %stop step %one iter_args(%q_arg0 = %q_00, %q_arg1 = %q1) -> (!quantum.bit, !quantum.bit) {
        // CHECK-NOT: "RX"
        %q_0 = quantum.custom "RX"(%theta) %q_arg0 : !quantum.bit
        // CHECK: [[qubit_1:%.+]] = {{.*}} "X"() [[arg1]]
        %q_1 = quantum.custom "X"() %q_arg1 : !quantum.bit
        // CHECK: [[qubit_2:%.+]]:2 = {{.*}} "CNOT"() [[arg0]], [[qubit_1]]
        %q_2:2 = quantum.custom "CNOT"() %q_0, %q_1 : !quantum.bit, !quantum.bit
        // CHECK: [[qubit_3:%.+]] = {{.*}} "RX"([[cst_0]]) [[qubit_2]]#0
        %q_3 = quantum.custom "RX"(%theta) %q_2#0 : !quantum.bit
        // CHECK: "X"
        %q_4 = quantum.custom "X"() %q_2#1 : !quantum.bit
        scf.yield %q_3, %q_4 : !quantum.bit, !quantum.bit
    }

    // CHECK: [[qubit_6:%.+]] = {{.*}} "RX"([[cst]]) [[scf]]#0
    func.return %scf#0, %scf#1 : !quantum.bit, !quantum.bit
}

// -----

// Negative tests, case 1: fixed single rotations of different names don't merge

func.func @test_merge_rotations(%arg0: f64, %arg1: f64) -> !quantum.bit {
    // CHECK: [[reg:%.+]] = quantum.alloc( 1) : !quantum.reg
    // CHECK: [[qubit:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit

    // CHECK: [[ret:%.+]] = quantum.custom "RX"(%arg0) [[qubit]] : !quantum.bit
    // CHECK: [[ret_0:%.+]] = quantum.custom "RZ"(%arg1) [[ret]] : !quantum.bit
    %2 = quantum.custom "RX"(%arg0) %1 : !quantum.bit
    %3 = quantum.custom "RZ"(%arg1) %2 : !quantum.bit

    // CHECK: return [[ret_0]]
    return %3 : !quantum.bit
}

// -----

// Negative tests, case 2: fixed single rotations and arbitrary rotations don't merge

func.func @test_merge_rotations(%arg0: f64, %arg1: f64) -> !quantum.bit {
    // CHECK: [[reg:%.+]] = quantum.alloc( 1) : !quantum.reg
    // CHECK: [[qubit:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit

    // CHECK: [[ret:%.+]] = quantum.custom "RX"(%arg0) [[qubit]] : !quantum.bit
    // CHECK: [[ret_0:%.+]] = quantum.custom "Rot"(%arg1) [[ret]] : !quantum.bit
    %2 = quantum.custom "RX"(%arg0) %1 : !quantum.bit
    %3 = quantum.custom "Rot"(%arg1) %2 : !quantum.bit

    // CHECK: return [[ret_0]]
    return %3 : !quantum.bit
}

// -----

// Negative tests, case 3: MultiRZ and CRot don't merge

func.func @test_merge_rotations(%arg0: f64, %arg1: f64) -> (!quantum.bit, !quantum.bit) {
    // CHECK: [[reg:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[qubit1:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit2:%.+]] = quantum.extract [[reg]][ 1] : !quantum.reg -> !quantum.bit
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    // CHECK: [[ret:%.+]]:2 = quantum.multirz(%arg0) [[qubit1]], [[qubit2]] : !quantum.bit, !quantum.bit
    // CHECK: [[ret_0:%.+]]:2 = quantum.custom "CRot"(%arg1) [[ret]]#0, [[ret]]#1 : !quantum.bit, !quantum.bit
    %3:2 = quantum.multirz (%arg0) %1, %2 : !quantum.bit, !quantum.bit
    %4:2 = quantum.custom "CRot"(%arg1) %3#0, %3#1 : !quantum.bit, !quantum.bit

    // CHECK: return [[ret_0]]#0, [[ret_0]]#1
    return %4#0, %4#1 : !quantum.bit, !quantum.bit
}

// PPR Tests

// -----

// simple PPR merge

// CHECK-LABEL: merge_Y_pi_4
func.func public @merge_Y_pi_4(%q1: !quantum.bit) {
    // CHECK-NOT: qec.ppr ["Y"](4)
    // CHECK: qec.ppr ["Y"](2)
    %0 = qec.ppr ["Y"](4) %q1: !quantum.bit
    %1 = qec.ppr ["Y"](4) %0: !quantum.bit
    func.return
}

// -----

// merging to identity

func.func public @test_merge_rotations(%q1: !quantum.bit) {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["X"](2) %q1: !quantum.bit
    %1 = qec.ppr ["X"](2) %0: !quantum.bit
    func.return
}

// -----

// multiple merges

// CHECK-LABEL: merge_multi_Y
func.func public @merge_multi_Y(%q1: !quantum.bit) {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["Y"](4) %q1: !quantum.bit
    %1 = qec.ppr ["Y"](4) %0: !quantum.bit
    %2 = qec.ppr ["Y"](8) %1: !quantum.bit
    %3 = qec.ppr ["Y"](8) %2: !quantum.bit
    %4 = qec.ppr ["Y"](8) %3: !quantum.bit
    %5 = qec.ppr ["Y"](8) %4: !quantum.bit
    func.return
}

// -----

// not merging when incompatible

// CHECK-LABEL: dont_merge
func.func public @dont_merge(%q1: !quantum.bit) {
    // CHECK: qec.ppr ["Y"](4)
    // CHECK: qec.ppr ["X"](4)
    // CHECK: qec.ppr ["Z"](8)
    // CHECK: qec.ppr ["X"](8)
    // CHECK: qec.ppr ["Y"](2)
    // CHECK: qec.ppr ["Z"](2)
    %0 = qec.ppr ["Y"](4) %q1: !quantum.bit
    %1 = qec.ppr ["X"](4) %0: !quantum.bit
    %2 = qec.ppr ["Z"](8) %1: !quantum.bit
    %3 = qec.ppr ["X"](8) %2: !quantum.bit
    %4 = qec.ppr ["Y"](2) %3: !quantum.bit
    %5 = qec.ppr ["Z"](2) %4: !quantum.bit
    func.return
}

// -----

// updating references

// CHECK-LABEL: merge_correct_references
func.func public @merge_correct_references(%q1: !quantum.bit) {
    // CHECK: [[in:%.+]] = qec.ppr ["Y"](4)
    // CHECK: [[merge_out:%.+]] = qec.ppr ["X"](2) [[in]]
    // CHECK: qec.ppr ["Z"](4) [[merge_out]]
    %0 = qec.ppr ["Y"](4) %q1: !quantum.bit
    %1 = qec.ppr ["X"](4) %0: !quantum.bit
    %2 = qec.ppr ["X"](4) %1: !quantum.bit
    %3 = qec.ppr ["Z"](4) %2: !quantum.bit
    func.return
}

// -----

// multi-qubit merge

// CHECK-LABEL: merge_multi_XYZ
func.func public @merge_multi_XYZ(%q1: !quantum.bit, %q2: !quantum.bit, %q3: !quantum.bit) {
    // CHECK: qec.ppr ["X", "Y", "Z"](4)
    // CHECK-NOT: qec.ppr ["X", "Y", "Z"](2)
    // CHECK-NOT: qec.ppr ["X", "Y", "Z"](8)
    %0:3 = qec.ppr ["X", "Y", "Z"](4) %q1, %q2, %q3: !quantum.bit, !quantum.bit, !quantum.bit
    %1:3 = qec.ppr ["X", "Y", "Z"](4) %0#0, %0#1, %0#2: !quantum.bit, !quantum.bit, !quantum.bit
    %2:3 = qec.ppr ["X", "Y", "Z"](8) %1#0, %1#1, %1#2: !quantum.bit, !quantum.bit, !quantum.bit
    %3:3 = qec.ppr ["X", "Y", "Z"](8) %2#0, %2#1, %2#2: !quantum.bit, !quantum.bit, !quantum.bit
    %4:3 = qec.ppr ["X", "Y", "Z"](2) %3#0, %3#1, %3#2: !quantum.bit, !quantum.bit, !quantum.bit
    func.return
}

// -----

// simple inverse PPRs

// CHECK-LABEL: cancel_Z_pi_2
func.func public @cancel_Z_pi_2(%q1: !quantum.bit) {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["Z"](2) %q1: !quantum.bit
    %1 = qec.ppr ["Z"](-2) %0: !quantum.bit
    func.return
}

// -----

// multiple inverse PPRs

// CHECK-LABEL: cancel_multi_Z
func.func public @cancel_multi_Z(%q1: !quantum.bit) {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["Z"](8) %q1: !quantum.bit
    %1 = qec.ppr ["X"](-4) %0: !quantum.bit
    %2 = qec.ppr ["X"](4) %1: !quantum.bit
    %3 = qec.ppr ["Z"](-8) %2: !quantum.bit
    func.return
}

// -----

// incorrect order inverse PPRs

// CHECK-LABEL: dont_cancel
func.func public @dont_cancel(%q1: !quantum.bit) {
    // CHECK: qec.ppr ["Z"](8)
    // CHECK: qec.ppr ["X"](-4)
    // CHECK: qec.ppr ["Z"](-8)
    // CHECK: qec.ppr ["X"](4)
    %0 = qec.ppr ["Z"](8) %q1: !quantum.bit
    %1 = qec.ppr ["X"](-4) %0: !quantum.bit
    %2 = qec.ppr ["Z"](-8) %1: !quantum.bit
    %3 = qec.ppr ["X"](4) %2: !quantum.bit
    func.return
}

// -----

// merge through ops on other wires

// CHECK-LABEL: merge_through
func.func public @merge_through(%q1: !quantum.bit, %q2: !quantum.bit) -> !quantum.bit {
    // CHECK-NOT: qec.ppr ["X"](4)
    // CHECK-DAG: quantum.custom
    // CHECK-DAG: qec.ppr ["X"](2)
    %0 = qec.ppr ["X"](4) %q1: !quantum.bit
    %1 = quantum.custom "Hadamard"() %q2: !quantum.bit
    %2 = qec.ppr ["X"](4) %0: !quantum.bit
    func.return %1: !quantum.bit
}

// ----- 

// correct reference updates on cancel

// CHECK-LABEL: cancel_correct_references
func.func public @cancel_correct_references(%q1: !quantum.bit) {
    // CHECK: [[in:%.+]] = qec.ppr ["Y"](2)
    // CHECK: qec.ppr ["Z"](4) [[in]]
    %0 = qec.ppr ["Y"](2) %q1: !quantum.bit
    %1 = qec.ppr ["X"](8) %0: !quantum.bit
    %2 = qec.ppr ["X"](-8) %1: !quantum.bit
    %3 = qec.ppr ["Z"](4) %2: !quantum.bit
    func.return
}

// -----

// multi-qubit inverse PPRs

// CHECK-LABEL: cancel_multi_XY
func.func public @cancel_multi_XY(%q1: !quantum.bit, %q2: !quantum.bit) {
    // CHECK-NOT: qec.ppr
    %0:2 = qec.ppr ["X", "Y"](2) %q1, %q2: !quantum.bit, !quantum.bit
    %1:2 = qec.ppr ["X", "Y"](-2) %0#0, %0#1: !quantum.bit, !quantum.bit
    %2:2 = qec.ppr ["X", "Y"](4) %1#0, %1#1: !quantum.bit, !quantum.bit
    %3:2 = qec.ppr ["X", "Y"](-4) %2#0, %2#1: !quantum.bit, !quantum.bit
    func.return
}

// -----

// multi-qubit incompatible inverse PPRs

// CHECK-LABEL: dont_cancel_multi
func.func public @dont_cancel_multi(%q1: !quantum.bit, %q2: !quantum.bit) {
    // CHECK: qec.ppr ["X", "Y"](2)
    // CHECK: qec.ppr ["Y", "X"](-2)
    // CHECK: qec.ppr ["Z", "Y"](4)
    // CHECK: qec.ppr ["Y", "Z"](-4)
    %0:2 = qec.ppr ["X", "Y"](2) %q1, %q2: !quantum.bit, !quantum.bit
    %1:2 = qec.ppr ["Y", "X"](-2) %0#0, %0#1: !quantum.bit, !quantum.bit
    %2:2 = qec.ppr ["Z", "Y"](4) %1#0, %1#1: !quantum.bit, !quantum.bit
    %3:2 = qec.ppr ["Y", "Z"](-4) %2#0, %2#1: !quantum.bit, !quantum.bit
    func.return
}

// -----

// merge and cancel PPRs

// CHECK-LABEL: multi_cancel_and_merge
func.func public @multi_cancel_and_merge(%q1: !quantum.bit, %q2: !quantum.bit) {
    // CHECK-NOT: qec.ppr ["X", "Y"]
    // CHECK: qec.ppr ["Y", "Z"](4)
    %0:2 = qec.ppr ["Y", "Z"](8) %q1, %q2: !quantum.bit, !quantum.bit
    %1:2 = qec.ppr ["X", "Y"](-4) %0#0, %0#1: !quantum.bit, !quantum.bit
    %2:2 = qec.ppr ["X", "Y"](4) %1#0, %1#1: !quantum.bit, !quantum.bit
    %3:2 = qec.ppr ["Y", "Z"](8) %2#0, %2#1: !quantum.bit, !quantum.bit
    func.return
}

// -----

// cancel and merge PPRs

// CHECK-LABEL: multi_merge_and_cancel
func.func public @multi_merge_and_cancel(%q1: !quantum.bit, %q2: !quantum.bit) {
    // CHECK-NOT: qec.ppr
    %0:2 = qec.ppr ["Z", "X"](4) %q1, %q2: !quantum.bit, !quantum.bit
    %1:2 = qec.ppr ["Z", "X"](-8) %0#0, %0#1: !quantum.bit, !quantum.bit
    %2:2 = qec.ppr ["Z", "X"](-8) %1#0, %1#1: !quantum.bit, !quantum.bit
    func.return
}

// -----

// don't merge through other operations

// CHECK-LABEL: mixed_operations
func.func public @mixed_operations(%q1: !quantum.bit) {
    // CHECK: qec.ppr ["Z"](4)
    // CHECK: quantum.custom "Hadamard"()
    // CHECK: qec.ppr ["Z"](4)
    %0 = qec.ppr ["Z"](4) %q1: !quantum.bit
    %1 = quantum.custom "Hadamard"() %0: !quantum.bit
    %2 = qec.ppr ["Z"](4) %1: !quantum.bit
    func.return
}

// -----

// don't merge if only one qubit matches

// CHECK-LABEL: half_compatible_qubits
func.func public @half_compatible_qubits(%q1: !quantum.bit, %q2: !quantum.bit, %q3: !quantum.bit) {
    // CHECK: qec.ppr ["X", "X"](4)
    // CHECK: qec.ppr ["X", "X"](4)
    %0, %1 = qec.ppr ["X", "X"](4) %q1, %q2: !quantum.bit, !quantum.bit
    %2:2 = qec.ppr ["X", "X"](4) %0, %q3: !quantum.bit, !quantum.bit
    func.return
}

// -----

// merge correctly when conditionals are present

// CHECK-LABEL: merge_conditionals
func.func public @merge_conditionals(%q1: !quantum.bit, %q2: !quantum.bit, %arg0: i1) {
    // CHECK: qec.ppr ["X", "Z"](4) {{%.+}}, {{%.+}} cond({{%.+}})
    // CHECK-NOT: qec.ppr ["X", "Z"](8)
    %0, %1 = qec.ppr ["X", "Z"](8) %q1, %q2 cond(%arg0): !quantum.bit, !quantum.bit
    %2, %3 = qec.ppr ["X", "Z"](8) %0, %1 cond(%arg0): !quantum.bit, !quantum.bit
    func.return
}

// -----

// don't merge compatible Pauli words when conditionals differ

// CHECK-LABEL: dont_merge_incompatible_conditionals
func.func public @dont_merge_incompatible_conditionals(%q1: !quantum.bit, %q2: !quantum.bit, %arg0: i1, %arg1: i1) {
    // CHECK-NOT: qec.ppr ["X", "Z"](4)
    // CHECK: [[in:%.+]]:2 = qec.ppr ["X", "Z"](8) 
    // CHECK: qec.ppr ["X", "Z"](8) [[in]]#0, [[in]]#1
    %0:2 = qec.ppr ["X", "Z"](8) %q1, %q2 cond(%arg0): !quantum.bit, !quantum.bit
    %1:2 = qec.ppr ["X", "Z"](8) %0#0, %0#1 cond(%arg1): !quantum.bit, !quantum.bit
    func.return
}

// -----

// don't merge PPRs with conditional PPRs

// CHECK-LABEL: dont_merge_conditionals
func.func public @dont_merge_conditionals(%q1: !quantum.bit, %q2: !quantum.bit, %arg0: i1) {
    // CHECK-NOT: qec.ppr ["X", "Z"](4)
    // CHECK: [[in:%.+]]:2 = qec.ppr ["X", "Z"](8)
    // CHECK: qec.ppr ["X", "Z"](8) [[in]]#0, [[in]]#1
    %0:2 = qec.ppr ["X", "Z"](8) %q1, %q2: !quantum.bit, !quantum.bit
    %1:2 = qec.ppr ["X", "Z"](8) %0#0, %0#1 cond(%arg0): !quantum.bit, !quantum.bit
    func.return
}

// -----

// re-arranging qubits is ok as long as the pauli words are re-arranged too

// CHECK-LABEL: merge_permutations
func.func public @merge_permutations(%z0: !quantum.bit, %y0: !quantum.bit) {
    // CHECK: ([[zIn:%.+]]: !quantum.bit, [[yIn:%.+]]: !quantum.bit)
    // CHECK-NOT: qec.ppr ["Z", "Y"]
    // CHECK-NOT: qec.ppr ["Y", "Z"](4)
    // CHECK: qec.ppr ["Y", "Z"](2) [[yIn]], [[zIn]]
    %z1, %y1 = qec.ppr ["Z", "Y"](4) %z0, %y0: !quantum.bit, !quantum.bit
    %y2, %z2 = qec.ppr ["Y", "Z"](4) %y1, %z1: !quantum.bit, !quantum.bit
    func.return
}

// -----

// check permutations with duplicate Pauli symbols

// CHECK-LABEL: merge_permutations_with_duplicates
func.func public @merge_permutations_with_duplicates(%q0: !quantum.bit, %q1: !quantum.bit, %q2: !quantum.bit) {
    // CHECK: ([[q0:%.+]]: !quantum.bit, [[q1:%.+]]: !quantum.bit, [[q2:%.+]]: !quantum.bit)
    // CHECK-NOT: qec.ppr ["X", "Y", "X"]
    // CHECK-NOT: qec.ppr ["Y", "X", "X"](8)
    // CHECK: qec.ppr ["Y", "X", "X"](4) [[q1]], [[q2]], [[q0]]
    %3:3 = qec.ppr ["X", "Y", "X"](8) %q0, %q1, %q2: !quantum.bit, !quantum.bit, !quantum.bit
    %4:3 = qec.ppr ["Y", "X", "X"](8) %3#1, %3#2, %3#0: !quantum.bit, !quantum.bit, !quantum.bit
    func.return
}

// -----

// re-arranging qubits without re-arranging the Pauli word is NOT okay

// CHECK-LABEL: dont_merge_permutations_qubits
func.func public @dont_merge_permutations_qubits(%q0: !quantum.bit, %q1: !quantum.bit) {
    // CHECK: qec.ppr ["Y", "X"](8)
    // CHECK: qec.ppr ["Y", "X"](8)
    %2:2 = qec.ppr ["Y", "X"](8) %q0, %q1: !quantum.bit, !quantum.bit
    %3:2 = qec.ppr ["Y", "X"](8) %2#1, %2#0: !quantum.bit, !quantum.bit
    func.return
}

// -----

// re-arranging Pauli word without re-arranging qubits is not okay

// CHECK-LABEL: dont_merge_permutations_pauli
func.func public @dont_merge_permutations_pauli(%q0: !quantum.bit, %q1: !quantum.bit) {
    // CHECK: qec.ppr ["Z", "Y"](2)
    // CHECK: qec.ppr ["Y", "Z"](2)
    %2:2 = qec.ppr ["Z", "Y"](2) %q0, %q1: !quantum.bit, !quantum.bit
    %3:2 = qec.ppr ["Y", "Z"](2) %2#0, %2#1: !quantum.bit, !quantum.bit
    func.return
}

// -----

// ensure correct result assignments

// CHECK-LABEL: permutation_results
func.func public @permutation_results(%q0: !quantum.bit, %q1: !quantum.bit, %q2: !quantum.bit) -> (!quantum.bit, !quantum.bit, !quantum.bit) {
    // CHECK: [[result:%.+]]:3 = qec.ppr ["Z", "X", "Y"](2)
    // CHECK: return [[result]]#1, [[result]]#0, [[result]]#2
    %0:3 = qec.ppr ["X", "Z", "Y"](4) %q0, %q1, %q2: !quantum.bit, !quantum.bit, !quantum.bit
    %1:3 = qec.ppr ["Z", "X", "Y"](4) %0#1, %0#0, %0#2: !quantum.bit, !quantum.bit, !quantum.bit
    func.return %1#1, %1#0, %1#2: !quantum.bit, !quantum.bit, !quantum.bit
}

// -----

// ignore identity qubits when considering equivalence

// CHECK-LABEL: permute_ignore_identity_parent_op
func.func public @permute_ignore_identity_parent_op(%q0: !quantum.bit, %q1: !quantum.bit, %q2: !quantum.bit, %q3: !quantum.bit) {
    // CHECK: ([[q0:%.+]]: !quantum.bit, [[q1:%.+]]: !quantum.bit, [[q2:%.+]]: !quantum.bit, [[q3:%.+]])
    // CHECK: qec.ppr ["Z", "X"](2) [[q2]], [[q0]]
    %0:3 = qec.ppr ["X", "I", "Z"](4) %q0, %q1, %q2: !quantum.bit, !quantum.bit, !quantum.bit
    %1:3 = qec.ppr ["I", "Z", "X"](4) %q3, %0#2, %0#0: !quantum.bit, !quantum.bit, !quantum.bit
    func.return
}

// -----

// verify identity qubits are passed through

// CHECK-LABEL: pass_identity_qubits
func.func public @pass_identity_qubits(%q0: !quantum.bit, %q1: !quantum.bit) -> !quantum.bit {
    // CHECK: ([[q0:%.+]]: !quantum.bit, [[q1:%.+]]: !quantum.bit)
    // CHECK: return [[q0]]
    %0:2 = qec.ppr ["I", "X"](4) %q0, %q1: !quantum.bit, !quantum.bit
    %1:2 = qec.ppr ["X", "I"](4) %0#1, %0#0: !quantum.bit, !quantum.bit
    func.return %1#1: !quantum.bit
}

// -----

// merge different size pprs when removing identities makes them compatible

// CHECK-LABEL: identity_agnostic_sizing
func.func public @identity_agnostic_sizing(%q0: !quantum.bit, %q1: !quantum.bit, %q2: !quantum.bit) {
    // CHECK: ([[q0:%.+]]: !quantum.bit, [[q1:%.+]]: !quantum.bit, [[q2:%.+]]: !quantum.bit)
    // CHECK: qec.ppr ["Y", "Z"](4) [[q2]], [[q0]]
    %0:3 = qec.ppr ["Z", "I", "Y"](8) %q0, %q1, %q2: !quantum.bit, !quantum.bit, !quantum.bit
    %1:2 = qec.ppr ["Y", "Z"](8) %0#2, %0#0: !quantum.bit, !quantum.bit
    func.return
}

func.func public @dont_merge_superset(%q0: !quantum.bit, %q1: !quantum.bit) {
    // CHECK: qec.ppr ["X", "Y"](4)
    // CHECK: qec.ppr ["X"](4)
    %0:2 = qec.ppr ["X", "Y"](4) %q0, %q1 : !quantum.bit, !quantum.bit
    %1 = qec.ppr ["X"](4) %0#0 : !quantum.bit

    func.return
}


// Arbitrary Angle PPR Tests

// simple merge

// CHECK-LABEL: merge_Y
func.func public @merge_Y(%q0: !quantum.bit, %0: f64, %1: f64) {
    // CHECK: [[angle:%.+]] = arith.addf
    // CHECK: qec.ppr.arbitrary ["Y"]([[angle]])
    %2 = qec.ppr.arbitrary ["Y"](%0) %q0: !quantum.bit
    %3 = qec.ppr.arbitrary ["Y"](%1) %2: !quantum.bit
    func.return
}

// -----

// multiple merges

// CHECK-LABEL: merge_multi_Z
func.func public @merge_multi_Z(%q0: !quantum.bit, %0: f64, %1: f64, %2: f64) {
    // CHECK: [[angle:%.+]] = arith.addf
    // CHECK: [[angle2:%.+]] = arith.addf
    // CHECK: qec.ppr.arbitrary ["Z"]([[angle2]])
    // CHECK-NOT: qec.ppr.arbitrary
    %3 = qec.ppr.arbitrary ["Z"](%0) %q0: !quantum.bit
    %4 = qec.ppr.arbitrary ["Z"](%1) %3: !quantum.bit
    %5 = qec.ppr.arbitrary ["Z"](%2) %4: !quantum.bit
    func.return
}

// -----

// not merging when incompatible

// CHECK-LABEL: dont_merge
func.func public @dont_merge(%q0: !quantum.bit, %q1: !quantum.bit, %0: f64, %1: f64, %2: f64, %3: f64, %4: f64, %5: f64) {
    // CHECK-NOT: arith.addf
    // CHECK: qec.ppr.arbitrary ["Z", "X"]
    // CHECK: qec.ppr.arbitrary ["Y", "X"]
    // CHECK: qec.ppr.arbitrary ["Y", "Z"]
    // CHECK: qec.ppr.arbitrary ["X", "Z"]
    // CHECK: qec.ppr.arbitrary ["X", "Y"]
    // CHECK: qec.ppr.arbitrary ["Z", "Y"]
    %6:2 = qec.ppr.arbitrary ["Z", "X"](%0) %q0, %q1: !quantum.bit, !quantum.bit
    %7:2 = qec.ppr.arbitrary ["Y", "X"](%1) %6#0, %6#1: !quantum.bit, !quantum.bit
    %8:2 = qec.ppr.arbitrary ["Y", "Z"](%2) %7#0, %7#1: !quantum.bit, !quantum.bit
    %9:2 = qec.ppr.arbitrary ["X", "Z"](%3) %8#0, %8#1: !quantum.bit, !quantum.bit
    %10:2 = qec.ppr.arbitrary ["X", "Y"](%4) %9#0, %9#1: !quantum.bit, !quantum.bit
    %11:2 = qec.ppr.arbitrary ["Z", "Y"](%5) %10#0, %10#1: !quantum.bit, !quantum.bit
    func.return
}

// -----

// updating references

// CHECK-LABEL: merge_correct_references
func.func public @merge_correct_references(%q0: !quantum.bit, %0: f64, %1: f64, %2: f64, %3: f64) {
    // CHECK-DAG: [[angle:%.+]] = arith.addf
    // CHECK-DAG: [[in:%.+]] = qec.ppr.arbitrary ["X"]
    // CHECK: [[out:%.+]] = qec.ppr.arbitrary ["Z"]([[angle]]) [[in]]
    // CHECK: qec.ppr.arbitrary ["Y"]({{%.+}}) [[out]]
    %4 = qec.ppr.arbitrary ["X"](%0) %q0: !quantum.bit
    %5 = qec.ppr.arbitrary ["Z"](%1) %4: !quantum.bit
    %6 = qec.ppr.arbitrary ["Z"](%2) %5: !quantum.bit
    %7 = qec.ppr.arbitrary ["Y"](%3) %6: !quantum.bit
    func.return
}

// -----

// multi-qubit merge

// CHECK-LABEL: merge_multi_XZY
func.func public @merge_multi_XZY(%q0: !quantum.bit, %q1: !quantum.bit, %q2: !quantum.bit, %0: f64, %1: f64, %2: f64) {
    // CHECK: [[angle1:%.+]] = arith.addf
    // CHECK: [[angle2:%.+]] = arith.addf
    // CHECK: qec.ppr.arbitrary ["X", "Z", "Y"]([[angle2]])
    %3:3 = qec.ppr.arbitrary ["X", "Z", "Y"](%0) %q0, %q1, %q2: !quantum.bit, !quantum.bit, !quantum.bit
    %4:3 = qec.ppr.arbitrary ["X", "Z", "Y"](%1) %3#0, %3#1, %3#2: !quantum.bit, !quantum.bit, !quantum.bit
    %5:3 = qec.ppr.arbitrary ["X", "Z", "Y"](%2) %4#0, %4#1, %4#2: !quantum.bit, !quantum.bit, !quantum.bit
    func.return
}

// -----

// merge through other ops

// CHECK-LABEL: merge_through
func.func public @merge_through(%q0: !quantum.bit, %q1: !quantum.bit, %0: f64, %1: f64) -> !quantum.bit {
    // CHECK-DAG: [[angle:%.+]] = arith.addf
    // CHECK-DAG: quantum.custom
    // CHECK-DAG: qec.ppr.arbitrary ["X"]([[angle]])
    %2 = qec.ppr.arbitrary ["X"](%0) %q0: !quantum.bit
    %3 = quantum.custom "Hadamard"() %q1: !quantum.bit
    %4 = qec.ppr.arbitrary ["X"](%1) %2: !quantum.bit
    func.return %3: !quantum.bit
}

// ----- 

// don't merge through other operations

// CHECK-LABEL: mixed_operations
func.func public @mixed_operations(%q0: !quantum.bit, %q1: !quantum.bit, %0: f64, %1: f64) {
    // CHECK-NOT: arith.addf
    // CHECK: qec.ppr.arbitrary ["Z", "X"]
    // CHECK: quantum.custom
    // CHECK: qec.ppr.arbitrary ["Z", "X"]
    %2:2 = qec.ppr.arbitrary ["Z", "X"](%0) %q0, %q1: !quantum.bit, !quantum.bit
    %3 = quantum.custom "Hadamard"() %2#1: !quantum.bit
    %5:2 = qec.ppr.arbitrary ["Z", "X"](%1) %2#0, %3: !quantum.bit, !quantum.bit
    func.return
}

// -----

// don't merge if only one qubit matches

// CHECK-LABEL: half_compatible_qubits
func.func public @half_compatible_qubits(%q0: !quantum.bit, %q1: !quantum.bit, %q2: !quantum.bit, %0: f64, %1: f64) {
    // CHECK: qec.ppr.arbitrary ["X", "Z"]
    %2:2 = qec.ppr.arbitrary ["X", "Z"](%0) %q0, %q1: !quantum.bit, !quantum.bit
    %3:2 = qec.ppr.arbitrary ["X", "Z"](%1) %q2, %2#1 : !quantum.bit, !quantum.bit
    func.return
}

// -----

// re-arranging qubits is ok as long as the pauli words are re-arranged too

// CHECK-LABEL: merge_permutations
func.func public @merge_permutations(%z0: !quantum.bit, %y0: !quantum.bit, %0: f64, %1: f64) {
    // CHECK-DAG: [[angle:%.+]] = arith.addf
    // CHECK: qec.ppr.arbitrary ["Y", "Z"]([[angle]]) %arg1, %arg0
    %z1, %y1 = qec.ppr.arbitrary ["Z", "Y"](%0) %z0, %y0: !quantum.bit, !quantum.bit
    %y2, %z2 = qec.ppr.arbitrary ["Y", "Z"](%1) %y1, %z1: !quantum.bit, !quantum.bit
    func.return
}

// -----

// check permutations with duplicate Pauli symbols

// CHECK-LABEL: merge_permutations_with_duplicates
func.func public @merge_permutations_with_duplicates(%q0: !quantum.bit, %q1: !quantum.bit, %q2: !quantum.bit, %0: f64, %1: f64) {
    // CHECK: [[angle:%.+]] = arith.addf
    // CHECK: qec.ppr.arbitrary ["Y", "X", "X"]([[angle]]) %arg1, %arg2, %arg0
    %3:3 = qec.ppr.arbitrary ["X", "Y", "X"](%0) %q0, %q1, %q2: !quantum.bit, !quantum.bit, !quantum.bit
    %4:3 = qec.ppr.arbitrary ["Y", "X", "X"](%1) %3#1, %3#2, %3#0: !quantum.bit, !quantum.bit, !quantum.bit
    func.return
}

// -----

// re-arranging qubits without re-arranging the Pauli word is NOT okay

// CHECK-LABEL: dont_merge_permutations_qubits
func.func public @dont_merge_permutations_qubits(%q0: !quantum.bit, %q1: !quantum.bit, %0: f64, %1: f64) {
    // CHECK: qec.ppr.arbitrary ["Y", "X"]
    // CHECK: qec.ppr.arbitrary ["Y", "X"]
    %2:2 = qec.ppr.arbitrary ["Y", "X"](%0) %q0, %q1: !quantum.bit, !quantum.bit
    %3:2 = qec.ppr.arbitrary ["Y", "X"](%1) %2#1, %2#0: !quantum.bit, !quantum.bit
    func.return
}

// -----

// re-arranging Pauli word without re-arranging qubits is not okay

// CHECK-LABEL: dont_merge_permutations_pauli
func.func public @dont_merge_permutations_pauli(%q0: !quantum.bit, %q1: !quantum.bit, %0: f64, %1: f64) {
    // CHECK: qec.ppr.arbitrary ["Z", "Y"]
    // CHECK: qec.ppr.arbitrary ["Y", "Z"]
    %2:2 = qec.ppr.arbitrary ["Z", "Y"](%0) %q0, %q1: !quantum.bit, !quantum.bit
    %3:2 = qec.ppr.arbitrary ["Y", "Z"](%1) %2#0, %2#1: !quantum.bit, !quantum.bit
    return
}

// -----

// ignore identity qubits when considering equivalence

// CHECK-LABEL: permute_ignore_identity
func.func public @permute_ignore_identity(%q0: !quantum.bit, %q1: !quantum.bit, %q2: !quantum.bit) {
    // CHECK: ([[q0:%.+]]: !quantum.bit, [[q1:%.+]]: !quantum.bit, [[q2:%.+]]: !quantum.bit)
    // CHECK: [[angle:%.+]] = arith.constant 5.0
    // CHECK: qec.ppr.arbitrary ["Z", "X"]([[angle]]) [[q2]], [[q0]]
    %0 = arith.constant 0.8 : f64
    %1 = arith.constant 4.2 : f64
    %2:3 = qec.ppr.arbitrary ["X", "I", "Z"](%0) %q0, %q1, %q2: !quantum.bit, !quantum.bit, !quantum.bit
    %3:3 = qec.ppr.arbitrary ["I", "Z", "X"](%1) %2#1, %2#2, %2#0: !quantum.bit, !quantum.bit, !quantum.bit
    func.return
}

// -----

// merge different size pprs when removing identities makes them compatible

// CHECK-LABEL: identity_agnostic_sizing
func.func public @identity_agnostic_sizing(%q0: !quantum.bit, %q1: !quantum.bit, %q2: !quantum.bit) {
    // CHECK: ([[q0:%.+]]: !quantum.bit, [[q1:%.+]]: !quantum.bit, [[q2:%.+]]: !quantum.bit)
    // CHECK: [[angle:%.+]] = arith.constant
    // CHECK: qec.ppr.arbitrary ["Y", "Z"]([[angle]]) [[q2]], [[q0]]
    %0 = arith.constant 0.2 : f64    
    %1 = arith.constant 0.7 : f64
    %2:3 = qec.ppr.arbitrary ["Z", "I", "Y"](%0) %q0, %q1, %q2: !quantum.bit, !quantum.bit, !quantum.bit
    %3:2 = qec.ppr.arbitrary ["Y", "Z"](%1) %2#2, %2#0: !quantum.bit, !quantum.bit
    func.return
}

// -----

// check equivalent conditions are merged

// CHECK-LABEL: merge_condition
func.func public @merge_condition(%q0: !quantum.bit, %0: f64, %1: f64, %b0: i1) {
    // CHECK: [[angle:%.+]] = arith.addf
    // CHECK: qec.ppr.arbitrary ["X"]([[angle]]) {{%.+}} cond({{%.+}})
    %2 = qec.ppr.arbitrary ["X"](%0) %q0 cond(%b0): !quantum.bit
    %3 = qec.ppr.arbitrary ["X"](%1) %2 cond(%b0): !quantum.bit
    func.return
}

// -----

// dont merge different conditions

// CHECK-LABEL: dont_merge_condition
func.func public @dont_merge_condition(%q0: !quantum.bit, %0: f64, %1: f64, %b0: i1, %b1: i1) {
    // CHECK: [[in:%.+]] = qec.ppr.arbitrary ["X"]({{%.+}}) {{%.+}} cond({{%.+}})
    // CHECK: qec.ppr.arbitrary ["X"]({{%.+}}) [[in]] cond({{%.+}})
    %2 = qec.ppr.arbitrary ["X"](%0) %q0 cond(%b0): !quantum.bit
    %3 = qec.ppr.arbitrary ["X"](%1) %2 cond(%b1): !quantum.bit
    func.return
}

// -----

// don't merge conditions and non-conditions

// CHECK-LABEL: dont_merge_mixed
func.func public @dont_merge_mixed(%q0: !quantum.bit, %0: f64, %1: f64, %b0: i1) {
    // CHECK: [[in:%.+]] = qec.ppr.arbitrary ["X"]({{%.+}}) {{%.+}} cond({{%.+}})
    // CHECK: qec.ppr.arbitrary ["X"]({{%.+}}) [[in]]
    %2 = qec.ppr.arbitrary ["X"](%0) %q0 cond(%b0): !quantum.bit
    %3 = qec.ppr.arbitrary ["X"](%1) %2: !quantum.bit
    func.return
}

// -----

// merge with known angles

// CHECK-LABEL: merge_const_angles
func.func public @merge_const_angles(%q0: !quantum.bit) {
    // CHECK: [[newAngle:%.+]] = arith.constant 2.000000e+00 : f64
    // CHECK: qec.ppr.arbitrary ["Y"]([[newAngle]])
    %0 = arith.constant 0.420000e+00 : f64
    %1 = arith.constant 1.580000e+00 : f64
    %2 = qec.ppr.arbitrary ["Y"](%0) %q0: !quantum.bit
    %3 = qec.ppr.arbitrary ["Y"](%1) %2: !quantum.bit
    func.return
}

// -----

// merge known angle with unknown angle

// CHECK-LABEL: merge_const_var
func.func public @merge_const_var(%q0: !quantum.bit, %0: f64) {
    // CHECK: [[newAngle:%.+]] = arith.addf
    // CHECK: qec.ppr.arbitrary ["Z"]([[newAngle]])
    %1 = arith.constant 0.000420e+00 : f64
    %2 = qec.ppr.arbitrary ["Z"](%0) %q0: !quantum.bit
    %3 = qec.ppr.arbitrary ["Z"](%1) %2: !quantum.bit
    func.return
}

