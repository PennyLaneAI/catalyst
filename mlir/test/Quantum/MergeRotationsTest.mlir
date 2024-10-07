// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --pass-pipeline="builtin.module(merge-rotations{func-name=test_merge_rotations})" --split-input-file -verify-diagnostics %s | FileCheck %s

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