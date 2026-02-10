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

// RUN: quantum-opt --ions-decomposition --split-input-file -verify-diagnostics %s | FileCheck %s

func.func @test_ions_decomposition(%arg0: f64) -> !quantum.bit {
    // CHECK: [[PIO2:%.+]] = arith.constant 1.5707963267948966 : f64
    // CHECK: [[PIO4:%.+]] = arith.constant 0.78539816339744828 : f64
    // CHECK: [[MPIO2:%.+]] = arith.constant -1.5707963267948966 : f64
    // CHECK: [[reg:%.+]] = quantum.alloc( 1) : !quantum.reg
    // CHECK: [[qubit:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit1:%.+]] = quantum.custom "RX"([[MPIO2]]) [[qubit]] : !quantum.bit
    // CHECK: [[qubit2:%.+]] = quantum.custom "RY"([[PIO4]]) [[qubit1]] : !quantum.bit
    // CHECK: [[qubit3:%.+]] = quantum.custom "RX"([[PIO2]]) [[qubit2]] : !quantum.bit
    // CHECK: [[qubit4:%.+]] = quantum.custom "RX"(%arg0) [[qubit3]] : !quantum.bit
    // CHECK: [[qubit5:%.+]] = quantum.custom "RX"([[MPIO2]]) [[qubit4]] : !quantum.bit
    // CHECK: [[qubit6:%.+]] = quantum.custom "RY"([[PIO4]]) [[qubit5]] : !quantum.bit
    // CHECK: [[qubit7:%.+]] = quantum.custom "RX"([[PIO2]]) [[qubit6]] : !quantum.bit
    // CHECK: return [[qubit7]]
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.custom "T"() %1 : !quantum.bit
    %3 = quantum.custom "RX"(%arg0) %2 : !quantum.bit
    %4 = quantum.custom "T"() %3 : !quantum.bit
    return %4 : !quantum.bit
}

// -----

func.func @test_ions_decomposition(%arg0: f64) -> !quantum.bit {
    // CHECK: [[PIO2:%.+]] = arith.constant 1.5707963267948966 : f64
    // CHECK: [[MPIO2:%.+]] = arith.constant -1.5707963267948966 : f64
    // CHECK: [[reg:%.+]] = quantum.alloc( 1) : !quantum.reg
    // CHECK: [[qubit:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit1:%.+]] = quantum.custom "RX"([[MPIO2]]) [[qubit]] : !quantum.bit
    // CHECK: [[qubit2:%.+]] = quantum.custom "RY"([[MPIO2]]) [[qubit1]] : !quantum.bit
    // CHECK: [[qubit3:%.+]] = quantum.custom "RX"([[PIO2]]) [[qubit2]] : !quantum.bit
    // CHECK: [[qubit4:%.+]] = quantum.custom "RX"([[MPIO2]]) [[qubit3]] : !quantum.bit
    // CHECK: [[qubit5:%.+]] = quantum.custom "RY"([[PIO2]]) [[qubit4]] : !quantum.bit
    // CHECK: [[qubit6:%.+]] = quantum.custom "RX"([[PIO2]]) [[qubit5]] : !quantum.bit
    // CHECK: return [[qubit6]]
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.custom "S"() %1 {adjoint}: !quantum.bit
    %3 = quantum.custom "S"() %2 : !quantum.bit
    return %3 : !quantum.bit
}

// -----

func.func @test_ions_decomposition(%arg0: f64) -> !quantum.bit {
    // CHECK: [[PIO4:%.+]] = arith.constant 0.78539816339744828 : f64
    // CHECK: [[PIO2:%.+]] = arith.constant 1.5707963267948966 : f64
    // CHECK: [[MPIO4:%.+]] = arith.constant -0.78539816339744828 : f64
    // CHECK: [[MPIO2:%.+]] = arith.constant -1.5707963267948966 : f64
    // CHECK: [[reg:%.+]] = quantum.alloc( 1) : !quantum.reg
    // CHECK: [[qubit:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit1:%.+]] = quantum.custom "RX"([[MPIO2]]) [[qubit]] : !quantum.bit
    // CHECK: [[qubit2:%.+]] = quantum.custom "RY"([[MPIO4]]) [[qubit1]] : !quantum.bit
    // CHECK: [[qubit3:%.+]] = quantum.custom "RX"([[PIO2]]) [[qubit2]] : !quantum.bit
    // CHECK: [[qubit4:%.+]] = quantum.custom "RX"([[MPIO2]]) [[qubit3]] : !quantum.bit
    // CHECK: [[qubit5:%.+]] = quantum.custom "RY"([[PIO4]]) [[qubit4]] : !quantum.bit
    // CHECK: [[qubit6:%.+]] = quantum.custom "RX"([[PIO2]]) [[qubit5]] : !quantum.bit
    // CHECK: return [[qubit6]]
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.custom "T"() %1 {adjoint}: !quantum.bit
    %3 = quantum.custom "T"() %2 : !quantum.bit
    return %3 : !quantum.bit
}

// -----
func.func @test_ions_decomposition(%arg0: f64, %arg1: f64, %arg2: f64) -> !quantum.bit {
    // CHECK: [[PIO2:%.+]] = arith.constant 1.5707963267948966 : f64
    // CHECK: [[MPIO2:%.+]] = arith.constant -1.5707963267948966 : f64
    // CHECK: [[reg:%.+]] = quantum.alloc( 1) : !quantum.reg
    // CHECK: [[qubit:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit1:%.+]] = quantum.custom "RX"([[MPIO2]]) [[qubit]] : !quantum.bit
    // CHECK: [[qubit2:%.+]] = quantum.custom "RY"(%arg0) [[qubit1]] : !quantum.bit
    // CHECK: [[qubit3:%.+]] = quantum.custom "RX"([[PIO2]]) [[qubit2]] : !quantum.bit
    // CHECK: [[qubit4:%.+]] = quantum.custom "RX"(%arg1) [[qubit3]] : !quantum.bit
    // CHECK: [[qubit5:%.+]] = quantum.custom "RX"([[MPIO2]]) [[qubit4]] : !quantum.bit
    // CHECK: [[qubit6:%.+]] = quantum.custom "RY"(%arg2) [[qubit5]] : !quantum.bit
    // CHECK: [[qubit7:%.+]] = quantum.custom "RX"([[PIO2]]) [[qubit6]] : !quantum.bit
    // CHECK: return [[qubit7]]
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.custom "RZ"(%arg0) %1 : !quantum.bit
    %3 = quantum.custom "RX"(%arg1) %2 : !quantum.bit
    %4 = quantum.custom "RZ"(%arg2) %3 : !quantum.bit
    return %4 : !quantum.bit
}

// -----

func.func @test_ions_decomposition(%arg0: f64, %arg1: f64, %arg2: f64) -> (!quantum.bit, !quantum.bit) {
    // CHECK: [[MPIO2:%.+]] = arith.constant -1.5707963267948966 : f64
    // CHECK: [[PIO2:%.+]] = arith.constant 1.5707963267948966 : f64
    // CHECK: [[reg:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[qubit0:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit1:%.+]] = quantum.extract [[reg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit2:%.+]] = quantum.custom "RY"([[PIO2]]) [[qubit0]] : !quantum.bit
    // CHECK: [[qubits3:%.+]]:2 = quantum.custom "MS"([[PIO2]]) [[qubit2]], [[qubit1]] : !quantum.bit, !quantum.bit
    // CHECK: [[qubit4:%.+]] = quantum.custom "RX"([[MPIO2]]) [[qubits3]]#0 : !quantum.bit
    // CHECK: [[qubit5:%.+]] = quantum.custom "RY"([[MPIO2]]) [[qubits3]]#1 : !quantum.bit
    // CHECK: [[qubit6:%.+]] = quantum.custom "RY"([[MPIO2]]) [[qubit4]] : !quantum.bit
    // CHECK: return [[qubit5]], [[qubit6]] : !quantum.bit, !quantum.bit
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3:2 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit
    return %3#0, %3#1 : !quantum.bit, !quantum.bit
}
