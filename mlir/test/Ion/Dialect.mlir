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

// RUN: quantum-opt %s --split-input-file -verify-diagnostics | FileCheck %s

func.func @example_pulse(%arg0: f64) -> !quantum.bit {
    %0 = quantum.alloc( 1) : !quantum.reg

    // CHECK: [[q0:%.+]] = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit

    // CHECK: ion.pulse(%arg0 : f64) [[q0]] {beam = #ion.beam<
    // CHECK-SAME: transition_index = 0 : i64,
    // CHECK-SAME: rabi = 1.010000e+01 : f64,
    // CHECK-SAME: detuning = 1.111000e+01 : f64,
    // CHECK-SAME: polarization = dense<[0, 1]> : tensor<2xi64>,
    // CHECK-SAME: wavevector = dense<[0, 1]> : tensor<2xi64>>,
    // CHECK-SAME: phase = 0.000000e+00 : f64}
    ion.pulse(%arg0: f64) %1 {
        beam=#ion.beam<
            transition_index=0,
            rabi=10.10,
            detuning=11.11,
            polarization=dense<[0, 1]>: tensor<2xi64>,
            wavevector=dense<[0, 1]>: tensor<2xi64>
        >,
        phase=0.0
    }

    // CHECK: return [[q0]] : !quantum.bit
    return %1: !quantum.bit
}


func.func @example_parallel_protocol(%arg0: f64) -> !quantum.bit {
    %0 = quantum.alloc( 1) : !quantum.reg

    // CHECK: [[q0:%.+]] = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit

    // CHECK: [[paraproto:%.+]] = ion.parallelprotocol([[q0]]) : !quantum.bit {
    %2 = ion.parallelprotocol(%1): !quantum.bit {
        ^bb0(%arg1: !quantum.bit):
        // CHECK: ion.pulse(%arg0 : f64) %arg1 {beam = #ion.beam<
        // CHECK-SAME: transition_index = 1 : i64,
        // CHECK-SAME: rabi = 1.010000e+01 : f64,
        // CHECK-SAME: detuning = 1.111000e+01 : f64,
        // CHECK-SAME: polarization = dense<[0, 1]> : tensor<2xi64>,
        // CHECK-SAME: wavevector = dense<[0, 1]> : tensor<2xi64>>,
        // CHECK-SAME: phase = 0.000000e+00 : f64}
        ion.pulse(%arg0: f64) %arg1 {
            beam=#ion.beam<
                transition_index=1,
                rabi=10.10,
                detuning=11.11,
                polarization=dense<[0, 1]>: tensor<2xi64>,
                wavevector=dense<[0, 1]>: tensor<2xi64>
            >,
            phase=0.0
        }
        // CHECK: ion.pulse(%arg0 : f64) %arg1 {beam = #ion.beam<
        // CHECK-SAME: transition_index = 0 : i64,
        // CHECK-SAME: rabi = 1.010000e+01 : f64,
        // CHECK-SAME: detuning = 1.111000e+01 : f64,
        // CHECK-SAME: polarization = dense<[0, 1]> : tensor<2xi64>,
        // CHECK-SAME: wavevector = dense<[0, 1]> : tensor<2xi64>>,
        // CHECK-SAME: phase = 0.000000e+00 : f64}
        ion.pulse(%arg0: f64) %arg1 {
            beam=#ion.beam<
                transition_index=0,
                rabi=10.10,
                detuning=11.11,
                polarization=dense<[0, 1]>: tensor<2xi64>,
                wavevector=dense<[0, 1]>: tensor<2xi64>
            >,
            phase=0.0
        }
        // CHECK: ion.yield %arg1 : !quantum.bit
        ion.yield %arg1: !quantum.bit
    }

    // CHECK: return [[paraproto]] : !quantum.bit
    return %2: !quantum.bit
}

func.func @example_parallel_protocol_two_qubits(%arg0: f64) -> (!quantum.bit, !quantum.bit) {
    %0 = quantum.alloc( 1) : !quantum.reg

    // CHECK: [[q0:%.+]] = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit

    // CHECK: [[q1:%.+]] = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[1] : !quantum.reg -> !quantum.bit

    // CHECK: [[paraproto:%.+]]{{:2}} = ion.parallelprotocol([[q0]], [[q1]]) : !quantum.bit, !quantum.bit {
    %3:2 = ion.parallelprotocol(%1, %2): !quantum.bit, !quantum.bit {
        ^bb0(%arg1: !quantum.bit, %arg2: !quantum.bit):
        // CHECK: ion.pulse(%arg0 : f64) %arg1 {beam = #ion.beam<
        // CHECK-SAME: transition_index = 2 : i64,
        // CHECK-SAME: rabi = 1.010000e+01 : f64,
        // CHECK-SAME: detuning = 1.111000e+01 : f64,
        // CHECK-SAME: polarization = dense<[0, 1]> : tensor<2xi64>,
        // CHECK-SAME: wavevector = dense<[0, 1]> : tensor<2xi64>>,
        // CHECK-SAME: phase = 0.000000e+00 : f64}
        ion.pulse(%arg0: f64) %arg1 {
            beam=#ion.beam<
                transition_index=2,
                rabi=10.10,
                detuning=11.11,
                polarization=dense<[0, 1]>: tensor<2xi64>,
                wavevector=dense<[0, 1]>: tensor<2xi64>
            >,
            phase=0.0
        }
        // CHECK: ion.pulse(%arg0 : f64) %arg2 {beam = #ion.beam<
        // CHECK-SAME: transition_index = 1 : i64,
        // CHECK-SAME: rabi = 1.010000e+01 : f64,
        // CHECK-SAME: detuning = 1.111000e+01 : f64,
        // CHECK-SAME: polarization = dense<[0, 1]> : tensor<2xi64>,
        // CHECK-SAME: wavevector = dense<[0, 1]> : tensor<2xi64>>,
        // CHECK-SAME: phase = 0.000000e+00 : f64}
        ion.pulse(%arg0: f64) %arg2 {
            beam=#ion.beam<
                transition_index=1,
                rabi=10.10,
                detuning=11.11,
                polarization=dense<[0, 1]>: tensor<2xi64>,
                wavevector=dense<[0, 1]>: tensor<2xi64>
            >,
            phase=0.0
        }
        // CHECK: ion.yield %arg1, %arg2 : !quantum.bit, !quantum.bit
        ion.yield %arg1, %arg2: !quantum.bit, !quantum.bit
    }

    // CHECK: return [[paraproto]]#0, [[paraproto]]#1 : !quantum.bit, !quantum.bit
    return %3#0, %3#1: !quantum.bit, !quantum.bit
}

// No FileCheck here, return success if the MLIR can be parsed
func.func @example_ion() -> !ion.ion {
    %0 = ion.ion {
        name="YB117",
        mass=10.1,
        charge=12.1,
        position=dense<[0, 1]>: tensor<2xi64>,
        levels=[
            #ion.level<
                principal=1,
                spin=1.1,
                orbital=2.2,
                nuclear=3.3,
                spin_orbital=4.4,
                spin_orbital_nuclear=5.5,
                spin_orbital_nuclear_magnetization=6.6,
                energy=8.8
            >,
            #ion.level<
                principal=1,
                spin=1.1,
                orbital=2.2,
                nuclear=3.3,
                spin_orbital=4.4,
                spin_orbital_nuclear=5.5,
                spin_orbital_nuclear_magnetization=6.6,
                energy=8.8
            >
        ],
        transitions=[
            #ion.transition<
                level_0 = #ion.level<
                    principal=1,
                    spin=1.1,
                    orbital=2.2,
                    nuclear=3.3,
                    spin_orbital=4.4,
                    spin_orbital_nuclear=5.5,
                    spin_orbital_nuclear_magnetization=6.6,
                    energy=8.8
                >,
                level_1 = #ion.level<
                    principal=1,
                    spin=1.1,
                    orbital=2.2,
                    nuclear=3.3,
                    spin_orbital=4.4,
                    spin_orbital_nuclear=5.5,
                    spin_orbital_nuclear_magnetization=6.6,
                    energy=8.8
                >,
                einstein_a=10.10
            >,
            #ion.transition<
                level_0 = #ion.level<
                    principal=1,
                    spin=1.1,
                    orbital=2.2,
                    nuclear=3.3,
                    spin_orbital=4.4,
                    spin_orbital_nuclear=5.5,
                    spin_orbital_nuclear_magnetization=6.6,
                    energy=8.8
                >,
                level_1 = #ion.level<
                    principal=1,
                    spin=1.1,
                    orbital=2.2,
                    nuclear=3.3,
                    spin_orbital=4.4,
                    spin_orbital_nuclear=5.5,
                    spin_orbital_nuclear_magnetization=6.6,
                    energy=8.8
                >,
                einstein_a=10.10
            >
        ]
    }: !ion.ion
    return %0: !ion.ion
}
