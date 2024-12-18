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

// RUN: quantum-opt %s --pass-pipeline="builtin.module(func.func(quantum-to-ion{\
// RUN:    device-toml-loc=%S/oqd_device_parameters.toml \
// RUN:    qubit-toml-loc=%S/oqd_qubit_parameters.toml \
// RUN:    gate-to-pulse-toml-loc=%S/oqd_gate_decomposition_parameters.toml}))" \
// RUN: --split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: example_ion_two_qubit
func.func @example_ion_two_qubit(%arg0: f64) -> !quantum.bit {
    // %0 = ion.ion {
    //     name="YB117",
    //     mass=10.1,
    //     charge=12.1,
    //     position=dense<[0, 1]>: tensor<2xi64>,
    //     levels=[
    //         #ion.level<
    //             principal=1,
    //             spin=1.1,
    //             orbital=2.2,
    //             nuclear=3.3,
    //             spin_orbital=4.4,
    //             spin_orbital_nuclear=5.5,
    //             spin_orbital_nuclear_magnetization=6.6,
    //             energy=8.8
    //         >,
    //         #ion.level<
    //             principal=1,
    //             spin=1.1,
    //             orbital=2.2,
    //             nuclear=3.3,
    //             spin_orbital=4.4,
    //             spin_orbital_nuclear=5.5,
    //             spin_orbital_nuclear_magnetization=6.6,
    //             energy=8.8
    //         >
    //     ],
    //     transitions=[
    //         #ion.transition<
    //             level_0 = #ion.level<
    //                 principal=1,
    //                 spin=1.1,
    //                 orbital=2.2,
    //                 nuclear=3.3,
    //                 spin_orbital=4.4,
    //                 spin_orbital_nuclear=5.5,
    //                 spin_orbital_nuclear_magnetization=6.6,
    //                 energy=8.8
    //             >,
    //             level_1 = #ion.level<
    //                 principal=1,
    //                 spin=1.1,
    //                 orbital=2.2,
    //                 nuclear=3.3,
    //                 spin_orbital=4.4,
    //                 spin_orbital_nuclear=5.5,
    //                 spin_orbital_nuclear_magnetization=6.6,
    //                 energy=8.8
    //             >,
    //             einstein_a=10.10
    //         >,
    //         #ion.transition<
    //             level_0 = #ion.level<
    //                 principal=1,
    //                 spin=1.1,
    //                 orbital=2.2,
    //                 nuclear=3.3,
    //                 spin_orbital=4.4,
    //                 spin_orbital_nuclear=5.5,
    //                 spin_orbital_nuclear_magnetization=6.6,
    //                 energy=8.8
    //             >,
    //             level_1 = #ion.level<
    //                 principal=1,
    //                 spin=1.1,
    //                 orbital=2.2,
    //                 nuclear=3.3,
    //                 spin_orbital=4.4,
    //                 spin_orbital_nuclear=5.5,
    //                 spin_orbital_nuclear_magnetization=6.6,
    //                 energy=8.8
    //             >,
    //             einstein_a=10.10
    //         >
    //     ]
    // }: !ion.ion

    %1 = quantum.alloc( 2) : !quantum.reg
    %2 = quantum.extract %1[ 0] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %1[ 1] : !quantum.reg -> !quantum.bit


    // CHECK-DAG: [[rabi1:%.+]] = arith.constant 1.100000e+00 : f64
    // CHECK-DAG: [[rabi2:%.+]] = arith.constant 1.230000e+00 : f64

    // CHECK: [[rx1out:%.+]] = ion.parallelprotocol({{%.+}}) : !quantum.bit {
    // CHECK-NEXT: ^{{.*}}(%arg1: !quantum.bit):
    // CHECK-NEXT: [[timerx1:%.+]] = arith.divf %arg0, [[rabi1]] : f64
    // CHECK-NEXT: ion.pulse([[timerx1]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 0 : i64,
    // CHECK-SAME:         rabi = 1.100000e+00 : f64,
    // CHECK-SAME:         detuning = 2.200000e+00 : f64,
    // CHECK-SAME:         polarization = dense<[0, 1]> : vector<2xi64>,
    // CHECK-SAME:         wavevector = dense<[-2, 3]> : vector<2xi64>>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timerx1]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 1 : i64,
    // CHECK-SAME:         rabi = 1.100000e+00 : f64,
    // CHECK-SAME:         detuning = 2.200000e+00 : f64,
    // CHECK-SAME:         polarization = dense<[0, 1]> : vector<2xi64>,
    // CHECK-SAME:         wavevector = dense<[-2, 3]> : vector<2xi64>>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: }
    %4 = quantum.custom "RX"(%arg0) %2 : !quantum.bit

    // CHECK: [[ry1out:%.+]] = ion.parallelprotocol([[rx1out]]) : !quantum.bit {
    // CHECK-NEXT: ^{{.*}}(%arg1: !quantum.bit):
    // CHECK-NEXT: [[timery1:%.+]] = arith.divf %arg0, [[rabi1]] : f64
    // CHECK-NEXT: ion.pulse([[timery1]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 0 : i64,
    // CHECK-SAME:         rabi = 1.100000e+00 : f64,
    // CHECK-SAME:         detuning = 2.200000e+00 : f64,
    // CHECK-SAME:         polarization = dense<[0, 1]> : vector<2xi64>,
    // CHECK-SAME:         wavevector = dense<[-2, 3]> : vector<2xi64>>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timery1]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 1 : i64,
    // CHECK-SAME:         rabi = 1.100000e+00 : f64,
    // CHECK-SAME:         detuning = 2.200000e+00 : f64,
    // CHECK-SAME:         polarization = dense<[0, 1]> : vector<2xi64>,
    // CHECK-SAME:         wavevector = dense<[-2, 3]> : vector<2xi64>>,
    // CHECK-SAME:     phase = 3.1415926535{{[0-9]*}} : f64}
    // CHECK-NEXT: }
    %5 = quantum.custom "RY"(%arg0) %4 : !quantum.bit

    // CHECK: [[rx2out:%.+]] = ion.parallelprotocol([[ry1out]]) : !quantum.bit {
    // CHECK-NEXT: ^{{.*}}(%arg1: !quantum.bit):
    // CHECK-NEXT: [[timerx2:%.+]] = arith.divf %arg0, [[rabi1]] : f64
    // CHECK-NEXT: ion.pulse([[timerx2]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 0 : i64,
    // CHECK-SAME:         rabi = 1.100000e+00 : f64,
    // CHECK-SAME:         detuning = 2.200000e+00 : f64,
    // CHECK-SAME:         polarization = dense<[0, 1]> : vector<2xi64>,
    // CHECK-SAME:         wavevector = dense<[-2, 3]> : vector<2xi64>>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timerx2]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 1 : i64,
    // CHECK-SAME:         rabi = 1.100000e+00 : f64,
    // CHECK-SAME:         detuning = 2.200000e+00 : f64,
    // CHECK-SAME:         polarization = dense<[0, 1]> : vector<2xi64>,
    // CHECK-SAME:         wavevector = dense<[-2, 3]> : vector<2xi64>>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: }
    %6 = quantum.custom "RX"(%arg0) %5 : !quantum.bit

    // CHECK: [[msout:%.+]] = ion.parallelprotocol([[rx2out]], {{%.+}}) : !quantum.bit, !quantum.bit {
    // CHECK-NEXT: ^{{.*}}(%arg1: !quantum.bit, %arg2: !quantum.bit):
    // CHECK-NEXT: [[timems:%.+]] = arith.divf %arg0, [[rabi2]] : f64
    // CHECK-NEXT: ion.pulse([[timems]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 0 : i64,
    // CHECK-SAME:         rabi = 1.230000e+00 : f64,
    // CHECK-SAME:         detuning = 4.560000e+00 : f64,
    // CHECK-SAME:         polarization = dense<[7, 8]> : vector<2xi64>,
    // CHECK-SAME:         wavevector = dense<[9, 10]> : vector<2xi64>>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 1 : i64,
    // CHECK-SAME:         rabi = 1.230000e+00 : f64,
    // CHECK-SAME:         detuning = 5.660000e+00 : f64,
    // CHECK-SAME:         polarization = dense<[7, 8]> : vector<2xi64>,
    // CHECK-SAME:         wavevector = dense<[-9, -10]> : vector<2xi64>>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 1 : i64,
    // CHECK-SAME:         rabi = 1.230000e+00 : f64,
    // CHECK-SAME:         detuning = 3.4{{.*}} : f64,
    // CHECK-SAME:         polarization = dense<[7, 8]> : vector<2xi64>,
    // CHECK-SAME:         wavevector = dense<[-9, -10]> : vector<2xi64>>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 0 : i64,
    // CHECK-SAME:         rabi = 1.230000e+00 : f64,
    // CHECK-SAME:         detuning = 4.560000e+00 : f64,
    // CHECK-SAME:         polarization = dense<[7, 8]> : vector<2xi64>,
    // CHECK-SAME:         wavevector = dense<[9, 10]> : vector<2xi64>>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 1 : i64,
    // CHECK-SAME:         rabi = 1.230000e+00 : f64,
    // CHECK-SAME:         detuning = 8.960000e+00 : f64,
    // CHECK-SAME:         polarization = dense<[7, 8]> : vector<2xi64>,
    // CHECK-SAME:         wavevector = dense<[-9, -10]> : vector<2xi64>>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 1 : i64,
    // CHECK-SAME:         rabi = 1.230000e+00 : f64,
    // CHECK-SAME:         detuning = 0.1{{.*}} : f64,
    // CHECK-SAME:         polarization = dense<[7, 8]> : vector<2xi64>,
    // CHECK-SAME:         wavevector = dense<[-9, -10]> : vector<2xi64>>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: }
    %7:2 = quantum.custom "MS"(%arg0) %6, %3 : !quantum.bit, !quantum.bit
    return %7#0: !quantum.bit
}


// -----


// CHECK-LABEL: example_ion_three_qubit
func.func @example_ion_three_qubit(%arg0: f64) -> (!quantum.bit, !quantum.bit, !quantum.bit) {
    // %0 = ion.ion {
    //     name="YB117",
    //     mass=10.1,
    //     charge=12.1,
    //     position=dense<[0, 1]>: tensor<2xi64>,
    //     levels=[
    //         #ion.level<
    //             principal=1,
    //             spin=1.1,
    //             orbital=2.2,
    //             nuclear=3.3,
    //             spin_orbital=4.4,
    //             spin_orbital_nuclear=5.5,
    //             spin_orbital_nuclear_magnetization=6.6,
    //             energy=8.8
    //         >,
    //         #ion.level<
    //             principal=1,
    //             spin=1.1,
    //             orbital=2.2,
    //             nuclear=3.3,
    //             spin_orbital=4.4,
    //             spin_orbital_nuclear=5.5,
    //             spin_orbital_nuclear_magnetization=6.6,
    //             energy=8.8
    //         >
    //     ],
    //     transitions=[
    //         #ion.transition<
    //             level_0 = #ion.level<
    //                 principal=1,
    //                 spin=1.1,
    //                 orbital=2.2,
    //                 nuclear=3.3,
    //                 spin_orbital=4.4,
    //                 spin_orbital_nuclear=5.5,
    //                 spin_orbital_nuclear_magnetization=6.6,
    //                 energy=8.8
    //             >,
    //             level_1 = #ion.level<
    //                 principal=1,
    //                 spin=1.1,
    //                 orbital=2.2,
    //                 nuclear=3.3,
    //                 spin_orbital=4.4,
    //                 spin_orbital_nuclear=5.5,
    //                 spin_orbital_nuclear_magnetization=6.6,
    //                 energy=8.8
    //             >,
    //             einstein_a=10.10
    //         >,
    //         #ion.transition<
    //             level_0 = #ion.level<
    //                 principal=1,
    //                 spin=1.1,
    //                 orbital=2.2,
    //                 nuclear=3.3,
    //                 spin_orbital=4.4,
    //                 spin_orbital_nuclear=5.5,
    //                 spin_orbital_nuclear_magnetization=6.6,
    //                 energy=8.8
    //             >,
    //             level_1 = #ion.level<
    //                 principal=1,
    //                 spin=1.1,
    //                 orbital=2.2,
    //                 nuclear=3.3,
    //                 spin_orbital=4.4,
    //                 spin_orbital_nuclear=5.5,
    //                 spin_orbital_nuclear_magnetization=6.6,
    //                 energy=8.8
    //             >,
    //             einstein_a=10.10
    //         >
    //     ]
    // }: !ion.ion

    %1 = quantum.alloc( 3) : !quantum.reg
    %2 = quantum.extract %1[ 0] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %1[ 1] : !quantum.reg -> !quantum.bit
    %4 = quantum.extract %1[ 2] : !quantum.reg -> !quantum.bit

    // CHECK-DAG: [[rabi1:%.+]] = arith.constant 1.230000e+00 : f64
    // CHECK-DAG: [[rabi2:%.+]] = arith.constant 4.560000e+00 : f64
    // CHECK-DAG: [[rabi3:%.+]] = arith.constant 99.989999999999994 : f64

    // CHECK: [[ms1out:%.+]]:2 = ion.parallelprotocol({{%.+}}, {{%.+}}) : !quantum.bit, !quantum.bit {

    // CHECK-NEXT: ^{{.*}}(%arg1: !quantum.bit, %arg2: !quantum.bit):
    // CHECK-NEXT: [[timems1:%.+]] = arith.divf %arg0, [[rabi1]] : f64
    // CHECK-NEXT: ion.pulse([[timems1]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 0 : i64,
    // CHECK-SAME:         rabi = 1.230000e+00 : f64,
    // CHECK-SAME:         detuning = 4.560000e+00 : f64,
    // CHECK-SAME:         polarization = dense<[7, 8]> : vector<2xi64>,
    // CHECK-SAME:         wavevector = dense<[9, 10]> : vector<2xi64>>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems1]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 1 : i64,
    // CHECK-SAME:         rabi = 1.230000e+00 : f64,
    // CHECK-SAME:         detuning = 5.660000e+00 : f64,
    // CHECK-SAME:         polarization = dense<[7, 8]> : vector<2xi64>,
    // CHECK-SAME:         wavevector = dense<[-9, -10]> : vector<2xi64>>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems1]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 1 : i64,
    // CHECK-SAME:         rabi = 1.230000e+00 : f64,
    // CHECK-SAME:         detuning = 3.4599999999999995 : f64,
    // CHECK-SAME:         polarization = dense<[7, 8]> : vector<2xi64>,
    // CHECK-SAME:         wavevector = dense<[-9, -10]> : vector<2xi64>>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems1]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 0 : i64,
    // CHECK-SAME:         rabi = 1.230000e+00 : f64,
    // CHECK-SAME:         detuning = 4.560000e+00 : f64,
    // CHECK-SAME:         polarization = dense<[7, 8]> : vector<2xi64>,
    // CHECK-SAME:         wavevector = dense<[9, 10]> : vector<2xi64>>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems1]] : f64) %arg1 {
    // CHECK-SAME:    beam = #ion.beam<
    // CHECK-SAME:        transition_index = 1 : i64,
    // CHECK-SAME:        rabi = 1.230000e+00 : f64,
    // CHECK-SAME:        detuning = 8.960000e+00 : f64,
    // CHECK-SAME:        polarization = dense<[7, 8]> : vector<2xi64>,
    // CHECK-SAME:        wavevector = dense<[-9, -10]> : vector<2xi64>>,
    // CHECK-SAME:    phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems1]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 1 : i64,
    // CHECK-SAME:         rabi = 1.230000e+00 : f64,
    // CHECK-SAME:         detuning = 0.15999999999999925 : f64,
    // CHECK-SAME:         polarization = dense<[7, 8]> : vector<2xi64>,
    // CHECK-SAME:         wavevector = dense<[-9, -10]> : vector<2xi64>>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: }
    %5:2 = quantum.custom "MS"(%arg0) %2, %3 : !quantum.bit, !quantum.bit

    // CHECK: [[ms2out:%.+]]:2 = ion.parallelprotocol([[ms1out]]#0, {{%.+}}) : !quantum.bit, !quantum.bit {
    // CHECK-NEXT: ^{{.*}}(%arg1: !quantum.bit, %arg2: !quantum.bit):
    // CHECK-NEXT: [[timems2:%.+]] = arith.divf %arg0, [[rabi2]] : f64
    // CHECK-NEXT: ion.pulse([[timems2]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 0 : i64,
    // CHECK-SAME:         rabi = 4.560000e+00 : f64,
    // CHECK-SAME:         detuning = 7.8899999999999996 : f64,
    // CHECK-SAME:         polarization = dense<[1, 2]> : vector<2xi64>,
    // CHECK-SAME:         wavevector = dense<[-3, 4]> : vector<2xi64>>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems2]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 1 : i64,
    // CHECK-SAME:         rabi = 4.560000e+00 : f64,
    // CHECK-SAME:         detuning = 8.990000e+00 : f64,
    // CHECK-SAME:         polarization = dense<[1, 2]> : vector<2xi64>,
    // CHECK-SAME:         wavevector = dense<[3, -4]> : vector<2xi64>>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems2]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 1 : i64,
    // CHECK-SAME:         rabi = 4.560000e+00 : f64,
    // CHECK-SAME:         detuning = 6.7899999999999991 : f64,
    // CHECK-SAME:         polarization = dense<[1, 2]> : vector<2xi64>,
    // CHECK-SAME:         wavevector = dense<[3, -4]> : vector<2xi64>>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems2]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 0 : i64,
    // CHECK-SAME:         rabi = 4.560000e+00 : f64,
    // CHECK-SAME:         detuning = 7.8899999999999996 : f64,
    // CHECK-SAME:         polarization = dense<[1, 2]> : vector<2xi64>,
    // CHECK-SAME:         wavevector = dense<[-3, 4]> : vector<2xi64>>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems2]] : f64) %arg1 {
    // CHECK-SAME:    beam = #ion.beam<
    // CHECK-SAME:        transition_index = 1 : i64,
    // CHECK-SAME:        rabi = 4.560000e+00 : f64,
    // CHECK-SAME:        detuning = 1.559000e+01 : f64,
    // CHECK-SAME:        polarization = dense<[1, 2]> : vector<2xi64>,
    // CHECK-SAME:        wavevector = dense<[3, -4]> : vector<2xi64>>,
    // CHECK-SAME:    phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems2]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 1 : i64,
    // CHECK-SAME:         rabi = 4.560000e+00 : f64,
    // CHECK-SAME:         detuning = 0.1899999999999995 : f64,
    // CHECK-SAME:         polarization = dense<[1, 2]> : vector<2xi64>,
    // CHECK-SAME:         wavevector = dense<[3, -4]> : vector<2xi64>>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: }
    %6:2 = quantum.custom "MS"(%arg0) %5#0, %4 : !quantum.bit, !quantum.bit

    // CHECK: [[ms3out:%.+]]:2 = ion.parallelprotocol([[ms1out]]#1, [[ms2out]]#1) : !quantum.bit, !quantum.bit {
    // CHECK-NEXT: ^{{.*}}(%arg1: !quantum.bit, %arg2: !quantum.bit):
    // CHECK-NEXT: [[timems3:%.+]] = arith.divf %arg0, [[rabi3]] : f64
    // CHECK-NEXT: ion.pulse([[timems3]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 0 : i64,
    // CHECK-SAME:         rabi = 99.989999999999994 : f64,
    // CHECK-SAME:         detuning = 1.001000e+02 : f64,
    // CHECK-SAME:         polarization = dense<[37, 42]> : vector<2xi64>,
    // CHECK-SAME:         wavevector = dense<[-42, -37]> : vector<2xi64>>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems3]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 1 : i64,
    // CHECK-SAME:         rabi = 99.989999999999994 : f64,
    // CHECK-SAME:         detuning = 1.045000e+02 : f64,
    // CHECK-SAME:         polarization = dense<[37, 42]> : vector<2xi64>,
    // CHECK-SAME:         wavevector = dense<[42, 37]> : vector<2xi64>>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems3]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 1 : i64,
    // CHECK-SAME:         rabi = 99.989999999999994 : f64,
    // CHECK-SAME:         detuning = 95.699999999999989 : f64,
    // CHECK-SAME:         polarization = dense<[37, 42]> : vector<2xi64>,
    // CHECK-SAME:         wavevector = dense<[42, 37]> : vector<2xi64>>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems3]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 0 : i64,
    // CHECK-SAME:         rabi = 99.989999999999994 : f64,
    // CHECK-SAME:         detuning = 1.001000e+02 : f64,
    // CHECK-SAME:         polarization = dense<[37, 42]> : vector<2xi64>,
    // CHECK-SAME:         wavevector = dense<[-42, -37]> : vector<2xi64>>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems3]] : f64) %arg1 {
    // CHECK-SAME:    beam = #ion.beam<
    // CHECK-SAME:        transition_index = 1 : i64,
    // CHECK-SAME:        rabi = 99.989999999999994 : f64,
    // CHECK-SAME:        detuning = 1.078000e+02 : f64,
    // CHECK-SAME:        polarization = dense<[37, 42]> : vector<2xi64>,
    // CHECK-SAME:        wavevector = dense<[42, 37]> : vector<2xi64>>,
    // CHECK-SAME:    phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems3]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 1 : i64,
    // CHECK-SAME:         rabi = 99.989999999999994 : f64,
    // CHECK-SAME:         detuning = 92.399999999999991 : f64,
    // CHECK-SAME:         polarization = dense<[37, 42]> : vector<2xi64>,
    // CHECK-SAME:         wavevector = dense<[42, 37]> : vector<2xi64>>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: }
    %7:2 = quantum.custom "MS"(%arg0) %5#1, %6#1 : !quantum.bit, !quantum.bit
    return %6#0, %7#0, %7#1: !quantum.bit, !quantum.bit, !quantum.bit
}
