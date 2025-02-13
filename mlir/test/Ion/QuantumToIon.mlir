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
func.func @example_ion_two_qubit(%arg0: f64) -> !quantum.bit  attributes {qnode} {


    // COM: attr-dict in op's assembly format sorts fields alphabetically
    // COM: so we have to CHECK-SAME in alphabetical order as well
    // COM: The order is charge, levels, mass, name, position, transitions

    // CHECK: {{%.+}} = ion.ion {
    // CHECK-SAME:    charge = 1.000000e+00
    // CHECK-SAME:    levels = [
    // CHECK-SAME:        #ion.level<
    // CHECK-SAME:            label = "downstate",
    // CHECK-SAME:            principal = 6
    // CHECK-SAME:            spin = 5.000000e-01
    // CHECK-SAME:            orbital = 1.000000e+00
    // CHECK-SAME:            nuclear = 1.500000e+00
    // CHECK-SAME:            spin_orbital = 2.000000e+00
    // CHECK-SAME:            spin_orbital_nuclear = 2.500000e+00
    // CHECK-SAME:            spin_orbital_nuclear_magnetization = -3.000000e+00
    // CHECK-SAME:            energy = 0.000000e+00
    // CHECK-SAME:        >,
    // CHECK-SAME:        #ion.level<
    // CHECK-SAME:            label = "upstate",
    // CHECK-SAME:            principal = 6
    // CHECK-SAME:            spin = 1.500000e+00
    // CHECK-SAME:            orbital = 2.000000e+00
    // CHECK-SAME:            nuclear = 2.500000e+00
    // CHECK-SAME:            spin_orbital = 3.000000e+00
    // CHECK-SAME:            spin_orbital_nuclear = 3.500000e+00
    // CHECK-SAME:            spin_orbital_nuclear_magnetization = -4.000000e+00
    // CHECK-SAME:            energy = 1.264300e+10
    // CHECK-SAME:        >,
    // CHECK-SAME:        #ion.level<
    // CHECK-SAME:            label = "estate",
    // CHECK-SAME:            principal = 5
    // CHECK-SAME:            spin = 2.500000e+00
    // CHECK-SAME:            orbital = 3.000000e+00
    // CHECK-SAME:            nuclear = 3.500000e+00
    // CHECK-SAME:            spin_orbital = 4.000000e+00
    // CHECK-SAME:            spin_orbital_nuclear = 4.500000e+00
    // CHECK-SAME:            spin_orbital_nuclear_magnetization = -5.000000e+00
    // CHECK-SAME:            energy = 8.115200e+14
    // CHECK-SAME:        >
    // CHECK-SAME:    ],
    // CHECK-SAME:    mass = 1.710000e+02
    // CHECK-SAME:    name = "Yb171"
    // CHECK-SAME:    position = array<f64: 1.000000e+00, 2.000000e+00, -1.000000e+00>,
    // CHECK-SAME:    transitions = [
    // CHECK-SAME:        #ion.transition<
    // CHECK-SAME:            level_0 = "downstate",
    // CHECK-SAME:            level_1 = "estate",
    // CHECK-SAME:            einstein_a = 2.200000e+00 : f64,
    // CHECK-SAME:            multipole = "E2"
    // CHECK-SAME:        >,
    // CHECK-SAME:        #ion.transition<
    // CHECK-SAME:            level_0 = "downstate",
    // CHECK-SAME:            level_1 = "upstate",
    // CHECK-SAME:            einstein_a = 1.100000e+00 : f64,
    // CHECK-SAME:            multipole = "M1"
    // CHECK-SAME:        >,
    // CHECK-SAME:        #ion.transition<
    // CHECK-SAME:            level_0 = "estate",
    // CHECK-SAME:            level_1 = "upstate",
    // CHECK-SAME:            einstein_a = 3.300000e+00 : f64,
    // CHECK-SAME:            multipole = "E1"
    // CHECK-SAME:        >
    // CHECK-SAME:    ]
    // CHECK-SAME: } : !ion.ion

    // CHECK: ion.mode {
    // CHECK-SAME:    modes = [
    // CHECK-SAME:        #ion.phonon<
    // CHECK-SAME:            energy = 1.100000e+00 : f64,
    // CHECK-SAME:            eignevector = [1.000000e+00, 0.000000e+00, 0.000000e+00]
    // CHECK-SAME:        >,
    // CHECK-SAME:        #ion.phonon<
    // CHECK-SAME:            energy = 2.200000e+00 : f64,
    // CHECK-SAME:            eignevector = [0.000000e+00, 1.000000e+00, 0.000000e+00]
    // CHECK-SAME:        >,
    // CHECK-SAME:        #ion.phonon<
    // CHECK-SAME:            energy = 3.300000e+00 : f64,
    // CHECK-SAME:            eignevector = [0.000000e+00, 0.000000e+00, 1.000000e+00]
    // CHECK-SAME:        >
    // CHECK-SAME:    ]
    // CHECK-SAME: } 

    %1 = quantum.alloc( 2) : !quantum.reg

    // CHECK: [[qubit0:%.+]] = quantum.extract %1[ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit1:%.+]] = quantum.extract %1[ 1] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %1[ 0] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %1[ 1] : !quantum.reg -> !quantum.bit

    // CHECK: [[rx1out:%.+]] = ion.parallelprotocol([[qubit0]]) : !quantum.bit {
    // CHECK-NEXT: ^{{.*}}(%arg1: !quantum.bit):
    // CHECK-NEXT: [[rabi1:%.+]] = arith.constant 1.100000e+00 : f64
    // CHECK-NEXT: [[timerx1:%.+]] = arith.divf %arg0, [[rabi1]] : f64
    // CHECK-NEXT: ion.pulse([[timerx1]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 0 : i64,
    // CHECK-SAME:         rabi = 1.100000e+00 : f64,
    // CHECK-SAME:         detuning = 2.200000e+00 : f64,
    // CHECK-SAME:         polarization = [0, 1, 2],
    // CHECK-SAME:         wavevector = [-2, 3, 4]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timerx1]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 1 : i64,
    // CHECK-SAME:         rabi = 1.100000e+00 : f64,
    // CHECK-SAME:         detuning = 2.200000e+00 : f64,
    // CHECK-SAME:         polarization = [0, 1, 2],
    // CHECK-SAME:         wavevector = [-2, 3, 4]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT:   ion.yield %arg1 : !quantum.bit
    // CHECK-NEXT: }
    %4 = quantum.custom "RX"(%arg0) %2 : !quantum.bit

    // CHECK: [[ry1out:%.+]] = ion.parallelprotocol([[rx1out]]) : !quantum.bit {
    // CHECK-NEXT: ^{{.*}}(%arg1: !quantum.bit):
    // CHECK-NEXT: [[rabi1:%.+]] = arith.constant 1.100000e+00 : f64
    // CHECK-NEXT: [[timery1:%.+]] = arith.divf %arg0, [[rabi1]] : f64
    // CHECK-NEXT: ion.pulse([[timery1]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 0 : i64,
    // CHECK-SAME:         rabi = 1.100000e+00 : f64,
    // CHECK-SAME:         detuning = 2.200000e+00 : f64,
    // CHECK-SAME:         polarization = [0, 1, 2],
    // CHECK-SAME:         wavevector = [-2, 3, 4]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timery1]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 1 : i64,
    // CHECK-SAME:         rabi = 1.100000e+00 : f64,
    // CHECK-SAME:         detuning = 2.200000e+00 : f64,
    // CHECK-SAME:         polarization = [0, 1, 2],
    // CHECK-SAME:         wavevector = [-2, 3, 4]>,
    // CHECK-SAME:     phase = 3.1415926535{{[0-9]*}} : f64}
    // CHECK-NEXT:   ion.yield %arg1 : !quantum.bit
    // CHECK-NEXT: }
    %5 = quantum.custom "RY"(%arg0) %4 : !quantum.bit

    // CHECK: [[rx2out:%.+]] = ion.parallelprotocol([[ry1out]]) : !quantum.bit {
    // CHECK-NEXT: ^{{.*}}(%arg1: !quantum.bit):
    // CHECK-NEXT: [[rabi1:%.+]] = arith.constant 1.100000e+00 : f64
    // CHECK-NEXT: [[timerx2:%.+]] = arith.divf %arg0, [[rabi1]] : f64
    // CHECK-NEXT: ion.pulse([[timerx2]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 0 : i64,
    // CHECK-SAME:         rabi = 1.100000e+00 : f64,
    // CHECK-SAME:         detuning = 2.200000e+00 : f64,
    // CHECK-SAME:         polarization = [0, 1, 2],
    // CHECK-SAME:         wavevector = [-2, 3, 4]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timerx2]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 1 : i64,
    // CHECK-SAME:         rabi = 1.100000e+00 : f64,
    // CHECK-SAME:         detuning = 2.200000e+00 : f64,
    // CHECK-SAME:         polarization = [0, 1, 2],
    // CHECK-SAME:         wavevector = [-2, 3, 4]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT:   ion.yield %arg1 : !quantum.bit
    // CHECK-NEXT: }
    %6 = quantum.custom "RX"(%arg0) %5 : !quantum.bit

    // CHECK: [[msout:%.+]] = ion.parallelprotocol([[rx2out]], [[qubit1]]) : !quantum.bit, !quantum.bit {
    // CHECK-NEXT: ^{{.*}}(%arg1: !quantum.bit, %arg2: !quantum.bit):
    // CHECK-NEXT: [[rabi2:%.+]] = arith.constant 1.230000e+00 : f64
    // CHECK-NEXT: [[timems:%.+]] = arith.divf %arg0, [[rabi2]] : f64
    // CHECK-NEXT: ion.pulse([[timems]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 0 : i64,
    // CHECK-SAME:         rabi = 1.230000e+00 : f64,
    // CHECK-SAME:         detuning = 4.560000e+00 : f64,
    // CHECK-SAME:         polarization = [7, 8, 9],
    // CHECK-SAME:         wavevector = [9, 10, 11]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 1 : i64,
    // CHECK-SAME:         rabi = 1.230000e+00 : f64,
    // CHECK-SAME:         detuning = 5.660000e+00 : f64,
    // CHECK-SAME:         polarization = [7, 8, 9],
    // CHECK-SAME:         wavevector = [-9, -10, -11]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 1 : i64,
    // CHECK-SAME:         rabi = 1.230000e+00 : f64,
    // CHECK-SAME:         detuning = 3.4{{.*}} : f64,
    // CHECK-SAME:         polarization = [7, 8, 9],
    // CHECK-SAME:         wavevector = [-9, -10, -11]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems]] : f64) %arg2 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 0 : i64,
    // CHECK-SAME:         rabi = 1.230000e+00 : f64,
    // CHECK-SAME:         detuning = 4.560000e+00 : f64,
    // CHECK-SAME:         polarization = [7, 8, 9],
    // CHECK-SAME:         wavevector = [9, 10, 11]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems]] : f64) %arg2 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 1 : i64,
    // CHECK-SAME:         rabi = 1.230000e+00 : f64,
    // CHECK-SAME:         detuning = 8.960000e+00 : f64,
    // CHECK-SAME:         polarization = [7, 8, 9],
    // CHECK-SAME:         wavevector = [-9, -10, -11]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems]] : f64) %arg2 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 1 : i64,
    // CHECK-SAME:         rabi = 1.230000e+00 : f64,
    // CHECK-SAME:         detuning = 0.1{{.*}} : f64,
    // CHECK-SAME:         polarization = [7, 8, 9],
    // CHECK-SAME:         wavevector = [-9, -10, -11]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT:   ion.yield %arg1, %arg2 : !quantum.bit, !quantum.bit
    // CHECK-NEXT: }
    %7:2 = quantum.custom "MS"(%arg0) %6, %3 : !quantum.bit, !quantum.bit
    return %7#0: !quantum.bit
}
