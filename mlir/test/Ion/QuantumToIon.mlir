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

// RUN: quantum-opt %s --pass-pipeline="builtin.module(func.func(gates-to-pulses{\
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
    // CHECK-SAME:        >,
    // CHECK-SAME:        #ion.level<
    // CHECK-SAME:            label = "estate2",
    // CHECK-SAME:            principal = 5
    // CHECK-SAME:            spin = 5.000000e-01
    // CHECK-SAME:            orbital = 1.000000e+00
    // CHECK-SAME:            nuclear = 5.000000e-01
    // CHECK-SAME:            spin_orbital = 5.000000e-01
    // CHECK-SAME:            spin_orbital_nuclear = 1.000000e+00
    // CHECK-SAME:            spin_orbital_nuclear_magnetization = 1.000000e+00
    // CHECK-SAME:            energy = 1256.6369999999999
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
    // CHECK-SAME:            eigenvector = [1.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00]
    // CHECK-SAME:        >
    // CHECK-SAME:    ]
    // CHECK-SAME: }  

    %1 = quantum.alloc( 2) : !quantum.reg

    // CHECK: [[qubit0:%.+]] = quantum.extract %1[ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit1:%.+]] = quantum.extract %1[ 1] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %1[ 0] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %1[ 1] : !quantum.reg -> !quantum.bit

    // CHECK:       [[cst:%.+]] = arith.constant 12.566370614359172 : f64
    // CHECK-NEXT:  [[remainder:%.+]] = arith.remf %arg0, [[cst:%.+]] : f64
    // CHECK-NEXT:  [[cst_0:%.+]] = arith.constant 0.000000e+00 : f64
    // CHECK-NEXT:  [[negative:%.+]] = arith.cmpf olt, [[remainder:%.+]], [[cst_0:%.+]] : f64
    // CHECK-NEXT:  [[normalized_angle:%.+]] = scf.if [[negative:%.+]] -> (f64) {
    // CHECK-NEXT:    [[adjusted:%.+]] = arith.addf [[remainder:%.+]], [[cst:%.+]] : f64
    // CHECK-NEXT:    scf.yield [[adjusted:%.+]] : f64
    // CHECK-NEXT:  } else {
    // CHECK-NEXT:    scf.yield [[remainder:%.+]] : f64
    // CHECK-NEXT:  }
    // CHECK-NEXT: [[cst_1:%.+]] = arith.constant 1.100000e+00 : f64
    // CHECK-NEXT: [[cst_2:%.+]] = arith.constant 4.400000e+00 : f64
    // CHECK-NEXT: [[mult:%.+]] = arith.mulf [[normalized_angle:%.+]], [[cst_2:%.+]] : f64
    // CHECK-NEXT: [[square:%.+]] = arith.mulf [[cst_1:%.+]], [[cst_1:%.+]] : f64
    // CHECK-NEXT: [[timerx1:%.+]] = arith.divf [[mult:%.+]], [[square:%.+]] : f64
    // CHECK-NEXT: [[rx1out:%.+]] = ion.parallelprotocol([[qubit0]]) : !quantum.bit {
    // CHECK-NEXT: ^{{.*}}(%arg1: !quantum.bit):
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
    // CHECK-SAME:         transition_index = 2 : i64,
    // CHECK-SAME:         rabi = 1.100000e+00 : f64,
    // CHECK-SAME:         detuning = 2.200000e+00 : f64,
    // CHECK-SAME:         polarization = [0, 1, 2],
    // CHECK-SAME:         wavevector = [-2, 3, 4]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT:   ion.yield %arg1 : !quantum.bit
    // CHECK-NEXT: }
    %4 = quantum.custom "RX"(%arg0) %2 : !quantum.bit

    // CHECK:       [[cst:%.+]] = arith.constant 12.566370614359172 : f64
    // CHECK-NEXT:  [[remainder:%.+]] = arith.remf %arg0, [[cst:%.+]] : f64
    // CHECK-NEXT:  [[cst_0:%.+]] = arith.constant 0.000000e+00 : f64
    // CHECK-NEXT:  [[negative:%.+]] = arith.cmpf olt, [[remainder:%.+]], [[cst_0:%.+]] : f64
    // CHECK-NEXT:  [[normalized_angle:%.+]] = scf.if [[negative:%.+]] -> (f64) {
    // CHECK-NEXT:    [[adjusted:%.+]] = arith.addf [[remainder:%.+]], [[cst:%.+]] : f64
    // CHECK-NEXT:    scf.yield [[adjusted:%.+]] : f64
    // CHECK-NEXT:  } else {
    // CHECK-NEXT:    scf.yield [[remainder:%.+]] : f64
    // CHECK-NEXT:  }
    // CHECK-NEXT: [[cst_1:%.+]] = arith.constant 1.100000e+00 : f64
    // CHECK-NEXT: [[cst_2:%.+]] = arith.constant 4.400000e+00 : f64
    // CHECK-NEXT: [[mult:%.+]] = arith.mulf [[normalized_angle:%.+]], [[cst_2:%.+]] : f64
    // CHECK-NEXT: [[square:%.+]] = arith.mulf [[cst_1:%.+]], [[cst_1:%.+]] : f64
    // CHECK-NEXT: [[timery1:%.+]] = arith.divf [[mult:%.+]], [[square:%.+]] : f64
    // CHECK-NEXT: [[ry1out:%.+]] = ion.parallelprotocol([[rx1out]]) : !quantum.bit {
    // CHECK-NEXT: ^{{.*}}(%arg1: !quantum.bit):
    // CHECK-NEXT: ion.pulse([[timery1]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 0 : i64,
    // CHECK-SAME:         rabi = 1.100000e+00 : f64,
    // CHECK-SAME:         detuning = 2.200000e+00 : f64,
    // CHECK-SAME:         polarization = [0, 1, 2],
    // CHECK-SAME:         wavevector = [-2, 3, 4]>,
    // CHECK-SAME:     phase = 1.5707963267{{[0-9]*}} : f64}
    // CHECK-NEXT: ion.pulse([[timery1]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 2 : i64,
    // CHECK-SAME:         rabi = 1.100000e+00 : f64,
    // CHECK-SAME:         detuning = 2.200000e+00 : f64,
    // CHECK-SAME:         polarization = [0, 1, 2],
    // CHECK-SAME:         wavevector = [-2, 3, 4]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT:   ion.yield %arg1 : !quantum.bit
    // CHECK-NEXT: }
    %5 = quantum.custom "RY"(%arg0) %4 : !quantum.bit

    // CHECK:       [[cst:%.+]] = arith.constant 12.566370614359172 : f64
    // CHECK-NEXT:  [[remainder:%.+]] = arith.remf %arg0, [[cst:%.+]] : f64
    // CHECK-NEXT:  [[cst_0:%.+]] = arith.constant 0.000000e+00 : f64
    // CHECK-NEXT:  [[negative:%.+]] = arith.cmpf olt, [[remainder:%.+]], [[cst_0:%.+]] : f64
    // CHECK-NEXT:  [[normalized_angle:%.+]] = scf.if [[negative:%.+]] -> (f64) {
    // CHECK-NEXT:    [[adjusted:%.+]] = arith.addf [[remainder:%.+]], [[cst:%.+]] : f64
    // CHECK-NEXT:    scf.yield [[adjusted:%.+]] : f64
    // CHECK-NEXT:  } else {
    // CHECK-NEXT:    scf.yield [[remainder:%.+]] : f64
    // CHECK-NEXT:  }
    // CHECK-NEXT: [[cst_1:%.+]] = arith.constant 1.100000e+00 : f64
    // CHECK-NEXT: [[cst_2:%.+]] = arith.constant 4.400000e+00 : f64
    // CHECK-NEXT: [[mult:%.+]] = arith.mulf [[normalized_angle:%.+]], [[cst_2:%.+]] : f64
    // CHECK-NEXT: [[square:%.+]] = arith.mulf [[cst_1:%.+]], [[cst_1:%.+]] : f64
    // CHECK-NEXT: [[timerx2:%.+]] = arith.divf [[mult:%.+]], [[square:%.+]] : f64
    // CHECK-NEXT: [[rx2out:%.+]] = ion.parallelprotocol([[ry1out]]) : !quantum.bit {
    // CHECK-NEXT: ^{{.*}}(%arg1: !quantum.bit):
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
    // CHECK-SAME:         transition_index = 2 : i64,
    // CHECK-SAME:         rabi = 1.100000e+00 : f64,
    // CHECK-SAME:         detuning = 2.200000e+00 : f64,
    // CHECK-SAME:         polarization = [0, 1, 2],
    // CHECK-SAME:         wavevector = [-2, 3, 4]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT:   ion.yield %arg1 : !quantum.bit
    // CHECK-NEXT: }
    %6 = quantum.custom "RX"(%arg0) %5 : !quantum.bit

    // CHECK:       [[cst:%.+]] = arith.constant 12.566370614359172 : f64
    // CHECK-NEXT:  [[remainder:%.+]] = arith.remf %arg0, [[cst:%.+]] : f64
    // CHECK-NEXT:  [[cst_0:%.+]] = arith.constant 0.000000e+00 : f64
    // CHECK-NEXT:  [[negative:%.+]] = arith.cmpf olt, [[remainder:%.+]], [[cst_0:%.+]] : f64
    // CHECK-NEXT:  [[normalized_angle:%.+]] = scf.if [[negative:%.+]] -> (f64) {
    // CHECK-NEXT:    [[adjusted:%.+]] = arith.addf [[remainder:%.+]], [[cst:%.+]] : f64
    // CHECK-NEXT:    scf.yield [[adjusted:%.+]] : f64
    // CHECK-NEXT:  } else {
    // CHECK-NEXT:    scf.yield [[remainder:%.+]] : f64
    // CHECK-NEXT:  }
    // CHECK-NEXT: [[cst_1:%.+]] = arith.constant 1.230000e+00 : f64
    // CHECK-NEXT: [[cst_2:%.+]] = arith.constant 9.120000e+00 : f64
    // CHECK-NEXT: [[mult:%.+]] = arith.mulf [[normalized_angle:%.+]], [[cst_2:%.+]] : f64
    // CHECK-NEXT: [[square:%.+]] = arith.mulf [[cst_1:%.+]], [[cst_1:%.+]] : f64
    // CHECK-NEXT: [[timems:%.+]] = arith.divf [[mult:%.+]], [[square:%.+]] : f64
    // CHECK-NEXT: [[msout:%.+]] = ion.parallelprotocol([[rx2out]], [[qubit1]]) : !quantum.bit, !quantum.bit {
    // CHECK-NEXT: ^{{.*}}(%arg1: !quantum.bit, %arg2: !quantum.bit):
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
    // CHECK-SAME:         transition_index = 2 : i64,
    // CHECK-SAME:         rabi = 1.230000e+00 : f64,
    // CHECK-SAME:         detuning = 5.660000e+00 : f64,
    // CHECK-SAME:         polarization = [7, 8, 9],
    // CHECK-SAME:         wavevector = [-9, -10, -11]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 2 : i64,
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
    // CHECK-SAME:         transition_index = 2 : i64,
    // CHECK-SAME:         rabi = 1.230000e+00 : f64,
    // CHECK-SAME:         detuning = 8.960000e+00 : f64,
    // CHECK-SAME:         polarization = [7, 8, 9],
    // CHECK-SAME:         wavevector = [-9, -10, -11]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems]] : f64) %arg2 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 2 : i64,
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


// -----


// CHECK-LABEL: example_ion_three_qubit
func.func @example_ion_three_qubit(%arg0: f64) -> (!quantum.bit, !quantum.bit, !quantum.bit) attributes {qnode} {

    // CHECK: {{%.+}} = ion.ion

    %1 = quantum.alloc( 3) : !quantum.reg

    // CHECK: [[qubit0:%.+]] = quantum.extract %1[ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit1:%.+]] = quantum.extract %1[ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit2:%.+]] = quantum.extract %1[ 2] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %1[ 0] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %1[ 1] : !quantum.reg -> !quantum.bit
    %4 = quantum.extract %1[ 2] : !quantum.reg -> !quantum.bit

    // CHECK:       [[cst:%.+]] = arith.constant 12.566370614359172 : f64
    // CHECK-NEXT:  [[remainder:%.+]] = arith.remf %arg0, [[cst:%.+]] : f64
    // CHECK-NEXT:  [[cst_0:%.+]] = arith.constant 0.000000e+00 : f64
    // CHECK-NEXT:  [[negative:%.+]] = arith.cmpf olt, [[remainder:%.+]], [[cst_0:%.+]] : f64
    // CHECK-NEXT:  [[normalized_angle:%.+]] = scf.if [[negative:%.+]] -> (f64) {
    // CHECK-NEXT:    [[adjusted:%.+]] = arith.addf [[remainder:%.+]], [[cst:%.+]] : f64
    // CHECK-NEXT:    scf.yield [[adjusted:%.+]] : f64
    // CHECK-NEXT:  } else {
    // CHECK-NEXT:    scf.yield [[remainder:%.+]] : f64
    // CHECK-NEXT:  }
    // CHECK-NEXT: [[cst_1:%.+]] = arith.constant 1.230000e+00 : f64
    // CHECK-NEXT: [[cst_2:%.+]] = arith.constant 9.120000e+00 : f64
    // CHECK-NEXT: [[mult:%.+]] = arith.mulf [[normalized_angle:%.+]], [[cst_2:%.+]] : f64
    // CHECK-NEXT: [[square:%.+]] = arith.mulf [[cst_1:%.+]], [[cst_1:%.+]] : f64
    // CHECK-NEXT: [[timems1:%.+]] = arith.divf [[mult:%.+]], [[square:%.+]] : f64
    // CHECK-NEXT: [[ms1out:%.+]]:2 = ion.parallelprotocol([[qubit0]], [[qubit1]]) : !quantum.bit, !quantum.bit {
    // CHECK-NEXT: ^{{.*}}(%arg1: !quantum.bit, %arg2: !quantum.bit):
    // CHECK-NEXT: ion.pulse([[timems1]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 0 : i64,
    // CHECK-SAME:         rabi = 1.230000e+00 : f64,
    // CHECK-SAME:         detuning = 4.560000e+00 : f64,
    // CHECK-SAME:         polarization = [7, 8, 9],
    // CHECK-SAME:         wavevector = [9, 10, 11]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems1]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 2 : i64,
    // CHECK-SAME:         rabi = 1.230000e+00 : f64,
    // CHECK-SAME:         detuning = 5.660000e+00 : f64,
    // CHECK-SAME:         polarization = [7, 8, 9],
    // CHECK-SAME:         wavevector = [-9, -10, -11]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems1]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 2 : i64,
    // CHECK-SAME:         rabi = 1.230000e+00 : f64,
    // CHECK-SAME:         detuning = 3.4599999999999995 : f64,
    // CHECK-SAME:         polarization = [7, 8, 9],
    // CHECK-SAME:         wavevector = [-9, -10, -11]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems1]] : f64) %arg2 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 0 : i64,
    // CHECK-SAME:         rabi = 1.230000e+00 : f64,
    // CHECK-SAME:         detuning = 4.560000e+00 : f64,
    // CHECK-SAME:         polarization = [7, 8, 9],
    // CHECK-SAME:         wavevector = [9, 10, 11]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems1]] : f64) %arg2 {
    // CHECK-SAME:    beam = #ion.beam<
    // CHECK-SAME:        transition_index = 2 : i64,
    // CHECK-SAME:        rabi = 1.230000e+00 : f64,
    // CHECK-SAME:        detuning = 8.960000e+00 : f64,
    // CHECK-SAME:        polarization = [7, 8, 9],
    // CHECK-SAME:        wavevector = [-9, -10, -11]>,
    // CHECK-SAME:    phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems1]] : f64) %arg2 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 2 : i64,
    // CHECK-SAME:         rabi = 1.230000e+00 : f64,
    // CHECK-SAME:         detuning = 0.15999999999999925 : f64,
    // CHECK-SAME:         polarization = [7, 8, 9],
    // CHECK-SAME:         wavevector = [-9, -10, -11]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT:   ion.yield %arg1, %arg2 : !quantum.bit, !quantum.bit
    // CHECK-NEXT: }
    %5:2 = quantum.custom "MS"(%arg0) %2, %3 : !quantum.bit, !quantum.bit

    // CHECK:       [[cst:%.+]] = arith.constant 12.566370614359172 : f64
    // CHECK-NEXT:  [[remainder:%.+]] = arith.remf %arg0, [[cst:%.+]] : f64
    // CHECK-NEXT:  [[cst_0:%.+]] = arith.constant 0.000000e+00 : f64
    // CHECK-NEXT:  [[negative:%.+]] = arith.cmpf olt, [[remainder:%.+]], [[cst_0:%.+]] : f64
    // CHECK-NEXT:  [[normalized_angle:%.+]] = scf.if [[negative:%.+]] -> (f64) {
    // CHECK-NEXT:    [[adjusted:%.+]] = arith.addf [[remainder:%.+]], [[cst:%.+]] : f64
    // CHECK-NEXT:    scf.yield [[adjusted:%.+]] : f64
    // CHECK-NEXT:  } else {
    // CHECK-NEXT:    scf.yield [[remainder:%.+]] : f64
    // CHECK-NEXT:  }
    // CHECK-NEXT: [[cst_1:%.+]] = arith.constant 4.560000e+00 : f64
    // CHECK-NEXT: [[cst_2:%.+]] = arith.constant 1.578000e+01 : f64
    // CHECK-NEXT: [[mult:%.+]] = arith.mulf [[normalized_angle:%.+]], [[cst_2:%.+]] : f64
    // CHECK-NEXT: [[square:%.+]] = arith.mulf [[cst_1:%.+]], [[cst_1:%.+]] : f64
    // CHECK-NEXT: [[timems2:%.+]] = arith.divf [[mult:%.+]], [[square:%.+]] : f64
    // CHECK-NEXT: [[ms2out:%.+]]:2 = ion.parallelprotocol([[ms1out]]#0, [[qubit2]]) : !quantum.bit, !quantum.bit {
    // CHECK-NEXT: ^{{.*}}(%arg1: !quantum.bit, %arg2: !quantum.bit):
    // CHECK-NEXT: ion.pulse([[timems2]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 0 : i64,
    // CHECK-SAME:         rabi = 4.560000e+00 : f64,
    // CHECK-SAME:         detuning = 7.8899999999999996 : f64,
    // CHECK-SAME:         polarization = [1, 2, 3],
    // CHECK-SAME:         wavevector = [-3, 4, 5]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems2]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 2 : i64,
    // CHECK-SAME:         rabi = 4.560000e+00 : f64,
    // CHECK-SAME:         detuning = 8.990000e+00 : f64,
    // CHECK-SAME:         polarization = [1, 2, 3],
    // CHECK-SAME:         wavevector = [3, -4, -5]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems2]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 2 : i64,
    // CHECK-SAME:         rabi = 4.560000e+00 : f64,
    // CHECK-SAME:         detuning = 6.7899999999999991 : f64,
    // CHECK-SAME:         polarization = [1, 2, 3],
    // CHECK-SAME:         wavevector = [3, -4, -5]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems2]] : f64) %arg2 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 0 : i64,
    // CHECK-SAME:         rabi = 4.560000e+00 : f64,
    // CHECK-SAME:         detuning = 7.8899999999999996 : f64,
    // CHECK-SAME:         polarization = [1, 2, 3],
    // CHECK-SAME:         wavevector = [-3, 4, 5]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems2]] : f64) %arg2 {
    // CHECK-SAME:    beam = #ion.beam<
    // CHECK-SAME:        transition_index = 2 : i64,
    // CHECK-SAME:        rabi = 4.560000e+00 : f64,
    // CHECK-SAME:        detuning = 1.559000e+01 : f64,
    // CHECK-SAME:        polarization = [1, 2, 3],
    // CHECK-SAME:        wavevector = [3, -4, -5]>,
    // CHECK-SAME:    phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems2]] : f64) %arg2 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 2 : i64,
    // CHECK-SAME:         rabi = 4.560000e+00 : f64,
    // CHECK-SAME:         detuning = 0.1899999999999995 : f64,
    // CHECK-SAME:         polarization = [1, 2, 3],
    // CHECK-SAME:         wavevector = [3, -4, -5]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT:   ion.yield %arg1, %arg2 : !quantum.bit, !quantum.bit
    // CHECK-NEXT: }
    %6:2 = quantum.custom "MS"(%arg0) %5#0, %4 : !quantum.bit, !quantum.bit

    // CHECK:       [[cst:%.+]] = arith.constant 12.566370614359172 : f64
    // CHECK-NEXT:  [[remainder:%.+]] = arith.remf %arg0, [[cst:%.+]] : f64
    // CHECK-NEXT:  [[cst_0:%.+]] = arith.constant 0.000000e+00 : f64
    // CHECK-NEXT:  [[negative:%.+]] = arith.cmpf olt, [[remainder:%.+]], [[cst_0:%.+]] : f64
    // CHECK-NEXT:  [[normalized_angle:%.+]] = scf.if [[negative:%.+]] -> (f64) {
    // CHECK-NEXT:    [[adjusted:%.+]] = arith.addf [[remainder:%.+]], [[cst:%.+]] : f64
    // CHECK-NEXT:    scf.yield [[adjusted:%.+]] : f64
    // CHECK-NEXT:  } else {
    // CHECK-NEXT:    scf.yield [[remainder:%.+]] : f64
    // CHECK-NEXT:  }
    // CHECK-NEXT: [[cst_1:%.+]] = arith.constant 99.989999999999994 : f64
    // CHECK-NEXT: [[cst_2:%.+]] = arith.constant 2.002000e+02 : f64
    // CHECK-NEXT: [[mult:%.+]] = arith.mulf [[normalized_angle:%.+]], [[cst_2:%.+]] : f64
    // CHECK-NEXT: [[square:%.+]] = arith.mulf [[cst_1:%.+]], [[cst_1:%.+]] : f64
    // CHECK-NEXT: [[timems3:%.+]] = arith.divf [[mult:%.+]], [[square:%.+]] : f64
    // CHECK-NEXT: [[ms3out:%.+]]:2 = ion.parallelprotocol([[ms1out]]#1, [[ms2out]]#1) : !quantum.bit, !quantum.bit {
    // CHECK-NEXT: ^{{.*}}(%arg1: !quantum.bit, %arg2: !quantum.bit):
    // CHECK-NEXT: [[p1:%.+]] = ion.pulse([[timems3]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 0 : i64,
    // CHECK-SAME:         rabi = 99.989999999999994 : f64,
    // CHECK-SAME:         detuning = 1.001000e+02 : f64,
    // CHECK-SAME:         polarization = [37, 42, 43],
    // CHECK-SAME:         wavevector = [-42, -37, -43]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64} : !ion.pulse
    // CHECK-NEXT: [[p2:%.+]] = ion.pulse([[timems3]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 2 : i64,
    // CHECK-SAME:         rabi = 99.989999999999994 : f64,
    // CHECK-SAME:         detuning = 1.045000e+02 : f64,
    // CHECK-SAME:         polarization = [37, 42, 43],
    // CHECK-SAME:         wavevector = [42, 37, 43]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64} : !ion.pulse
    // CHECK-NEXT: [[p3:%.+]] = ion.pulse([[timems3]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 2 : i64,
    // CHECK-SAME:         rabi = 99.989999999999994 : f64,
    // CHECK-SAME:         detuning = 95.699999999999989 : f64,
    // CHECK-SAME:         polarization = [37, 42, 43],
    // CHECK-SAME:         wavevector = [42, 37, 43]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64} : !ion.pulse
    // CHECK-NEXT: [[p4:%.+]] = ion.pulse([[timems3]] : f64) %arg2 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 0 : i64,
    // CHECK-SAME:         rabi = 99.989999999999994 : f64,
    // CHECK-SAME:         detuning = 1.001000e+02 : f64,
    // CHECK-SAME:         polarization = [37, 42, 43],
    // CHECK-SAME:         wavevector = [-42, -37, -43]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64} : !ion.pulse
    // CHECK-NEXT: [[p5:%.+]] = ion.pulse([[timems3]] : f64) %arg2 {
    // CHECK-SAME:    beam = #ion.beam<
    // CHECK-SAME:        transition_index = 2 : i64,
    // CHECK-SAME:        rabi = 99.989999999999994 : f64,
    // CHECK-SAME:        detuning = 1.078000e+02 : f64,
    // CHECK-SAME:        polarization = [37, 42, 43],
    // CHECK-SAME:        wavevector = [42, 37, 43]>,
    // CHECK-SAME:    phase = 0.000000e+00 : f64} : !ion.pulse
    // CHECK-NEXT: [[p6:%.+]] = ion.pulse([[timems3]] : f64) %arg2 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 2 : i64,
    // CHECK-SAME:         rabi = 99.989999999999994 : f64,
    // CHECK-SAME:         detuning = 92.399999999999991 : f64,
    // CHECK-SAME:         polarization = [37, 42, 43],
    // CHECK-SAME:         wavevector = [42, 37, 43]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64} : !ion.pulse
    // CHECK-NEXT:   ion.yield %arg1, %arg2 : !quantum.bit, !quantum.bit
    // CHECK-NEXT: }
    %7:2 = quantum.custom "MS"(%arg0) %5#1, %6#1 : !quantum.bit, !quantum.bit
    return %6#0, %7#0, %7#1: !quantum.bit, !quantum.bit, !quantum.bit
}
