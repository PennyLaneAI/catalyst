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
    // CHECK-NEXT: [[ionqubit0:%.+]] = builtin.unrealized_conversion_cast [[qubit0]] : !quantum.bit to !ion.qubit
    // CHECK-NEXT: [[rx1out_ion:%.+]] = ion.parallelprotocol([[ionqubit0]]) : !ion.qubit {
    // CHECK-NEXT: ^{{.*}}(%arg1: !ion.qubit):
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
    // CHECK-NEXT:   ion.yield %arg1 : !ion.qubit
    // CHECK-NEXT: }
    // CHECK-NEXT: [[rx1out:%.+]] = builtin.unrealized_conversion_cast [[rx1out_ion]] : !ion.qubit to !quantum.bit
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
    // CHECK-NEXT: [[rx1out_ion:%.+]] = builtin.unrealized_conversion_cast [[rx1out]] : !quantum.bit to !ion.qubit
    // CHECK-NEXT: [[ry1out_ion:%.+]] = ion.parallelprotocol([[rx1out_ion]]) : !ion.qubit {
    // CHECK-NEXT: ^{{.*}}(%arg1: !ion.qubit):
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
    // CHECK-NEXT:   ion.yield %arg1 : !ion.qubit
    // CHECK-NEXT: }
    // CHECK-NEXT: [[ry1out:%.+]] = builtin.unrealized_conversion_cast [[ry1out_ion]] : !ion.qubit to !quantum.bit
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
    // CHECK-NEXT: [[ry1out_ion_2:%.+]] = builtin.unrealized_conversion_cast [[ry1out]] : !quantum.bit to !ion.qubit
    // CHECK-NEXT: [[rx2out_ion:%.+]] = ion.parallelprotocol([[ry1out_ion_2]]) : !ion.qubit {
    // CHECK-NEXT: ^{{.*}}(%arg1: !ion.qubit):
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
    // CHECK-NEXT:   ion.yield %arg1 : !ion.qubit
    // CHECK-NEXT: }
    // CHECK-NEXT: [[rx2out:%.+]] = builtin.unrealized_conversion_cast [[rx2out_ion]] : !ion.qubit to !quantum.bit
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
    // CHECK-NEXT: [[rx2out_ion_2:%.+]] = builtin.unrealized_conversion_cast [[rx2out]] : !quantum.bit to !ion.qubit
    // CHECK-NEXT: [[qubit1_ion:%.+]] = builtin.unrealized_conversion_cast [[qubit1]] : !quantum.bit to !ion.qubit
    // CHECK-NEXT: [[msout_ion:%.+]]:2 = ion.parallelprotocol([[rx2out_ion_2]], [[qubit1_ion]]) : !ion.qubit, !ion.qubit {
    // CHECK-NEXT: ^{{.*}}(%arg1: !ion.qubit, %arg2: !ion.qubit):
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
    // CHECK-SAME:         rabi = 99.989999999999994 : f64,
    // CHECK-SAME:         detuning = 1.001000e+02 : f64,
    // CHECK-SAME:         polarization = [37, 42, 43],
    // CHECK-SAME:         wavevector = [-42, -37, -43]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 2 : i64,
    // CHECK-SAME:         rabi = 4.560000e+00 : f64,
    // CHECK-SAME:         detuning = 7.8899999999999996 : f64,
    // CHECK-SAME:         polarization = [1, 2, 3],
    // CHECK-SAME:         wavevector = [-3, 4, 5]>,
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
    // CHECK-SAME:         rabi = 99.989999999999994 : f64,
    // CHECK-SAME:         detuning = 1.001000e+02 : f64,
    // CHECK-SAME:         polarization = [37, 42, 43],
    // CHECK-SAME:         wavevector = [-42, -37, -43]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems]] : f64) %arg2 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 2 : i64,
    // CHECK-SAME:         rabi = 4.560000e+00 : f64,
    // CHECK-SAME:         detuning = 7.8899999999999996 : f64,
    // CHECK-SAME:         polarization = [1, 2, 3],
    // CHECK-SAME:         wavevector = [-3, 4, 5]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT:   ion.yield %arg1, %arg2 : !ion.qubit, !ion.qubit
    // CHECK-NEXT: }
    // CHECK-NEXT: [[msout_0:%.+]] = builtin.unrealized_conversion_cast [[msout_ion]]#0 : !ion.qubit to !quantum.bit
    // CHECK-NEXT: [[msout_1:%.+]] = builtin.unrealized_conversion_cast [[msout_ion]]#1 : !ion.qubit to !quantum.bit
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
    // CHECK-NEXT: [[ionqubit0_ms:%.+]] = builtin.unrealized_conversion_cast [[qubit0]] : !quantum.bit to !ion.qubit
    // CHECK-NEXT: [[ionqubit1_ms:%.+]] = builtin.unrealized_conversion_cast [[qubit1]] : !quantum.bit to !ion.qubit
    // CHECK-NEXT: [[ms1out_ion:%.+]]:2 = ion.parallelprotocol([[ionqubit0_ms]], [[ionqubit1_ms]]) : !ion.qubit, !ion.qubit {
    // CHECK-NEXT: ^{{.*}}(%arg1: !ion.qubit, %arg2: !ion.qubit):
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
    // CHECK-SAME:         rabi = 99.989999999999994 : f64,
    // CHECK-SAME:         detuning = 1.001000e+02 : f64,
    // CHECK-SAME:         polarization = [37, 42, 43],
    // CHECK-SAME:         wavevector = [-42, -37, -43]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems1]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 2 : i64,
    // CHECK-SAME:         rabi = 4.560000e+00 : f64,
    // CHECK-SAME:         detuning = 7.8899999999999996 : f64,
    // CHECK-SAME:         polarization = [1, 2, 3],
    // CHECK-SAME:         wavevector = [-3, 4, 5]>,
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
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 2 : i64,
    // CHECK-SAME:         rabi = 99.989999999999994 : f64,
    // CHECK-SAME:         detuning = 1.001000e+02 : f64,
    // CHECK-SAME:         polarization = [37, 42, 43],
    // CHECK-SAME:         wavevector = [-42, -37, -43]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems1]] : f64) %arg2 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 2 : i64,
    // CHECK-SAME:         rabi = 4.560000e+00 : f64,
    // CHECK-SAME:         detuning = 7.8899999999999996 : f64,
    // CHECK-SAME:         polarization = [1, 2, 3],
    // CHECK-SAME:         wavevector = [-3, 4, 5]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT:   ion.yield %arg1, %arg2 : !ion.qubit, !ion.qubit
    // CHECK-NEXT: }
    // CHECK-NEXT: [[ms1out_0:%.+]] = builtin.unrealized_conversion_cast [[ms1out_ion]]#0 : !ion.qubit to !quantum.bit
    // CHECK-NEXT: [[ms1out_1:%.+]] = builtin.unrealized_conversion_cast [[ms1out_ion]]#1 : !ion.qubit to !quantum.bit
    %5:2 = quantum.custom "MS"(%arg0) %2, %3 : !quantum.bit, !quantum.bit

    // MS gate 2: pair (0,2) -> beamBaseIndex=3: global=beams2[3], red=beams2[4], blue=beams2[5]
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
    // CHECK-NEXT: [[cst_1:%.+]] = arith.constant 2.000000e+00 : f64
    // CHECK-NEXT: [[cst_2:%.+]] = arith.constant 1.000000e+01 : f64
    // CHECK-NEXT: [[mult:%.+]] = arith.mulf [[normalized_angle:%.+]], [[cst_2:%.+]] : f64
    // CHECK-NEXT: [[square:%.+]] = arith.mulf [[cst_1:%.+]], [[cst_1:%.+]] : f64
    // CHECK-NEXT: [[timems2:%.+]] = arith.divf [[mult:%.+]], [[square:%.+]] : f64
    // CHECK-NEXT: [[ms1out_0_ion:%.+]] = builtin.unrealized_conversion_cast [[ms1out_0]] : !quantum.bit to !ion.qubit
    // CHECK-NEXT: [[qubit2_ion:%.+]] = builtin.unrealized_conversion_cast [[qubit2]] : !quantum.bit to !ion.qubit
    // CHECK-NEXT: [[ms2out_ion:%.+]]:2 = ion.parallelprotocol([[ms1out_0_ion]], [[qubit2_ion]]) : !ion.qubit, !ion.qubit {
    // CHECK-NEXT: ^{{.*}}(%arg1: !ion.qubit, %arg2: !ion.qubit):
    // CHECK-NEXT: ion.pulse([[timems2]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 0 : i64,
    // CHECK-SAME:         rabi = 2.000000e+00 : f64,
    // CHECK-SAME:         detuning = 5.000000e+00 : f64,
    // CHECK-SAME:         polarization = [10, 11, 12],
    // CHECK-SAME:         wavevector = [12, 13, 14]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems2]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 2 : i64,
    // CHECK-SAME:         rabi = 1.000000e+01 : f64,
    // CHECK-SAME:         detuning = 1.100000e+01 : f64,
    // CHECK-SAME:         polarization = [10, 11, 12],
    // CHECK-SAME:         wavevector = [-12, -13, -14]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems2]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 2 : i64,
    // CHECK-SAME:         rabi = 5.000000e+00 : f64,
    // CHECK-SAME:         detuning = 9.000000e+00 : f64,
    // CHECK-SAME:         polarization = [10, 11, 12],
    // CHECK-SAME:         wavevector = [-12, -13, -14]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems2]] : f64) %arg2 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 0 : i64,
    // CHECK-SAME:         rabi = 2.000000e+00 : f64,
    // CHECK-SAME:         detuning = 5.000000e+00 : f64,
    // CHECK-SAME:         polarization = [10, 11, 12],
    // CHECK-SAME:         wavevector = [12, 13, 14]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems2]] : f64) %arg2 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 2 : i64,
    // CHECK-SAME:         rabi = 1.000000e+01 : f64,
    // CHECK-SAME:         detuning = 1.100000e+01 : f64,
    // CHECK-SAME:         polarization = [10, 11, 12],
    // CHECK-SAME:         wavevector = [-12, -13, -14]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT: ion.pulse([[timems2]] : f64) %arg2 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 2 : i64,
    // CHECK-SAME:         rabi = 5.000000e+00 : f64,
    // CHECK-SAME:         detuning = 9.000000e+00 : f64,
    // CHECK-SAME:         polarization = [10, 11, 12],
    // CHECK-SAME:         wavevector = [-12, -13, -14]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64}
    // CHECK-NEXT:   ion.yield %arg1, %arg2 : !ion.qubit, !ion.qubit
    // CHECK-NEXT: }
    // CHECK-NEXT: [[ms2out_0:%.+]] = builtin.unrealized_conversion_cast [[ms2out_ion]]#0 : !ion.qubit to !quantum.bit
    // CHECK-NEXT: [[ms2out_1:%.+]] = builtin.unrealized_conversion_cast [[ms2out_ion]]#1 : !ion.qubit to !quantum.bit
    %6:2 = quantum.custom "MS"(%arg0) %5#0, %4 : !quantum.bit, !quantum.bit

    // MS gate 3: pair (1,2) -> beamBaseIndex=6: global=beams2[6], red=beams2[7], blue=beams2[8]
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
    // CHECK-NEXT: [[cst_1:%.+]] = arith.constant 3.000000e+00 : f64
    // CHECK-NEXT: [[cst_2:%.+]] = arith.constant 1.200000e+01 : f64
    // CHECK-NEXT: [[mult:%.+]] = arith.mulf [[normalized_angle:%.+]], [[cst_2:%.+]] : f64
    // CHECK-NEXT: [[square:%.+]] = arith.mulf [[cst_1:%.+]], [[cst_1:%.+]] : f64
    // CHECK-NEXT: [[timems3:%.+]] = arith.divf [[mult:%.+]], [[square:%.+]] : f64
    // CHECK-NEXT: [[ms1out_1_ion:%.+]] = builtin.unrealized_conversion_cast [[ms1out_1]] : !quantum.bit to !ion.qubit
    // CHECK-NEXT: [[ms2out_1_ion:%.+]] = builtin.unrealized_conversion_cast [[ms2out_1]] : !quantum.bit to !ion.qubit
    // CHECK-NEXT: [[ms3out_ion:%.+]]:2 = ion.parallelprotocol([[ms1out_1_ion]], [[ms2out_1_ion]]) : !ion.qubit, !ion.qubit {
    // CHECK-NEXT: ^{{.*}}(%arg1: !ion.qubit, %arg2: !ion.qubit):
    // CHECK-NEXT: [[p1:%.+]] = ion.pulse([[timems3]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 0 : i64,
    // CHECK-SAME:         rabi = 3.000000e+00 : f64,
    // CHECK-SAME:         detuning = 6.000000e+00 : f64,
    // CHECK-SAME:         polarization = [20, 21, 22],
    // CHECK-SAME:         wavevector = [22, 23, 24]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64} : !ion.pulse
    // CHECK-NEXT: [[p2:%.+]] = ion.pulse([[timems3]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 2 : i64,
    // CHECK-SAME:         rabi = 1.200000e+01 : f64,
    // CHECK-SAME:         detuning = 1.300000e+01 : f64,
    // CHECK-SAME:         polarization = [20, 21, 22],
    // CHECK-SAME:         wavevector = [-22, -23, -24]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64} : !ion.pulse
    // CHECK-NEXT: [[p3:%.+]] = ion.pulse([[timems3]] : f64) %arg1 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 2 : i64,
    // CHECK-SAME:         rabi = 6.000000e+00 : f64,
    // CHECK-SAME:         detuning = 1.000000e+01 : f64,
    // CHECK-SAME:         polarization = [20, 21, 22],
    // CHECK-SAME:         wavevector = [-22, -23, -24]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64} : !ion.pulse
    // CHECK-NEXT: [[p4:%.+]] = ion.pulse([[timems3]] : f64) %arg2 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 0 : i64,
    // CHECK-SAME:         rabi = 3.000000e+00 : f64,
    // CHECK-SAME:         detuning = 6.000000e+00 : f64,
    // CHECK-SAME:         polarization = [20, 21, 22],
    // CHECK-SAME:         wavevector = [22, 23, 24]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64} : !ion.pulse
    // CHECK-NEXT: [[p5:%.+]] = ion.pulse([[timems3]] : f64) %arg2 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 2 : i64,
    // CHECK-SAME:         rabi = 1.200000e+01 : f64,
    // CHECK-SAME:         detuning = 1.300000e+01 : f64,
    // CHECK-SAME:         polarization = [20, 21, 22],
    // CHECK-SAME:         wavevector = [-22, -23, -24]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64} : !ion.pulse
    // CHECK-NEXT: [[p6:%.+]] = ion.pulse([[timems3]] : f64) %arg2 {
    // CHECK-SAME:     beam = #ion.beam<
    // CHECK-SAME:         transition_index = 2 : i64,
    // CHECK-SAME:         rabi = 6.000000e+00 : f64,
    // CHECK-SAME:         detuning = 1.000000e+01 : f64,
    // CHECK-SAME:         polarization = [20, 21, 22],
    // CHECK-SAME:         wavevector = [-22, -23, -24]>,
    // CHECK-SAME:     phase = 0.000000e+00 : f64} : !ion.pulse
    // CHECK-NEXT:   ion.yield %arg1, %arg2 : !ion.qubit, !ion.qubit
    // CHECK-NEXT: }
    // CHECK-NEXT: [[ms3out_0:%.+]] = builtin.unrealized_conversion_cast [[ms3out_ion]]#0 : !ion.qubit to !quantum.bit
    // CHECK-NEXT: [[ms3out_1:%.+]] = builtin.unrealized_conversion_cast [[ms3out_ion]]#1 : !ion.qubit to !quantum.bit
    %7:2 = quantum.custom "MS"(%arg0) %5#1, %6#1 : !quantum.bit, !quantum.bit
    return %6#0, %7#0, %7#1: !quantum.bit, !quantum.bit, !quantum.bit
}
