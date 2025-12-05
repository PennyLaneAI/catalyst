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

// RUN: quantum-opt %s --convert-ion-to-rtio --split-input-file  -verify-diagnostics | FileCheck %s

// RX(1)

// CHECK: memref.global "private" constant @__qubit_map_0 : memref<2xindex> = dense<[0, 1]>
// CHECK-LABEL: func.func @__kernel__()
// CHECK-SAME: attributes {diff_method = "parameter-shift", qnode}
module @circuit {
  func.func public @circuit_0() -> (memref<4xi64>, memref<4xi64>) attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %0 = ion.ion {charge = -1.000000e+00 : f64, levels = [#ion.level<label = "downstate", principal = 6 : i64, spin = 5.000000e-01 : f64, orbital = 0.000000e+00 : f64, nuclear = 5.000000e-01 : f64, spin_orbital = 5.000000e-01 : f64, spin_orbital_nuclear = 0.000000e+00 : f64, spin_orbital_nuclear_magnetization = 0.000000e+00 : f64, energy = 0.000000e+00 : f64>, #ion.level<label = "upstate", principal = 6 : i64, spin = 5.000000e-01 : f64, orbital = 0.000000e+00 : f64, nuclear = 5.000000e-01 : f64, spin_orbital = 5.000000e-01 : f64, spin_orbital_nuclear = 1.000000e+00 : f64, spin_orbital_nuclear_magnetization = 0.000000e+00 : f64, energy = 79438311838.671509 : f64>, #ion.level<label = "estate", principal = 6 : i64, spin = 5.000000e-01 : f64, orbital = 1.000000e+00 : f64, nuclear = 5.000000e-01 : f64, spin_orbital = 5.000000e-01 : f64, spin_orbital_nuclear = 1.000000e+00 : f64, spin_orbital_nuclear_magnetization = -1.000000e+00 : f64, energy = 0x43321C22CEFBBBDF : f64>, #ion.level<label = "estate2", principal = 6 : i64, spin = 5.000000e-01 : f64, orbital = 1.000000e+00 : f64, nuclear = 5.000000e-01 : f64, spin_orbital = 5.000000e-01 : f64, spin_orbital_nuclear = 1.000000e+00 : f64, spin_orbital_nuclear_magnetization = 1.000000e+00 : f64, energy = 0x433456BB2DC221F8 : f64>], mass = 1.710000e+02 : f64, name = "Yb171", position = array<f64: 0.000000e+00, 0.000000e+00, 0.000000e+00>, transitions = [#ion.transition<level_0 = "downstate", level_1 = "estate", einstein_a = 1.000000e+00 : f64, multipole = "E1">, #ion.transition<level_0 = "downstate", level_1 = "estate2", einstein_a = 1.000000e+00 : f64, multipole = "E1">, #ion.transition<level_0 = "upstate", level_1 = "estate", einstein_a = 1.000000e+00 : f64, multipole = "E1">, #ion.transition<level_0 = "upstate", level_1 = "estate2", einstein_a = 1.000000e+00 : f64, multipole = "E1">]} : !ion.ion
    ion.mode {modes = [#ion.phonon<energy = 6283185.307179586 : f64, eigenvector = [7.071068e-01, 0.000000e+00, 0.000000e+00, 7.071068e-01, 0.000000e+00, 0.000000e+00]>]}
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 1.5707963267948966 : f64
    quantum.device shots(%c1_i64) ["/path/to/device.dylib", "oqd", "{}"]
    %1 = quantum.alloc( 2) : !quantum.reg
    %2 = quantum.extract %1[ 1] : !quantum.reg -> !quantum.bit
    %cst_0 = arith.constant 12.566370614359172 : f64
    %3 = arith.remf %cst, %cst_0 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    %4 = arith.cmpf olt, %3, %cst_1 : f64
    %5 = scf.if %4 -> (f64) {
      %16 = arith.addf %3, %cst_0 : f64
      scf.yield %16 : f64
    } else {
      scf.yield %3 : f64
    }
    %cst_2 = arith.constant 62831853071.79586 : f64
    %cst_3 = arith.constant 417140672543652.75 : f64
    %6 = arith.mulf %5, %cst_3 : f64
    %7 = arith.mulf %cst_2, %cst_2 : f64
    %8 = arith.divf %6, %7 : f64
    %9 = builtin.unrealized_conversion_cast %2 : !quantum.bit to !ion.qubit
    // CHECK: %[[EMPTY:.*]] = rtio.empty : !rtio.event
    // CHECK: %[[CH:.*]] = rtio.channel : !rtio.channel<"dds", [2 : i64], 2>
    // CHECK: %[[PULSE:.*]] = rtio.pulse %[[CH]] duration(%{{.*}}) frequency(%{{.*}}) phase(%{{.*}}) wait(%[[EMPTY]])
    // CHECK: return
    %10 = ion.parallelprotocol(%9) : !ion.qubit {
    ^bb0(%arg0: !ion.qubit):
      %16 = ion.pulse(%8 : f64) %arg0 {beam = #ion.beam<transition_index = 0 : i64, rabi = 62831853071.79586 : f64, detuning = 208570336271826.38 : f64, polarization = [1, 0, 0], wavevector = [0, 1, 0]>, phase = 0.000000e+00 : f64} : !ion.pulse
      %17 = ion.pulse(%8 : f64) %arg0 {beam = #ion.beam<transition_index = 2 : i64, rabi = 62831853071.79586 : f64, detuning = 208570336271826.38 : f64, polarization = [1, 0, 0], wavevector = [0, 1, 0]>, phase = 0.000000e+00 : f64} : !ion.pulse
      ion.yield %arg0 : !ion.qubit
    }
    %11 = builtin.unrealized_conversion_cast %10 : !ion.qubit to !quantum.bit
    %12 = quantum.extract %1[ 0] : !quantum.reg -> !quantum.bit
    %13 = quantum.compbasis qubits %12, %11 : !quantum.obs
    %alloca = memref.alloca() : memref<4xf64>
    %alloc = memref.alloc() : memref<4xi64>
    quantum.counts %13 in(%alloca : memref<4xf64>, %alloc : memref<4xi64>)
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<4xi64>
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%alloca : memref<4xf64>) outs(%alloc_4 : memref<4xi64>) {
    ^bb0(%in: f64, %out: i64):
      %16 = arith.fptosi %in : f64 to i64
      linalg.yield %16 : i64
    }
    %14 = quantum.insert %1[ 1], %11 : !quantum.reg, !quantum.bit
    %15 = quantum.insert %14[ 0], %12 : !quantum.reg, !quantum.bit
    quantum.dealloc %15 : !quantum.reg
    quantum.device_release
    return %alloc_4, %alloc : memref<4xi64>, memref<4xi64>
  }
  func.func @setup() {
    quantum.init
    return
  }
  func.func @teardown() {
    quantum.finalize
    return
  }
}

// -----

// Test: CNOT(0, 1)

// CHECK: memref.global "private" constant @__qubit_map_0 : memref<2xindex> = dense<[0, 1]>
// CHECK-LABEL: func.func @__kernel__()
module @cnot_circuit {
  func.func public @circuit_0() -> (memref<4xi64>, memref<4xi64>) attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %0 = ion.ion {charge = -1.000000e+00 : f64, levels = [#ion.level<label = "downstate", principal = 6 : i64, spin = 5.000000e-01 : f64, orbital = 0.000000e+00 : f64, nuclear = 5.000000e-01 : f64, spin_orbital = 5.000000e-01 : f64, spin_orbital_nuclear = 0.000000e+00 : f64, spin_orbital_nuclear_magnetization = 0.000000e+00 : f64, energy = 0.000000e+00 : f64>, #ion.level<label = "upstate", principal = 6 : i64, spin = 5.000000e-01 : f64, orbital = 0.000000e+00 : f64, nuclear = 5.000000e-01 : f64, spin_orbital = 5.000000e-01 : f64, spin_orbital_nuclear = 1.000000e+00 : f64, spin_orbital_nuclear_magnetization = 0.000000e+00 : f64, energy = 79438311838.671509 : f64>, #ion.level<label = "estate", principal = 6 : i64, spin = 5.000000e-01 : f64, orbital = 1.000000e+00 : f64, nuclear = 5.000000e-01 : f64, spin_orbital = 5.000000e-01 : f64, spin_orbital_nuclear = 1.000000e+00 : f64, spin_orbital_nuclear_magnetization = -1.000000e+00 : f64, energy = 0x43321C22CEFBBBDF : f64>, #ion.level<label = "estate2", principal = 6 : i64, spin = 5.000000e-01 : f64, orbital = 1.000000e+00 : f64, nuclear = 5.000000e-01 : f64, spin_orbital = 5.000000e-01 : f64, spin_orbital_nuclear = 1.000000e+00 : f64, spin_orbital_nuclear_magnetization = 1.000000e+00 : f64, energy = 0x433456BB2DC221F8 : f64>], mass = 1.710000e+02 : f64, name = "Yb171", position = array<f64: 0.000000e+00, 0.000000e+00, 0.000000e+00>, transitions = [#ion.transition<level_0 = "downstate", level_1 = "estate", einstein_a = 1.000000e+00 : f64, multipole = "E1">, #ion.transition<level_0 = "downstate", level_1 = "estate2", einstein_a = 1.000000e+00 : f64, multipole = "E1">, #ion.transition<level_0 = "upstate", level_1 = "estate", einstein_a = 1.000000e+00 : f64, multipole = "E1">, #ion.transition<level_0 = "upstate", level_1 = "estate2", einstein_a = 1.000000e+00 : f64, multipole = "E1">]} : !ion.ion
    ion.mode {modes = [#ion.phonon<energy = 6283185.307179586 : f64, eigenvector = [7.071068e-01, 0.000000e+00, 0.000000e+00, 7.071068e-01, 0.000000e+00, 0.000000e+00]>]}
    %cst = arith.constant -1.5707963267948966 : f64
    %cst_0 = arith.constant 1.5707963267948966 : f64
    %c1_i64 = arith.constant 1 : i64
    quantum.device shots(%c1_i64) ["/path/to/device.dylib", "oqd", "{}"]
    %1 = quantum.alloc( 2) : !quantum.reg
    %2 = quantum.extract %1[ 0] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %1[ 1] : !quantum.reg -> !quantum.bit

    %cst_1 = arith.constant 12.566370614359172 : f64
    %4 = arith.remf %cst_0, %cst_1 : f64
    %cst_2 = arith.constant 0.000000e+00 : f64
    %5 = arith.cmpf olt, %4, %cst_2 : f64
    %6 = scf.if %5 -> (f64) {
      %54 = arith.addf %4, %cst_1 : f64
      scf.yield %54 : f64
    } else {
      scf.yield %4 : f64
    }
    %cst_3 = arith.constant 62831853071.79586 : f64
    %cst_4 = arith.constant 417140672543652.75 : f64
    %7 = arith.mulf %6, %cst_4 : f64
    %8 = arith.mulf %cst_3, %cst_3 : f64
    %9 = arith.divf %7, %8 : f64
    %10 = builtin.unrealized_conversion_cast %2 : !quantum.bit to !ion.qubit

    // CHECK: %[[EMPTY:.*]] = rtio.empty : !rtio.event
    // CHECK: %[[CH0:.*]] = rtio.channel : !rtio.channel<"dds", [2 : i64], 0>
    // CHECK: %[[P1:.*]] = rtio.pulse %[[CH0]] {{.*}} wait(%[[EMPTY]])
    %11 = ion.parallelprotocol(%10) : !ion.qubit {
    ^bb0(%arg0: !ion.qubit):
      %54 = ion.pulse(%9 : f64) %arg0 {beam = #ion.beam<transition_index = 0 : i64, rabi = 62831853071.79586 : f64, detuning = 208570336271826.38 : f64, polarization = [1, 0, 0], wavevector = [0, 1, 0]>, phase = 1.5707963267948966 : f64} : !ion.pulse
      %55 = ion.pulse(%9 : f64) %arg0 {beam = #ion.beam<transition_index = 2 : i64, rabi = 62831853071.79586 : f64, detuning = 208570336271826.38 : f64, polarization = [1, 0, 0], wavevector = [0, 1, 0]>, phase = 0.000000e+00 : f64} : !ion.pulse
      ion.yield %arg0 : !ion.qubit
    }
    %12 = builtin.unrealized_conversion_cast %11 : !ion.qubit to !quantum.bit

    %cst_5 = arith.constant 12.566370614359172 : f64
    %13 = arith.remf %cst_0, %cst_5 : f64
    %cst_6 = arith.constant 0.000000e+00 : f64
    %14 = arith.cmpf olt, %13, %cst_6 : f64
    %15 = scf.if %14 -> (f64) {
      %54 = arith.addf %13, %cst_5 : f64
      scf.yield %54 : f64
    } else {
      scf.yield %13 : f64
    }
    %cst_7 = arith.constant 8885765876.3167324 : f64
    %cst_8 = arith.constant 417140672543652.75 : f64
    %16 = arith.mulf %15, %cst_8 : f64
    %17 = arith.mulf %cst_7, %cst_7 : f64
    %18 = arith.divf %16, %17 : f64
    %19 = builtin.unrealized_conversion_cast %12 : !quantum.bit to !ion.qubit
    %20 = builtin.unrealized_conversion_cast %3 : !quantum.bit to !ion.qubit

    // CHECK: %[[SYNC1:.*]] = rtio.sync %[[P1]], %[[EMPTY]] : !rtio.event
    // CHECK: %[[P2:.*]] = rtio.pulse %[[CH0]] {{.*}} wait(%[[SYNC1]])
    // CHECK: %[[CH1:.*]] = rtio.channel : !rtio.channel<"dds", [2 : i64], 1>
    // CHECK: %[[P3:.*]] = rtio.pulse %[[CH1]] {{.*}} wait(%[[SYNC1]])
    // CHECK: %[[CH2:.*]] = rtio.channel : !rtio.channel<"dds", [2 : i64], 2>
    // CHECK: %[[P4:.*]] = rtio.pulse %[[CH2]] {{.*}} wait(%[[SYNC1]])
    // CHECK: %[[CH3:.*]] = rtio.channel : !rtio.channel<"dds", [2 : i64], 3>
    // CHECK: %[[P5:.*]] = rtio.pulse %[[CH3]] {{.*}} wait(%[[SYNC1]])
    // CHECK: %[[SYNC2:.*]] = rtio.sync %[[P2]], %[[P3]], %[[P4]], %[[P5]] : !rtio.event
    %21:2 = ion.parallelprotocol(%19, %20) : !ion.qubit, !ion.qubit {
    ^bb0(%arg0: !ion.qubit, %arg1: !ion.qubit):
      %54 = ion.pulse(%18 : f64) %arg0 {beam = #ion.beam<transition_index = 0 : i64, rabi = 8885765876.3167324 : f64, detuning = 208570336271826.38 : f64, polarization = [0, 1, 0], wavevector = [-1, 0, 0]>, phase = 0.000000e+00 : f64} : !ion.pulse
      %55 = ion.pulse(%18 : f64) %arg0 {beam = #ion.beam<transition_index = 2 : i64, rabi = 8885765876.3167324 : f64, detuning = 208570342555011.69 : f64, polarization = [0, 1, 0], wavevector = [1, 0, 0]>, phase = 0.000000e+00 : f64} : !ion.pulse
      %56 = ion.pulse(%18 : f64) %arg0 {beam = #ion.beam<transition_index = 2 : i64, rabi = 8885765876.3167324 : f64, detuning = 208570329988641.06 : f64, polarization = [0, 1, 0], wavevector = [1, 0, 0]>, phase = 0.000000e+00 : f64} : !ion.pulse
      %57 = ion.pulse(%18 : f64) %arg1 {beam = #ion.beam<transition_index = 0 : i64, rabi = 8885765876.3167324 : f64, detuning = 208570336271826.38 : f64, polarization = [0, 1, 0], wavevector = [-1, 0, 0]>, phase = 0.000000e+00 : f64} : !ion.pulse
      %58 = ion.pulse(%18 : f64) %arg1 {beam = #ion.beam<transition_index = 2 : i64, rabi = 8885765876.3167324 : f64, detuning = 208570341926693.16 : f64, polarization = [0, 1, 0], wavevector = [1, 0, 0]>, phase = 0.000000e+00 : f64} : !ion.pulse
      %59 = ion.pulse(%18 : f64) %arg1 {beam = #ion.beam<transition_index = 2 : i64, rabi = 8885765876.3167324 : f64, detuning = 208570330616959.59 : f64, polarization = [0, 1, 0], wavevector = [1, 0, 0]>, phase = 0.000000e+00 : f64} : !ion.pulse
      ion.yield %arg0, %arg1 : !ion.qubit, !ion.qubit
    }
    %22 = builtin.unrealized_conversion_cast %21#0 : !ion.qubit to !quantum.bit
    %23 = builtin.unrealized_conversion_cast %21#1 : !ion.qubit to !quantum.bit

    %cst_9 = arith.constant 12.566370614359172 : f64
    %24 = arith.remf %cst, %cst_9 : f64
    %cst_10 = arith.constant 0.000000e+00 : f64
    %25 = arith.cmpf olt, %24, %cst_10 : f64
    %26 = scf.if %25 -> (f64) {
      %54 = arith.addf %24, %cst_9 : f64
      scf.yield %54 : f64
    } else {
      scf.yield %24 : f64
    }
    %cst_11 = arith.constant 62831853071.79586 : f64
    %cst_12 = arith.constant 417140672543652.75 : f64
    %27 = arith.mulf %26, %cst_12 : f64
    %28 = arith.mulf %cst_11, %cst_11 : f64
    %29 = arith.divf %27, %28 : f64
    %30 = builtin.unrealized_conversion_cast %22 : !quantum.bit to !ion.qubit

    // CHECK: %[[P6:.*]] = rtio.pulse %[[CH0]] {{.*}} wait(%[[SYNC2]])
    %31 = ion.parallelprotocol(%30) : !ion.qubit {
    ^bb0(%arg0: !ion.qubit):
      %54 = ion.pulse(%29 : f64) %arg0 {beam = #ion.beam<transition_index = 0 : i64, rabi = 62831853071.79586 : f64, detuning = 208570336271826.38 : f64, polarization = [1, 0, 0], wavevector = [0, 1, 0]>, phase = 0.000000e+00 : f64} : !ion.pulse
      %55 = ion.pulse(%29 : f64) %arg0 {beam = #ion.beam<transition_index = 2 : i64, rabi = 62831853071.79586 : f64, detuning = 208570336271826.38 : f64, polarization = [1, 0, 0], wavevector = [0, 1, 0]>, phase = 0.000000e+00 : f64} : !ion.pulse
      ion.yield %arg0 : !ion.qubit
    }
    %32 = builtin.unrealized_conversion_cast %31 : !ion.qubit to !quantum.bit

    %cst_13 = arith.constant 12.566370614359172 : f64
    %33 = arith.remf %cst, %cst_13 : f64
    %cst_14 = arith.constant 0.000000e+00 : f64
    %34 = arith.cmpf olt, %33, %cst_14 : f64
    %35 = scf.if %34 -> (f64) {
      %54 = arith.addf %33, %cst_13 : f64
      scf.yield %54 : f64
    } else {
      scf.yield %33 : f64
    }
    %cst_15 = arith.constant 62831853071.79586 : f64
    %cst_16 = arith.constant 417140672543652.75 : f64
    %36 = arith.mulf %35, %cst_16 : f64
    %37 = arith.mulf %cst_15, %cst_15 : f64
    %38 = arith.divf %36, %37 : f64
    %39 = builtin.unrealized_conversion_cast %23 : !quantum.bit to !ion.qubit

    // CHECK: %[[P7:.*]] = rtio.pulse %[[CH2]] {{.*}} wait(%[[SYNC2]])
    %40 = ion.parallelprotocol(%39) : !ion.qubit {
    ^bb0(%arg0: !ion.qubit):
      %54 = ion.pulse(%38 : f64) %arg0 {beam = #ion.beam<transition_index = 0 : i64, rabi = 62831853071.79586 : f64, detuning = 208570336271826.38 : f64, polarization = [1, 0, 0], wavevector = [0, 1, 0]>, phase = 1.5707963267948966 : f64} : !ion.pulse
      %55 = ion.pulse(%38 : f64) %arg0 {beam = #ion.beam<transition_index = 2 : i64, rabi = 62831853071.79586 : f64, detuning = 208570336271826.38 : f64, polarization = [1, 0, 0], wavevector = [0, 1, 0]>, phase = 0.000000e+00 : f64} : !ion.pulse
      ion.yield %arg0 : !ion.qubit
    }
    %41 = builtin.unrealized_conversion_cast %40 : !ion.qubit to !quantum.bit

    %cst_17 = arith.constant 12.566370614359172 : f64
    %42 = arith.remf %cst, %cst_17 : f64
    %cst_18 = arith.constant 0.000000e+00 : f64
    %43 = arith.cmpf olt, %42, %cst_18 : f64
    %44 = scf.if %43 -> (f64) {
      %54 = arith.addf %42, %cst_17 : f64
      scf.yield %54 : f64
    } else {
      scf.yield %42 : f64
    }
    %cst_19 = arith.constant 62831853071.79586 : f64
    %cst_20 = arith.constant 417140672543652.75 : f64
    %45 = arith.mulf %44, %cst_20 : f64
    %46 = arith.mulf %cst_19, %cst_19 : f64
    %47 = arith.divf %45, %46 : f64
    %48 = builtin.unrealized_conversion_cast %32 : !quantum.bit to !ion.qubit

    // CHECK: %[[P8:.*]] = rtio.pulse %[[CH0]] {{.*}} wait(%[[P6]])
    %49 = ion.parallelprotocol(%48) : !ion.qubit {
    ^bb0(%arg0: !ion.qubit):
      %54 = ion.pulse(%47 : f64) %arg0 {beam = #ion.beam<transition_index = 0 : i64, rabi = 62831853071.79586 : f64, detuning = 208570336271826.38 : f64, polarization = [1, 0, 0], wavevector = [0, 1, 0]>, phase = 1.5707963267948966 : f64} : !ion.pulse
      %55 = ion.pulse(%47 : f64) %arg0 {beam = #ion.beam<transition_index = 2 : i64, rabi = 62831853071.79586 : f64, detuning = 208570336271826.38 : f64, polarization = [1, 0, 0], wavevector = [0, 1, 0]>, phase = 0.000000e+00 : f64} : !ion.pulse
      ion.yield %arg0 : !ion.qubit
    }
    %50 = builtin.unrealized_conversion_cast %49 : !ion.qubit to !quantum.bit

    // CHECK: return
    %51 = quantum.compbasis qubits %41, %50 : !quantum.obs
    %alloca = memref.alloca() : memref<4xf64>
    %alloc = memref.alloc() : memref<4xi64>
    quantum.counts %51 in(%alloca : memref<4xf64>, %alloc : memref<4xi64>)
    %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<4xi64>
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%alloca : memref<4xf64>) outs(%alloc_21 : memref<4xi64>) {
    ^bb0(%in: f64, %out: i64):
      %54 = arith.fptosi %in : f64 to i64
      linalg.yield %54 : i64
    }
    %52 = quantum.insert %1[ 0], %41 : !quantum.reg, !quantum.bit
    %53 = quantum.insert %52[ 1], %50 : !quantum.reg, !quantum.bit
    quantum.dealloc %53 : !quantum.reg
    quantum.device_release
    return %alloc_21, %alloc : memref<4xi64>, memref<4xi64>
  }
  func.func @setup() {
    quantum.init
    return
  }
  func.func @teardown() {
    quantum.finalize
    return
  }
}

// -----

// This test verifies that sequential operations with sync

// CHECK: memref.global "private" constant @__qubit_map_0 : memref<1xindex> = dense<0>
// CHECK-LABEL: func.func @__kernel__()
module @sequential_circuit {
  func.func public @circuit_0() attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %0 = ion.ion {charge = -1.000000e+00 : f64, levels = [#ion.level<label = "downstate", principal = 6 : i64, spin = 5.000000e-01 : f64, orbital = 0.000000e+00 : f64, nuclear = 5.000000e-01 : f64, spin_orbital = 5.000000e-01 : f64, spin_orbital_nuclear = 0.000000e+00 : f64, spin_orbital_nuclear_magnetization = 0.000000e+00 : f64, energy = 0.000000e+00 : f64>, #ion.level<label = "upstate", principal = 6 : i64, spin = 5.000000e-01 : f64, orbital = 0.000000e+00 : f64, nuclear = 5.000000e-01 : f64, spin_orbital = 5.000000e-01 : f64, spin_orbital_nuclear = 1.000000e+00 : f64, spin_orbital_nuclear_magnetization = 0.000000e+00 : f64, energy = 79438311838.671509 : f64>, #ion.level<label = "estate", principal = 6 : i64, spin = 5.000000e-01 : f64, orbital = 1.000000e+00 : f64, nuclear = 5.000000e-01 : f64, spin_orbital = 5.000000e-01 : f64, spin_orbital_nuclear = 1.000000e+00 : f64, spin_orbital_nuclear_magnetization = -1.000000e+00 : f64, energy = 0x43321C22CEFBBBDF : f64>, #ion.level<label = "estate2", principal = 6 : i64, spin = 5.000000e-01 : f64, orbital = 1.000000e+00 : f64, nuclear = 5.000000e-01 : f64, spin_orbital = 5.000000e-01 : f64, spin_orbital_nuclear = 1.000000e+00 : f64, spin_orbital_nuclear_magnetization = 1.000000e+00 : f64, energy = 0x433456BB2DC221F8 : f64>], mass = 1.710000e+02 : f64, name = "Yb171", position = array<f64: 0.000000e+00, 0.000000e+00, 0.000000e+00>, transitions = [#ion.transition<level_0 = "downstate", level_1 = "estate", einstein_a = 1.000000e+00 : f64, multipole = "E1">, #ion.transition<level_0 = "downstate", level_1 = "estate2", einstein_a = 1.000000e+00 : f64, multipole = "E1">, #ion.transition<level_0 = "upstate", level_1 = "estate", einstein_a = 1.000000e+00 : f64, multipole = "E1">, #ion.transition<level_0 = "upstate", level_1 = "estate2", einstein_a = 1.000000e+00 : f64, multipole = "E1">]} : !ion.ion
    ion.mode {modes = [#ion.phonon<energy = 6283185.307179586 : f64, eigenvector = [7.071068e-01, 0.000000e+00, 0.000000e+00]>]}
    %c1_i64 = arith.constant 1 : i64
    quantum.device shots(%c1_i64) ["/path/to/device.dylib", "oqd", "{}"]
    %reg = quantum.alloc( 1) : !quantum.reg
    %q0 = quantum.extract %reg[ 0] : !quantum.reg -> !quantum.bit

    %cst_duration = arith.constant 1.0e-7 : f64

    %ion_q0 = builtin.unrealized_conversion_cast %q0 : !quantum.bit to !ion.qubit

    // CHECK: %[[EMPTY:.*]] = rtio.empty : !rtio.event
    // CHECK: %[[CH:.*]] = rtio.channel : !rtio.channel<"dds", [2 : i64], 0>
    // CHECK: %[[PULSE1:.*]] = rtio.pulse %[[CH]] {{.*}} wait(%[[EMPTY]])
    %out1 = ion.parallelprotocol(%ion_q0) : !ion.qubit {
    ^bb0(%arg0: !ion.qubit):
      %p0 = ion.pulse(%cst_duration : f64) %arg0 {beam = #ion.beam<transition_index = 2 : i64, rabi = 62831853071.79586 : f64, detuning = 208570336271826.38 : f64, polarization = [1, 0, 0], wavevector = [0, 1, 0]>, phase = 0.000000e+00 : f64} : !ion.pulse
      ion.yield %arg0 : !ion.qubit
    }

    // CHECK: %[[PULSE2:.*]] = rtio.pulse %[[CH]] {{.*}} wait(%[[PULSE1]])
    %out2 = ion.parallelprotocol(%out1) : !ion.qubit {
    ^bb0(%arg0: !ion.qubit):
      %p0 = ion.pulse(%cst_duration : f64) %arg0 {beam = #ion.beam<transition_index = 2 : i64, rabi = 62831853071.79586 : f64, detuning = 208570336271826.38 : f64, polarization = [1, 0, 0], wavevector = [0, 1, 0]>, phase = 1.5707963267948966 : f64} : !ion.pulse
      ion.yield %arg0 : !ion.qubit
    }

    // CHECK: return
    %bit_q0 = builtin.unrealized_conversion_cast %out2 : !ion.qubit to !quantum.bit
    %reg2 = quantum.insert %reg[ 0], %bit_q0 : !quantum.reg, !quantum.bit
    quantum.dealloc %reg2 : !quantum.reg
    quantum.device_release
    return
  }
}

