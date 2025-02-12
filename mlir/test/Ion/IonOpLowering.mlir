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

// RUN: quantum-opt %s --convert-ion-to-llvm --split-input-file -verify-diagnostics | FileCheck %s


// CHECK-LABEL: ion_op
func.func public @ion_op(%arg0: tensor<f64>, %arg1: tensor<f64>) attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {

    %c0_i64 = arith.constant 0 : i64
    quantum.device shots(%c0_i64) ["blah.so", "OQD", "{'shots': 0, 'mcmc': False}"]

// CHECK: quantum.device
// CHECK-SAME: ["oqd.qubit", "OQD", "{'shots': 0, 'mcmc': False}ION:
// COM: nlohmann-json library dumps double quotation literals in json strings as \22
// COM: nlohmann-json library dumps JSON specs in alphabetical order
// CHECK-SAME: {
// CHECK-SAME:   \22charge\22:1.0,
// CHECK-SAME:   \22class_\22:\22Ion\22,
// CHECK-SAME:   \22levels\22:[
// CHECK-SAME:     {
// CHECK-SAME:       \22class_\22:\22Level\22,
// CHECK-SAME:       \22energy\22:0.0,
// CHECK-SAME:       \22label\22:\22l0\22,
// CHECK-SAME:       \22nuclear\22:0.5,
// CHECK-SAME:       \22orbital\22:0.0,
// CHECK-SAME:       \22principal\22:6,
// CHECK-SAME:       \22spin\22:0.5,
// CHECK-SAME:       \22spin_orbital\22:0.5,
// CHECK-SAME:       \22spin_orbital_nuclear\22:0.0,
// CHECK-SAME:       \22spin_orbital_nuclear_magnetization\22:0.0
// CHECK-SAME:     },
// CHECK-SAME:     {
// CHECK-SAME:       \22class_\22:\22Level\22,
// CHECK-SAME:       \22energy\22:62.83185307179586,
// CHECK-SAME:       \22label\22:\22l1\22,
// CHECK-SAME:       \22nuclear\22:0.5,
// CHECK-SAME:       \22orbital\22:0.0,
// CHECK-SAME:       \22principal\22:6,
// CHECK-SAME:       \22spin\22:0.5,
// CHECK-SAME:       \22spin_orbital\22:0.5,
// CHECK-SAME:       \22spin_orbital_nuclear\22:1.0,
// CHECK-SAME:       \22spin_orbital_nuclear_magnetization\22:0.0
// CHECK-SAME:     },
// CHECK-SAME:     {
// CHECK-SAME:       \22class_\22:\22Level\22,
// CHECK-SAME:       \22energy\22:628.3185307179587,
// CHECK-SAME:       \22label\22:\22l2\22,
// CHECK-SAME:       \22nuclear\22:0.5,
// CHECK-SAME:       \22orbital\22:1.0,
// CHECK-SAME:       \22principal\22:5,
// CHECK-SAME:       \22spin\22:0.5,
// CHECK-SAME:       \22spin_orbital\22:0.5,
// CHECK-SAME:       \22spin_orbital_nuclear\22:1.0,
// CHECK-SAME:       \22spin_orbital_nuclear_magnetization\22:-1.0
// CHECK-SAME:     },
// CHECK-SAME:     {
// CHECK-SAME:       \22class_\22:\22Level\22,
// CHECK-SAME:       \22energy\22:1256.6370614359173,
// CHECK-SAME:       \22label\22:\22l3\22,
// CHECK-SAME:       \22nuclear\22:0.5,
// CHECK-SAME:       \22orbital\22:1.0,
// CHECK-SAME:       \22principal\22:5,
// CHECK-SAME:       \22spin\22:0.5,
// CHECK-SAME:       \22spin_orbital\22:0.5,
// CHECK-SAME:       \22spin_orbital_nuclear\22:1.0,
// CHECK-SAME:       \22spin_orbital_nuclear_magnetization\22:1.0
// CHECK-SAME:     }
// CHECK-SAME:   ],
// CHECK-SAME:   \22mass\22:171.0,
// CHECK-SAME:   \22position\22:[1.0,2.0,-1.0],
// CHECK-SAME:   \22transitions\22:[
// CHECK-SAME:     {
// CHECK-SAME:       \22class_\22:\22Transition\22,
// CHECK-SAME:       \22einsteinA\22:2.2,
// CHECK-SAME:       \22label\22:\22l0->l2\22,
// CHECK-SAME:       \22level1\22:
// CHECK-SAME:       {
// CHECK-SAME:           \22class_\22:\22Level\22,
// CHECK-SAME:           \22energy\22:0.0,
// CHECK-SAME:           \22label\22:\22l0\22,
// CHECK-SAME:           \22nuclear\22:0.5,
// CHECK-SAME:           \22orbital\22:0.0,
// CHECK-SAME:           \22principal\22:6,
// CHECK-SAME:           \22spin\22:0.5,
// CHECK-SAME:           \22spin_orbital\22:0.5,
// CHECK-SAME:           \22spin_orbital_nuclear\22:0.0,
// CHECK-SAME:           \22spin_orbital_nuclear_magnetization\22:0.0
// CHECK-SAME:       },
// CHECK-SAME:       \22level2\22:
// CHECK-SAME:       {
// CHECK-SAME:         \22class_\22:\22Level\22,
// CHECK-SAME:         \22energy\22:628.3185307179587,
// CHECK-SAME:         \22label\22:\22l2\22,
// CHECK-SAME:         \22nuclear\22:0.5,
// CHECK-SAME:         \22orbital\22:1.0,
// CHECK-SAME:         \22principal\22:5,
// CHECK-SAME:         \22spin\22:0.5,
// CHECK-SAME:         \22spin_orbital\22:0.5,
// CHECK-SAME:         \22spin_orbital_nuclear\22:1.0,
// CHECK-SAME:         \22spin_orbital_nuclear_magnetization\22:-1.0
// CHECK-SAME:       }
// CHECK-SAME:     },
// CHECK-SAME:     {
// CHECK-SAME:       \22class_\22:\22Transition\22,
// CHECK-SAME:       \22einsteinA\22:1.1,
// CHECK-SAME:       \22label\22:\22l1->l2\22,
// CHECK-SAME:       \22level1\22:
// CHECK-SAME:       {
// CHECK-SAME:         \22class_\22:\22Level\22,
// CHECK-SAME:         \22energy\22:62.83185307179586,
// CHECK-SAME:         \22label\22:\22l1\22,
// CHECK-SAME:         \22nuclear\22:0.5,
// CHECK-SAME:         \22orbital\22:0.0,
// CHECK-SAME:         \22principal\22:6,
// CHECK-SAME:         \22spin\22:0.5,
// CHECK-SAME:         \22spin_orbital\22:0.5,
// CHECK-SAME:        \22spin_orbital_nuclear\22:1.0,
// CHECK-SAME:         \22spin_orbital_nuclear_magnetization\22:0.0
// CHECK-SAME:       },
// CHECK-SAME:       \22level2\22:
// CHECK-SAME:       {
// CHECK-SAME:         \22class_\22:\22Level\22,
// CHECK-SAME:         \22energy\22:628.3185307179587,
// CHECK-SAME:         \22label\22:\22l2\22,
// CHECK-SAME:         \22nuclear\22:0.5,
// CHECK-SAME:         \22orbital\22:1.0,
// CHECK-SAME:         \22principal\22:5,
// CHECK-SAME:         \22spin\22:0.5,
// CHECK-SAME:         \22spin_orbital\22:0.5,
// CHECK-SAME:         \22spin_orbital_nuclear\22:1.0,
// CHECK-SAME:         \22spin_orbital_nuclear_magnetization\22:-1.0
// CHECK-SAME:       }
// CHECK-SAME:     },
// CHECK-SAME:     {
// CHECK-SAME:       \22class_\22:\22Transition\22,
// CHECK-SAME:       \22einsteinA\22:3.3,
// CHECK-SAME:       \22label\22:\22l0->l3\22,
// CHECK-SAME:       \22level1\22:
// CHECK-SAME:       {
// CHECK-SAME:           \22class_\22:\22Level\22,
// CHECK-SAME:           \22energy\22:0.0,
// CHECK-SAME:           \22label\22:\22l0\22,
// CHECK-SAME:           \22nuclear\22:0.5,
// CHECK-SAME:           \22orbital\22:0.0,
// CHECK-SAME:           \22principal\22:6,
// CHECK-SAME:           \22spin\22:0.5,
// CHECK-SAME:           \22spin_orbital\22:0.5,
// CHECK-SAME:           \22spin_orbital_nuclear\22:0.0,
// CHECK-SAME:           \22spin_orbital_nuclear_magnetization\22:0.0
// CHECK-SAME:       },
// CHECK-SAME:       \22level2\22:
// CHECK-SAME:       {
// CHECK-SAME:         \22class_\22:\22Level\22,
// CHECK-SAME:         \22energy\22:1256.6370614359173,
// CHECK-SAME:         \22label\22:\22l3\22,
// CHECK-SAME:         \22nuclear\22:0.5,
// CHECK-SAME:         \22orbital\22:1.0,
// CHECK-SAME:         \22principal\22:5,
// CHECK-SAME:         \22spin\22:0.5,
// CHECK-SAME:         \22spin_orbital\22:0.5,
// CHECK-SAME:         \22spin_orbital_nuclear\22:1.0,
// CHECK-SAME:         \22spin_orbital_nuclear_magnetization\22:1.0
// CHECK-SAME:       }
// CHECK-SAME:     }
// CHECK-SAME:   ]
// CHECK-SAME: }
// CHECK-SAME: "]


    %0 = ion.ion {
        charge = 1.000000e+00 : f64,
        mass = 1.710000e+02 : f64,
        name = "Yb171",
        position = array<f64: 1.0, 2.0, -1.0>,
        levels = [
            #ion.level<
                label="l0",
                principal = 6 : i64,
                spin = 5.000000e-01 : f64,
                orbital = 0.000000e+00 : f64,
                nuclear = 5.000000e-01 : f64,
                spin_orbital = 5.000000e-01 : f64,
                spin_orbital_nuclear = 0.000000e+00 : f64,
                spin_orbital_nuclear_magnetization = 0.000000e+00 : f64,
                energy = 0.000000e+00 : f64
            >,
            #ion.level<
                label="l1",
                principal = 6 : i64,
                spin = 5.000000e-01 : f64,
                orbital = 0.000000e+00 : f64,
                nuclear = 5.000000e-01 : f64,
                spin_orbital = 5.000000e-01 : f64,
                spin_orbital_nuclear = 1.000000e+00 : f64,
                spin_orbital_nuclear_magnetization = 0.000000e+00 : f64,
                energy = 62.83185307179586 : f64
            >,
            #ion.level<
                label="l2",
                principal = 5 : i64,
                spin = 5.000000e-01 : f64,
                orbital = 1.000000e+00 : f64,
                nuclear = 5.000000e-01 : f64,
                spin_orbital = 5.000000e-01 : f64,
                spin_orbital_nuclear = 1.000000e+00 : f64,
                spin_orbital_nuclear_magnetization = -1.000000e+00 : f64,
                energy = 628.3185307179587 : f64
            >,
            #ion.level<
                label="l3",
                principal = 5 : i64,
                spin = 5.000000e-01 : f64,
                orbital = 1.000000e+00 : f64,
                nuclear = 5.000000e-01 : f64,
                spin_orbital = 5.000000e-01 : f64,
                spin_orbital_nuclear = 1.000000e+00 : f64,
                spin_orbital_nuclear_magnetization = 1.000000e+00 : f64,
                energy = 1256.6370614359173 : f64
            >
        ],
        transitions = [
            #ion.transition<
                level_0 = "l0",
                level_1 = "l2",
                einstein_a = 2.200000e+00 : f64,
                multipole = "M1"
            >,
            #ion.transition<
                level_0 = "l1",
                level_1 = "l2",
                einstein_a = 1.100000e+00 : f64,
                multipole = "E1"
            >,
            #ion.transition<
                level_0 = "l0",
                level_1 = "l3",
                einstein_a = 3.300000e+00 : f64,
                multipole = "E2"
            >
        ]
    } : !ion.ion
    return
}
