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

    %0 = ion.ion {
        charge = 1.000000e+00 : f64, 
        mass = 1.710000e+02 : f64, 
        name = "Yb171", 
        position = array<f64: 1.0, 2.0, -1.0>,
        levels = [
            #ion.level<
                label="downstate",
                principal = 6 : i64, 
                spin = 4.000000e-01 : f64, 
                orbital = 5.000000e-01 : f64, 
                nuclear = 6.000000e-01 : f64, 
                spin_orbital = 8.000000e-01 : f64, 
                spin_orbital_nuclear = 9.000000e-01 : f64, 
                spin_orbital_nuclear_magnetization = 1.000000e+00 : f64, 
                energy = 0.000000e+00 : f64
            >, 
            #ion.level<
                label="estate",
                principal = 6 : i64, 
                spin = 1.400000e+00 : f64, 
                orbital = 1.500000e+00 : f64, 
                nuclear = 1.600000e+00 : f64, 
                spin_orbital = 1.800000e+00 : f64, 
                spin_orbital_nuclear = 1.900000e+00 : f64, 
                spin_orbital_nuclear_magnetization = 2.000000e+00 : f64, 
                energy = 1.264300e+10 : f64
            >, 
            #ion.level<
                label="upstate",
                principal = 5 : i64, 
                spin = 2.400000e+00 : f64, 
                orbital = 2.500000e+00 : f64, 
                nuclear = 2.600000e+00 : f64, 
                spin_orbital = 2.800000e+00 : f64, 
                spin_orbital_nuclear = 2.900000e+00 : f64, 
                spin_orbital_nuclear_magnetization = 3.000000e+00 : f64, 
                energy = 8.115200e+14 : f64
            >
        ], 
        transitions = [
            #ion.transition<
                level_0 = "downstate",
                level_1 = "estate",
                einstein_a = 2.200000e+00 : f64
            >, 
            #ion.transition<
                level_0 = "downstate", 
                level_1 = "upstate", 
                einstein_a = 1.100000e+00 : f64
            >, 
            #ion.transition<
                level_0 = "estate", 
                level_1 = "upstate", 
                einstein_a = 3.300000e+00 : f64
            >
        ]
    } : !ion.ion
    return
}
