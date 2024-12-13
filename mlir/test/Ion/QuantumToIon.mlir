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

// RUN: quantum-opt %s --quantum-to-ion --split-input-file -verify-diagnostics | FileCheck %s

// COM: the physical parameters come from
// COM: frontend/catalyst/third_party/oqd/src/oqd_gate_decomposition_parameters.toml

func.func @example_ion(%arg0: f64) -> !quantum.bit {
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

    %1 = quantum.alloc( 2) : !quantum.reg
    %2 = quantum.extract %1[ 0] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %1[ 1] : !quantum.reg -> !quantum.bit

    // CHECK: [[rx1out:%.+]] = ion.parallelprotocol(%2) : !quantum.bit {
    // CHECK-NEXT: ^{{.*}}(%arg1: !quantum.bit):
    // CHECK-NEXT: ion.pulse(%arg0 : f64) %arg1 {beam = #ion.beam<transition_index = 0 : i64, rabi = 1.100000e+00 : f64, detuning = 2.200000e+00 : f64, polarization = dense<[0, 1]> : vector<2xi64>, wavevector = dense<[-2, 3]> : vector<2xi64>>, phase = 0.00{{.*}} : f64}
    // CHECK-NEXT: ion.pulse(%arg0 : f64) %arg1 {beam = #ion.beam<transition_index = 1 : i64, rabi = 1.100000e+00 : f64, detuning = 2.200000e+00 : f64, polarization = dense<[0, 1]> : vector<2xi64>, wavevector = dense<[-2, 3]> : vector<2xi64>>, phase = 0.00{{.*}} : f64}
    // CHECK-NEXT: }
    %4 = quantum.custom "RX"(%arg0) %2 : !quantum.bit

    // CHECK: [[ry1out:%.+]] = ion.parallelprotocol([[rx1out]]) : !quantum.bit {
    // CHECK-NEXT: ^{{.*}}(%arg1: !quantum.bit):
    // CHECK-NEXT: ion.pulse(%arg0 : f64) %arg1 {beam = #ion.beam<transition_index = 0 : i64, rabi = 1.100000e+00 : f64, detuning = 2.200000e+00 : f64, polarization = dense<[0, 1]> : vector<2xi64>, wavevector = dense<[-2, 3]> : vector<2xi64>>, phase = 0.00{{.*}} : f64}
    // CHECK-NEXT: ion.pulse(%arg0 : f64) %arg1 {beam = #ion.beam<transition_index = 1 : i64, rabi = 1.100000e+00 : f64, detuning = 2.200000e+00 : f64, polarization = dense<[0, 1]> : vector<2xi64>, wavevector = dense<[-2, 3]> : vector<2xi64>>, phase = 3.14{{.*}} : f64}
    // CHECK-NEXT: }
    %5 = quantum.custom "RY"(%arg0) %4 : !quantum.bit

    // CHECK: [[rx2out:%.+]] = ion.parallelprotocol([[ry1out]]) : !quantum.bit {
    // CHECK-NEXT: ^{{.*}}(%arg1: !quantum.bit):
    // CHECK-NEXT: ion.pulse(%arg0 : f64) %arg1 {beam = #ion.beam<transition_index = 0 : i64, rabi = 1.100000e+00 : f64, detuning = 2.200000e+00 : f64, polarization = dense<[0, 1]> : vector<2xi64>, wavevector = dense<[-2, 3]> : vector<2xi64>>, phase = 0.00{{.*}} : f64}
    // CHECK-NEXT: ion.pulse(%arg0 : f64) %arg1 {beam = #ion.beam<transition_index = 1 : i64, rabi = 1.100000e+00 : f64, detuning = 2.200000e+00 : f64, polarization = dense<[0, 1]> : vector<2xi64>, wavevector = dense<[-2, 3]> : vector<2xi64>>, phase = 0.00{{.*}} : f64}
    // CHECK-NEXT: }
    %6 = quantum.custom "RX"(%arg0) %5 : !quantum.bit

    // CHECK: [[msout:%.+]] = ion.parallelprotocol([[rx2out]], %3) : !quantum.bit, !quantum.bit {
    // CHECK-NEXT: ^{{.*}}(%arg1: !quantum.bit, %arg2: !quantum.bit):
    // CHECK-NEXT: ion.pulse(%arg0 : f64) %arg1 {beam = #ion.beam<transition_index = 0 : i64, rabi = 1.230000e+00 : f64, detuning = 4.560000e+00 : f64, polarization = dense<[7, 8]> : vector<2xi64>, wavevector = dense<[9, 10]> : vector<2xi64>>, phase = 0.00{{.*}} : f64}
    // CHECK-NEXT: ion.pulse(%arg0 : f64) %arg1 {beam = #ion.beam<transition_index = 1 : i64, rabi = 1.230000e+00 : f64, detuning = 5.660000e+00 : f64, polarization = dense<[7, 8]> : vector<2xi64>, wavevector = dense<[9, 10]> : vector<2xi64>>, phase = 0.00{{.*}} : f64}
    // CHECK-NEXT: ion.pulse(%arg0 : f64) %arg1 {beam = #ion.beam<transition_index = 1 : i64, rabi = 1.230000e+00 : f64, detuning = 3.4599999999999995 : f64, polarization = dense<[7, 8]> : vector<2xi64>, wavevector = dense<[9, 10]> : vector<2xi64>>, phase = 0.00{{.*}} : f64}
    // CHECK-NEXT: ion.pulse(%arg0 : f64) %arg1 {beam = #ion.beam<transition_index = 0 : i64, rabi = 1.230000e+00 : f64, detuning = 4.560000e+00 : f64, polarization = dense<[7, 8]> : vector<2xi64>, wavevector = dense<[9, 10]> : vector<2xi64>>, phase = 0.00{{.*}} : f64}
    // CHECK-NEXT: ion.pulse(%arg0 : f64) %arg1 {beam = #ion.beam<transition_index = 1 : i64, rabi = 1.230000e+00 : f64, detuning = 8.960000e+00 : f64, polarization = dense<[7, 8]> : vector<2xi64>, wavevector = dense<[9, 10]> : vector<2xi64>>, phase = 0.00{{.*}} : f64}
    // CHECK-NEXT: ion.pulse(%arg0 : f64) %arg1 {beam = #ion.beam<transition_index = 1 : i64, rabi = 1.230000e+00 : f64, detuning = 0.15999999999999925 : f64, polarization = dense<[7, 8]> : vector<2xi64>, wavevector = dense<[9, 10]> : vector<2xi64>>, phase = 0.00{{.*}} : f64}
    // CHECK-NEXT: }
    %7:2 = quantum.custom "MS"(%arg0) %6, %3 : !quantum.bit, !quantum.bit
    return %7#0: !quantum.bit
}
