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

// RUN: quantum-opt %s --quantum-to-ion | FileCheck %s


func.func @example_ion(%arg0: f64) -> !quantum.bit {
    %0 = ion.ion {
    name="YB117", 
    mass=10.1, 
    charge=12.1, 
    position=dense<[0, 1]>: tensor<2xi64>, 
    levels=[#ion.level<principal=1, spin=1.1, orbital=2.2, nuclear=3.3, spin_orbital=4.4, spin_orbital_nuclear=5.5, spin_orbital_nuclear_magnetization=6.6, energy=8.8>, #ion.level<principal=1, spin=1.1, orbital=2.2, nuclear=3.3, spin_orbital=4.4, spin_orbital_nuclear=5.5, spin_orbital_nuclear_magnetization=6.6, energy=8.8>], 
    transitions=[#ion.transition<level_0 = #ion.level<principal=1, spin=1.1, orbital=2.2, nuclear=3.3, spin_orbital=4.4, spin_orbital_nuclear=5.5, spin_orbital_nuclear_magnetization=6.6, energy=8.8>, level_1 =#ion.level<principal=1, spin=1.1, orbital=2.2, nuclear=3.3, spin_orbital=4.4, spin_orbital_nuclear=5.5, spin_orbital_nuclear_magnetization=6.6, energy=8.8>, einstein_a=10.10>,#ion.transition<level_0 = #ion.level<principal=1, spin=1.1, orbital=2.2, nuclear=3.3, spin_orbital=4.4, spin_orbital_nuclear=5.5, spin_orbital_nuclear_magnetization=6.6, energy=8.8>, level_1 =#ion.level<principal=1, spin=1.1, orbital=2.2, nuclear=3.3, spin_orbital=4.4, spin_orbital_nuclear=5.5, spin_orbital_nuclear_magnetization=6.6, energy=8.8>, einstein_a=10.10>],
    beams=[#ion.beam<
        transition = #ion.transition<level_0 = #ion.level<principal=1, spin=1.1, orbital=2.2, nuclear=3.3, spin_orbital=4.4, spin_orbital_nuclear=5.5, spin_orbital_nuclear_magnetization=6.6, energy=8.8>, level_1 =#ion.level<principal=1, spin=1.1, orbital=2.2, nuclear=3.3, spin_orbital=4.4, spin_orbital_nuclear=5.5, spin_orbital_nuclear_magnetization=6.6, energy=8.8>, einstein_a=10.10>,
        rabi=10.10,
        detuning=11.11,
        polarization=dense<[0, 1]>: tensor<2xi64>,
        wavevector=dense<[0, 1]>: tensor<2xi64>>,#ion.beam<
        transition = #ion.transition<level_0 = #ion.level<principal=1, spin=1.1, orbital=2.2, nuclear=3.3, spin_orbital=4.4, spin_orbital_nuclear=5.5, spin_orbital_nuclear_magnetization=6.6, energy=8.8>, level_1 =#ion.level<principal=1, spin=1.1, orbital=2.2, nuclear=3.3, spin_orbital=4.4, spin_orbital_nuclear=5.5, spin_orbital_nuclear_magnetization=6.6, energy=8.8>, einstein_a=10.10>,
        rabi=10.10,
        detuning=11.11,
        polarization=dense<[0, 1]>: tensor<2xi64>,
        wavevector=dense<[0, 1]>: tensor<2xi64>> ]}: !ion.ion
    %1 = quantum.alloc( 1) : !quantum.reg
    %2 = quantum.extract %1[ 0] : !quantum.reg -> !quantum.bit
    %3 = quantum.custom "RX"(%arg0) %2 : !quantum.bit
    %4 = quantum.custom "RY"(%arg0) %3 : !quantum.bit
    %5 = quantum.custom "RX"(%arg0) %4 : !quantum.bit
    return %5: !quantum.bit
}

