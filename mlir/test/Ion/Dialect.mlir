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

// RUN: quantum-opt %s

// TODO: Add Filecheck above

func.func @example_pulse(%arg0: f64) -> !quantum.bit {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
    ion.pulse(%arg0: f64) %1 {beam=#ion.beam<
        transition_index = 0,
        rabi=10.10,
        detuning=11.11,
        polarization=dense<[0, 1]>: tensor<2xi64>,
        wavevector=dense<[0, 1]>: tensor<2xi64>>, phase=0.0}
    return %1: !quantum.bit
}


func.func @example_parallel_protocol(%arg0: f64) -> !quantum.bit {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
    %2 = ion.parallelprotocol(%1): !quantum.bit {
        ^bb0(%arg1: !quantum.bit):
        ion.pulse(%arg0: f64) %arg1 {beam=#ion.beam<
        transition_index = 1,
        rabi=10.10,
        detuning=11.11,
        polarization=dense<[0, 1]>: tensor<2xi64>,
        wavevector=dense<[0, 1]>: tensor<2xi64>>, phase=0.0}
        ion.pulse(%arg0: f64) %arg1 {beam=#ion.beam<
        transition_index = 0,
        rabi=10.10,
        detuning=11.11,
        polarization=dense<[0, 1]>: tensor<2xi64>,
        wavevector=dense<[0, 1]>: tensor<2xi64>>, phase=0.0}
        ion.yield %arg1: !quantum.bit
    }
    return %2: !quantum.bit
}

func.func @example_parallel_protocol_two_qubits(%arg0: f64) -> (!quantum.bit, !quantum.bit) {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[1] : !quantum.reg -> !quantum.bit
    %3:2 = ion.parallelprotocol(%1, %2): !quantum.bit, !quantum.bit {
        ^bb0(%arg1: !quantum.bit, %arg2: !quantum.bit):
        ion.pulse(%arg0: f64) %arg1 {beam=#ion.beam<
        transition_index = 2,
        rabi=10.10,
        detuning=11.11,
        polarization=dense<[0, 1]>: tensor<2xi64>,
        wavevector=dense<[0, 1]>: tensor<2xi64>>, phase=0.0}
        ion.pulse(%arg0: f64) %arg2 {beam=#ion.beam<
        transition_index = 1,
        rabi=10.10,
        detuning=11.11,
        polarization=dense<[0, 1]>: tensor<2xi64>,
        wavevector=dense<[0, 1]>: tensor<2xi64>>, phase=0.0}
        ion.yield %arg1, %arg2: !quantum.bit, !quantum.bit
    }
    return %3#0, %3#1: !quantum.bit, !quantum.bit
}


func.func @example_ion() -> !ion.ion {
    %0 = ion.ion {
    name="YB117", 
    mass=10.1, 
    charge=12.1, 
    position=dense<[0, 1]>: tensor<2xi64>, 
    levels=[#ion.level<principal=1, spin=1.1, orbital=2.2, nuclear=3.3, spin_orbital=4.4, spin_orbital_nuclear=5.5, spin_orbital_nuclear_magnetization=6.6, energy=8.8>, #ion.level<principal=1, spin=1.1, orbital=2.2, nuclear=3.3, spin_orbital=4.4, spin_orbital_nuclear=5.5, spin_orbital_nuclear_magnetization=6.6, energy=8.8>], 
    transitions=[#ion.transition<level_0 = #ion.level<principal=1, spin=1.1, orbital=2.2, nuclear=3.3, spin_orbital=4.4, spin_orbital_nuclear=5.5, spin_orbital_nuclear_magnetization=6.6, energy=8.8>, level_1 =#ion.level<principal=1, spin=1.1, orbital=2.2, nuclear=3.3, spin_orbital=4.4, spin_orbital_nuclear=5.5, spin_orbital_nuclear_magnetization=6.6, energy=8.8>, einstein_a=10.10>,#ion.transition<level_0 = #ion.level<principal=1, spin=1.1, orbital=2.2, nuclear=3.3, spin_orbital=4.4, spin_orbital_nuclear=5.5, spin_orbital_nuclear_magnetization=6.6, energy=8.8>, level_1 =#ion.level<principal=1, spin=1.1, orbital=2.2, nuclear=3.3, spin_orbital=4.4, spin_orbital_nuclear=5.5, spin_orbital_nuclear_magnetization=6.6, energy=8.8>, einstein_a=10.10>]}: !ion.ion
    ion.system(%0) {beams1=[#ion.beam<
        rabi=10.10,
        detuning=11.11,
        polarization=dense<[0, 1]>: tensor<2xi64>,
        wavevector=dense<[0, 1]>: tensor<2xi64>>,#ion.beam<
        rabi=10.10,
        detuning=11.11,
        polarization=dense<[0, 1]>: tensor<2xi64>,
        wavevector=dense<[0, 1]>: tensor<2xi64>> ],
        beams2=[#ion.beam<
        rabi=10.10,
        detuning=11.11,
        polarization=dense<[0, 1]>: tensor<2xi64>,
        wavevector=dense<[0, 1]>: tensor<2xi64>>,#ion.beam<
        rabi=10.10,
        detuning=11.11,
        polarization=dense<[0, 1]>: tensor<2xi64>,
        wavevector=dense<[0, 1]>: tensor<2xi64>> ],
        phonons=[#ion.phonon<energy=10.10, eigen_vector=dense<[0, 1]>: tensor<2xi64>>]}
    return %0: !ion.ion
}

