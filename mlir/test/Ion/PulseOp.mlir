func.func @example_pulse(%arg0: f64) -> !quantum.bit {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
    ion.pulse(%arg0: f64) %1 {beam=#ion.beam<
        transition = #ion.transition<level_0 = #ion.level<principal=1, spin=1.1, orbital=2.2, nuclear=3.3, spin_orbital=4.4, spin_orbital_nuclear=5.5, spin_orbital_nuclear_magnetization=6.6, energy=8.8>, level_1 =#ion.level<principal=1, spin=1.1, orbital=2.2, nuclear=3.3, spin_orbital=4.4, spin_orbital_nuclear=5.5, spin_orbital_nuclear_magnetization=6.6, energy=8.8>, einstein_a=10.10>,
        rabi=10.10,
        detuning=11.11,
        polarization=dense<[0, 1]>: tensor<2xi64>,
        wavevector=dense<[0, 1]>: tensor<2xi64>>}
    return %1: !quantum.bit
}


func.func @example_parallel_protocol(%arg0: f64) -> !quantum.bit {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
    %2 = ion.parallelprotocol(%1): !quantum.bit {
        ion.pulse(%arg0: f64) %1 {beam=#ion.beam<
        transition = #ion.transition<level_0 = #ion.level<principal=1, spin=1.1, orbital=2.2, nuclear=3.3, spin_orbital=4.4, spin_orbital_nuclear=5.5, spin_orbital_nuclear_magnetization=6.6, energy=8.8>, level_1 =#ion.level<principal=1, spin=1.1, orbital=2.2, nuclear=3.3, spin_orbital=4.4, spin_orbital_nuclear=5.5, spin_orbital_nuclear_magnetization=6.6, energy=8.8>, einstein_a=10.10>,
        rabi=10.10,
        detuning=11.11,
        polarization=dense<[0, 1]>: tensor<2xi64>,
        wavevector=dense<[0, 1]>: tensor<2xi64>>}
        ion.pulse(%arg0: f64) %1 {beam=#ion.beam<
        transition = #ion.transition<level_0 = #ion.level<principal=1, spin=1.1, orbital=2.2, nuclear=3.3, spin_orbital=4.4, spin_orbital_nuclear=5.5, spin_orbital_nuclear_magnetization=6.6, energy=8.8>, level_1 =#ion.level<principal=1, spin=1.1, orbital=2.2, nuclear=3.3, spin_orbital=4.4, spin_orbital_nuclear=5.5, spin_orbital_nuclear_magnetization=6.6, energy=8.8>, einstein_a=10.10>,
        rabi=10.10,
        detuning=11.11,
        polarization=dense<[0, 1]>: tensor<2xi64>,
        wavevector=dense<[0, 1]>: tensor<2xi64>>}
        ion.yield %1: !quantum.bit
    }
    return %2: !quantum.bit
}
