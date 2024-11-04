func.func @example(%arg0: f64) -> !quantum.bit {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
    %2 = ion.pulse(%arg0: f64) %1 {beam=#ion.beam<
        transition = #ion.transition<level_0 = #ion.level<principal=1, spin=1.1, orbital=2.2, nuclear=3.3, spin_orbital=4.4, spin_orbital_nuclear=5.5, spin_orbital_nuclear_magnetization=6.6, energy=8.8>, level_1 =#ion.level<principal=1, spin=1.1, orbital=2.2, nuclear=3.3, spin_orbital=4.4, spin_orbital_nuclear=5.5, spin_orbital_nuclear_magnetization=6.6, energy=8.8>, einstein_a=10.10>,
        rabi=10.10,
        detuning=11.11,
        polarization=dense<[0, 1]>: tensor<2xi64>,
        wavevector=dense<[0, 1]>: tensor<2xi64>>} : !quantum.bit
    return %2: !quantum.bit
}



// def BeamAttr : Ion_Attr<"Beam", "beam"> {
//   let summary = "A class to represent a laser beam.";

//   let parameters = (ins
//     "TransitionAttr":$transition,
//     "mlir::FloatAttr":$rabi,
//     "mlir::FloatAttr":$detuning,
//     "mlir::DenseIntElementsAttr": $polarization,
//     "mlir::DenseIntElementsAttr": $wavevector
//   );
