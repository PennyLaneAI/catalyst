add_mlir_public_c_api_library(QuantumCAPI
    Dialects.cpp

    LINK_LIBS PRIVATE
    MLIRCatalyst
    catalyst-transforms
    catalyst-mhlo-transforms
    MLIRQuantum
    quantum-transforms
    MLIRQEC
    qec-transforms
    MLIRGradient
    gradient-transforms
    MLIRMBQC
    mbqc-transforms
    MLIRMitigation
    mitigation-transforms
    MLIRIon
    ion-transforms
)
