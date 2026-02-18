add_mlir_public_c_api_library(QuantumCAPI
    Dialects.cpp

    LINK_LIBS PRIVATE
    MLIRCatalyst
    catalyst-transforms
    catalyst-stablehlo-transforms
    MLIRQuantum
    quantum-transforms
    MLIRPBC
    pbc-transforms
    MLIRGradient
    gradient-transforms
    MLIRMBQC
    mbqc-transforms
    MLIRMitigation
    mitigation-transforms
    MLIRIon
    ion-transforms
)
