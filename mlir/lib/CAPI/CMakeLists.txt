add_mlir_public_c_api_library(QuantumCAPI
    Dialects.cpp

    LINK_LIBS PRIVATE
    MLIRCatalyst
    MLIRCatalystTransforms
    MLIRQuantum
    quantum-transforms
    MLIRGradient
    MLIRCatalyst
    gradient-transforms
    MLIRMitigation
    MLIROpenQASM
    openqasm-transforms
)
