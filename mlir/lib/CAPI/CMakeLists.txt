add_mlir_public_c_api_library(QuantumCAPI
    CompilerDriver.cpp
    Dialects.cpp

    LINK_LIBS PRIVATE
    CatalystCompilerDriver
    MLIRCatalyst
    MLIRCatalystTransforms
    MLIRQuantum
    quantum-transforms
    MLIRGradient
    gradient-transforms

    MhloRegisterDialects
    StablehloRegister
    ${ALL_MHLO_PASSES}
)
