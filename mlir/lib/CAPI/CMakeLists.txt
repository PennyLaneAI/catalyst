add_mlir_public_c_api_library(QuantumCAPI
    Dialects.cpp

    LINK_LIBS PRIVATE
    MLIRQuantum
    MLIRGradient
)
