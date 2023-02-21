add_mlir_library(MLIRQuantum
    QuantumDialect.cpp
    QuantumOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Quantum

    DEPENDS
    MLIRQuantumOpsIncGen
)
