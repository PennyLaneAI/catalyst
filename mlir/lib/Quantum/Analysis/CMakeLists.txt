add_mlir_library(MLIRQuantumAnalysis
    QuantumDependenceAnalysis.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Quantum

    DEPENDS
    MLIRQuantumInterfacesIncGen
    MLIRQuantumOpsIncGen
)
