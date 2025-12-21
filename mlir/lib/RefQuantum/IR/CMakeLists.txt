add_mlir_library(MLIRRefQuantum
    RefQuantumDialect.cpp
    RefQuantumOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/RefQuantum

    DEPENDS
    MLIRRefQuantumOpsIncGen
    MLIRRefQuantumEnumsIncGen
)
