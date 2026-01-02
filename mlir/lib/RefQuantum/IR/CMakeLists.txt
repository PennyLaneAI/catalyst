add_mlir_library(MLIRRefQuantum
    RefQuantumDialect.cpp
    RefQuantumInterfaces.cpp
    RefQuantumOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/RefQuantum

    DEPENDS
    MLIRQuantumInterfacesIncGen
    MLIRRefQuantumOpsIncGen
)
