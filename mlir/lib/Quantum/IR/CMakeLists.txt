add_mlir_library(MLIRQuantum
    QuantumDialect.cpp
    QuantumInterfaces.cpp
    QuantumOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Quantum

    DEPENDS
    MLIRResourceInterfacesIncGen
    MLIRQuantumAttrDefsIncGen
    MLIRQuantumInterfacesIncGen
    MLIRQuantumOpsIncGen
    MLIRQRefOpsIncGen

    LINK_LIBS PRIVATE
    MLIRMBQC
    MLIRPBC
    MLIRQRef
)
