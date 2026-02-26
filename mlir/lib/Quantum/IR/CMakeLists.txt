add_mlir_library(MLIRQuantum
    QuantumDialect.cpp
    QuantumInterfaces.cpp
    QuantumOps.cpp
    QuantumTypes.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Quantum

    DEPENDS
    MLIRQuantumAttrDefsIncGen
    MLIRQuantumInterfacesIncGen
    MLIRQuantumOpsIncGen
    MLIRQRefOpsIncGen
)
