add_mlir_library(MLIRQRef
    QRefDialect.cpp
    QRefInterfaces.cpp
    QRefOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/QRef

    DEPENDS
    MLIRQuantumInterfacesIncGen
    MLIRQRefOpsIncGen
)
