add_mlir_library(MLIRQEC
    QECDialect.cpp
    QECOps.cpp
    QECOpInterfaces.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/QEC

    DEPENDS
    MLIRQECOpInterfacesIncGen
    MLIRQECOpsIncGen
    MLIRQECEnumsIncGen
)
