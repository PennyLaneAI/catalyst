add_mlir_library(MLIRPBC
    PBCDialect.cpp
    PBCOps.cpp
    PBCOpInterfaces.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/PBC

    DEPENDS
    MLIRPBCOpInterfacesIncGen
    MLIRPBCOpsIncGen
    MLIRPBCEnumsIncGen
)
