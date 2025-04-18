add_mlir_library(MLIRMBQC
    MBQCDialect.cpp
    MBQCInterfaces.cpp
    MBQCOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/MBQC

    DEPENDS
    MLIRMBQCInterfacesIncGen
    MLIRMBQCOpsIncGen
    MLIRMBQCEnumsIncGen
)
