add_mlir_library(MLIRMitigation
    MitigationDialect.cpp
    MitigationOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Mitigation

    DEPENDS
    MLIRMitigationOpsIncGen
    MLIRMitigationEnumsIncGen
)
