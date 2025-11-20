add_mlir_library(MLIRRTIO
    RTIODialect.cpp
    RTIOOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/RTIO

    DEPENDS
    MLIRRTIOOpsIncGen
    MLIRRTIOAttributesIncGen
)
