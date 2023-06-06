add_mlir_library(MLIRCatalyst
    CatalystDialect.cpp
    CatalystOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Catalyst

    DEPENDS
    MLIRCatalystOpsIncGen

    LINK_LIBS PRIVATE
)
