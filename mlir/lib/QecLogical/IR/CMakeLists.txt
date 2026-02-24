add_mlir_library(MLIRQecLogical
    QecLogicalDialect.cpp
    QecLogicalOps.cpp
    QecLogicalTypes.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/QecLogical

    DEPENDS
    MLIRQecLogicalOpsIncGen
)
