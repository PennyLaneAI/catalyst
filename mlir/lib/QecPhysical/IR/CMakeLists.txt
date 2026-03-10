add_mlir_library(MLIRQecPhysical
    QecPhysicalDialect.cpp
    QecPhysicalOps.cpp
    QecPhysicalTypes.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/QecPhysical

    DEPENDS
    MLIRQecPhysicalAttrDefsIncGen
    MLIRQecPhysicalOpsIncGen
)
