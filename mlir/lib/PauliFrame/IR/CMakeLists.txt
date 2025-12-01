add_mlir_library(MLIRPauliFrame
    PauliFrameDialect.cpp
    PauliFrameOps.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/PauliFrame

    DEPENDS
    MLIRPauliFrameOpsIncGen
    MLIRPauliFrameEnumsIncGen
)
