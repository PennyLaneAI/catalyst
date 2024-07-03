add_mlir_library(MLIROpenQASM
    OpenQASMDialect.cpp
    OpenQASMOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/OpenQASM

    DEPENDS
    MLIROpenQASMOpsIncGen
)
