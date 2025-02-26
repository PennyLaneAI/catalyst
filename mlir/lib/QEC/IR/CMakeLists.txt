add_mlir_library(MLIRQEC
    QECDialect.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/QEC

    DEPENDS
    MLIRQECDialectIncGen
)
