add_mlir_library(MLIR@@@NewDialect@@@
    @@@NewDialect@@@Dialect.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/@@@NewDialect@@@

    DEPENDS
    MLIR@@@NewDialect@@@DialectIncGen
    # MLIR@@@NewDialect@@@EnumsIncGen  # Uncomment if your dialect defines any enums
)
