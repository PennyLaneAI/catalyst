add_mlir_library(MLIRExecutor
    ExecutorDialect.cpp
    ExecutorOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Executor

    DEPENDS
    MLIRExecutorOpsIncGen
)
