add_mlir_library(MLIRRemote
    RemoteDialect.cpp
    RemoteOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Remote

    DEPENDS
    MLIRRemoteOpsIncGen
)
