add_mlir_library(MLIRTransport
    TransportDialect.cpp
    TransportOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Transport

    DEPENDS
    MLIRTransportOpsIncGen

    LINK_LIBS PRIVATE
    MLIRLLVMDialect
)
