add_mlir_library(CatalystCompilerDriver
    CatalystLLVMTarget.cpp
    Pipelines.cpp

    LINK_LIBS PRIVATE
)
