add_mlir_library(CatalystCompilerDriver
    LLVMTarget.cpp
    Pipelines.cpp

    LINK_LIBS PRIVATE
)
