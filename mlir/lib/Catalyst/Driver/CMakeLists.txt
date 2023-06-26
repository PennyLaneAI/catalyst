add_mlir_library(CatalystCompilerDriver
    CatalystLLVMTarget.cpp
    Pipelines.cpp

    LINK_LIBS PRIVATE
    "/home/jacob/catalyst/mlir/Enzyme/build/Enzyme/libEnzyme-17.a"
)
