add_mlir_library(MLIRCatalystTransforms
    ArrayListToMemRefPass.cpp
    scatter_lowering.cpp
    ScatterPatterns.cpp
    RegisterAllPasses.cpp
    tensor_init_lowering.cpp
    TensorInitPatterns.cpp

    DEPENDS
    MLIRCatalystPassIncGen
)
