add_mlir_library(MLIRCatalystTransforms
    ArrayListToMemRefPass.cpp
    RegisterAllPasses.cpp

    DEPENDS
    MLIRCatalystPassIncGen
)
