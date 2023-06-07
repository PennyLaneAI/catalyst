add_mlir_library(MLIRCatalystTransforms
    ArrayListToMemRefPass.cpp

    DEPENDS
    MLIRCatalystPassIncGen
)
