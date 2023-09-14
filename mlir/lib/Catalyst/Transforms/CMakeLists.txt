add_mlir_library(MLIRCatalystTransforms
    ArrayListToMemRefPass.cpp
    catalyst_to_llvm.cpp

    DEPENDS
    MLIRCatalystPassIncGen
)
