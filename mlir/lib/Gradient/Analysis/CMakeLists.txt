add_mlir_library(MLIRGradientAnalysis
    ActivityAnalysis.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Gradient
    ${PROJECT_SOURCE_DIR}/include/Quantum

    DEPENDS
    MLIRGradientOpsIncGen
    MLIRQuantumOpsIncGen

    LINK_LIBS PRIVATE
    GradientUtils
)
