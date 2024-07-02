add_mlir_library(MLIRGradient
    GradientDialect.cpp
    GradientInterfaces.cpp
    GradientOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Gradient

    DEPENDS
    MLIRGradientOpsIncGen
    MLIRGradientInterfacesIncGen

    LINK_LIBS PRIVATE
    GradientUtils
)
