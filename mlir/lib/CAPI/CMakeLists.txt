add_mlir_public_c_api_library(QuantumCAPI
    CompilerDriver.cpp
    Dialects.cpp

    LINK_LIBS PRIVATE
    CatalystCompilerDriver
    MLIRCatalyst
    MLIRCatalystTransforms
    MLIRQuantum
    quantum-transforms
    MLIRGradient
    gradient-transforms

    MhloRegisterDialects
    StablehloRegister

    # Taken from mlir-hlo/bindings/c/CMakeLists.txt:
    # For AllMhLoPasses:
    ChloPasses
    MhloPasses
    MhloToLhloConversion
    MhloToArithmeticConversion
    MhloToMemrefConversion
    MhloToStandard
    MhloToThloConversion
    MhloToLinalg
    MhloToStablehlo
    MhloShapeOpsToStandard
    StablehloToMhlo
)
