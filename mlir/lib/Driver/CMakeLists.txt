find_package(Enzyme REQUIRED CONFIG)
message(STATUS "Using EnzymeConfig.cmake in: ${Enzyme_DIR}")

# TODO: remove this once Enzyme exports the static library target
find_library(ENZYME_LIB EnzymeStatic-${LLVM_VERSION_MAJOR} PATHS ${Enzyme_DIR}/Enzyme/)
message(STATUS "Found Enzyme lib: ${ENZYME_LIB}")

include_directories(${ENZYME_SRC_DIR}/enzyme/Enzyme)

# Experimentally found through removing items
# from llvm/tools/llc/CMakeLists.txt.
# It does make sense that we need the parser to parse MLIR
# and the codegen to codegen.
set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    ${translation_libs}
    MLIROptLib
    MLIRCatalyst
    MLIRCatalystTransforms
    MLIRQuantum
    quantum-transforms
    MLIRGradient
    gradient-transforms
    MLIRMitigation
    mitigation-transforms
    MLIRIon
    ion-transforms
    MhloRegisterDialects
    StablehloRegister
    MLIRCatalystTest
    ${ALL_MHLO_PASSES}
    ${ENZYME_LIB}
)

add_mlir_library(CatalystCompilerDriver
    CompilerDriver.cpp
    CatalystLLVMTarget.cpp
    Pipelines.cpp

    LINK_LIBS PRIVATE
    ${EXTERNAL_LIB}
    ${LIBS}
    )
