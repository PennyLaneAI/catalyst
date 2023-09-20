# Compression library
# Potentially needed by LLVMSupport.
# along with zstd.
# TODO: Investigate if we can depend on only one
set(EXTERNAL_LIB z)
set(ENZYME_STATIC_LIB ON)
add_subdirectory(${ENZYME_SRC_DIR}/enzyme ${CMAKE_CURRENT_BINARY_DIR}/enzyme)

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
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    MLIROptLib
    MLIRCatalyst
    MLIRCatalystTransforms
    MLIRQuantum
    quantum-transforms
    MLIRGradient
    gradient-transforms
    MhloRegisterDialects
    StablehloRegister
    ${ALL_MHLO_PASSES}
)

add_mlir_library(CatalystCompilerDriver
    CompilerDriver.cpp
    CatalystLLVMTarget.cpp

    LINK_LIBS PRIVATE
    MhloRegisterDialects
    StablehloRegister
    ${ALL_MHLO_PASSES}
    ${EXTERNAL_LIB}
    ${LIBS}

    EnzymeStatic-${LLVM_VERSION_MAJOR}
    DEPENDS
    EnzymeStatic-${LLVM_VERSION_MAJOR}
    )
