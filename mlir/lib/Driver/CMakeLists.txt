#########
## fmt ##
#########

FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        12.1.0
)
set(FMT_TESTS OFF CACHE INTERNAL "Disable building fmt tests")
FetchContent_MakeAvailable(fmt)

#########################################
## CatalystCompilerDriver MLIR library ##
#########################################

find_package(Enzyme REQUIRED CONFIG)
message(STATUS "Using EnzymeConfig.cmake in: ${Enzyme_DIR}")

# TODO: remove this once Enzyme exports the static library target
find_library(ENZYME_LIB EnzymeStatic-${LLVM_VERSION_MAJOR} PATHS ${Enzyme_DIR}/Enzyme/)
if(ENZYME_LIB)
    add_library(EnzymeStatic STATIC IMPORTED)
    set_target_properties(EnzymeStatic PROPERTIES
        IMPORTED_LOCATION ${ENZYME_LIB}
        INTERFACE_LINK_LIBRARIES ""
    )
    set(ENZYME_LIB EnzymeStatic)
    message(STATUS "Found Enzyme lib: ${ENZYME_LIB}")
endif()

include_directories(SYSTEM ${ENZYME_SRC_DIR}/enzyme/Enzyme)

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
    ExternalStablehloLib
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIRRegisterAllExtensions
    MLIRCatalyst
    catalyst-transforms
    MLIRQuantum
    quantum-transforms
    MLIRQEC
    qec-transforms
    MLIRGradient
    gradient-transforms
    MLIRMBQC
    mbqc-transforms
    MLIRMitigation
    mitigation-transforms
    MLIRPauliFrame
    pauli-frame-transforms
    MLIRIon
    ion-transforms
    MLIRRTIO
    rtio-transforms
    MLIRCatalystTest
    ${ENZYME_LIB}
    fmt::fmt
)

add_mlir_library(CatalystCompilerDriver
    CompilerDriver.cpp
    CatalystLLVMTarget.cpp
    Pipelines.cpp
    HighResolutionOutputStrategy.cpp

    LINK_LIBS PRIVATE
    ${LIBS}
    MLIRSupport
)
