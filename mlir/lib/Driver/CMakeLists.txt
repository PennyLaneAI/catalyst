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
    MLIRQRef
    qref-transforms
    MLIRQuantum
    quantum-transforms
    MLIRPBC
    pbc-transforms
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
    Main.cpp
    CompilerDriver.cpp
    CatalystLLVMTarget.cpp
    PassInstrumentation.cpp
    Pipelines.cpp
    LineUtils.cpp
    HighResolutionOutputStrategy.cpp

    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PRIVATE
    ${LIBS}
    MLIRSupport
)

#######################################
## default_pipelines nanobind module ##
#######################################

# nanobind suggests including these lines to configure CMake to perform an optimized release build
# by default unless another build type is specified. Without this addition, binding code may run
# slowly and produce large binaries.
# See https://nanobind.readthedocs.io/en/latest/building.html#preliminaries
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Locate nanobind
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE nanobind_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(nanobind CONFIG REQUIRED)

set(DEFAULT_PIPELINES_SRC_FILES
    DefaultPipelines.cpp
)

# Target the stable ABI for Python 3.12+,
# which reduces the number of binary wheels that must be built
# (`STABLE_ABI` does nothing on older Python versions).
nanobind_add_module(default_pipelines STABLE_ABI
    ${DEFAULT_PIPELINES_SRC_FILES}
)

# This is necessary for compatibility with setuptools build extensions
set_target_properties(default_pipelines PROPERTIES
    SUFFIX ".so"
)

target_include_directories(default_pipelines PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(default_pipelines PRIVATE
    CatalystCompilerDriver
)
