cmake_minimum_required(VERSION 3.26)
project(catalyst_mlir LANGUAGES CXX C)
include(ExternalProject)

#####################################################################
# 0. Pre-build config
#####################################################################

# Define build-specific configuration options
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_BUILD_TYPE Release)

# MacOS specific option
set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

# Ensure compile commands are tracked
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find required external binaries
find_program(CCACHE_PATH ccache /usr/local/bin
                                /usr/bin
)

# Enable address sanitizer during build
option(ENABLE_ASAN OFF)
if(${ENABLE_ASAN})
  set(USE_SANITIZER_NAMES "Address")
  set(USE_SANITIZER_FLAGS "-fsanitize=address")
endif

# Get system type
if(UNIX AND NOT APPLE)
  set(LINUX TRUE)
endif()

# Define linker based on system type
if(APPLE)
  set(ENABLE_LLD OFF)
else
  set(ENABLE_LLD ON)
endif

# Define build-specific options
set(SYMBOL_VISIBILITY "default")
option(ENABLE_ZLIB ON)
option(ENABLE_ZSTD OFF)

# Output installation path for packages
set(CATALYST_PKG_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/Install" CACHE PATH "Path to installation directory for projects")

#####################################################################
# 1. Build MLIR project
#####################################################################

set(CATALYST_MLIR_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/BuildMLIR" CACHE PATH "Path to build directory for MLIR project")

# Set initial value and remap for LLVM expected format
set(CPU_TARGET ${CMAKE_SYSTEM_PROCESSOR})
if(${CPU_TARGET} STREQUAL "x86_64")
  set(CPU_TARGET "X86")
elseif((${CPU_TARGET} STREQUAL "aarch64") OR (${CPU_TARGET} STREQUAL "arm64"))
  set(CPU_TARGET "AArch64")
else()
  message(FATAL_ERROR "Catalyst is not currently supported on ${CPU_TARGET} architectures. Please file a bug report.")
endif()

find_package(Python 3 REQUIRED COMPONENTS Interpreter NumPy)

# Configure the LLVM MLIR project
execute_process(
    COMMAND ${CMAKE_COMMAND} -B${CATALYST_MLIR_BUILD_DIR} 
                             -S ${PROJECT_SOURCE_DIR}/llvm-project/llvm
                            -DLLVM_TARGETS_TO_BUILD=${CPU_TARGET}
                            -DLLVM_ENABLE_PROJECTS=mlir
                            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                            -DLLVM_BUILD_EXAMPLES=OFF
                            -DLLVM_ENABLE_ASSERTIONS=ON
                            -DMLIR_ENABLE_BINDINGS_PYTHON=ON
                            -DPython3_EXECUTABLE=${Python_EXECUTABLE}
                            -DPython3_NumPy_INCLUDE_DIRS=${Python_NumPy_INCLUDE_DIRS}
                            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                            -DCMAKE_CXX_COMPILER_LAUNCHER=${CCACHE_PATH}
                            -DLLVM_USE_SANITIZER=${USE_SANITIZER_NAMES}
                            -DLLVM_ENABLE_LLD=${ENABLE_LLD}
                            -DLLVM_ENABLE_ZLIB={ENABLE_ZLIB}
                            -DLLVM_ENABLE_ZSTD=${ENABLE_ZSTD}
                            -DCMAKE_CXX_VISIBILITY_PRESET=${SYMBOL_VISIBILITY}
                            -G Ninja
                            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
# Build the MLIR binaries and libraries
execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${CATALYST_MLIR_BUILD_DIR} --verbose
)
# Install the built tooling into the given directory
execute_process(
    COMMAND ${CMAKE_COMMAND} --install ${CATALYST_MLIR_BUILD_DIR} --prefix ${CATALYST_PKG_INSTALL_DIR}
)
# When updating LLVM, test to see if mlir/unittests/Bytecode/BytecodeTest.cpp:55 
# is passing and remove filter
execute_process(
    COMMAND ${CMAKE_COMMAND} -E env LIT_FILTER_OUT="Bytecode"
    --build ${CATALYST_MLIR_BUILD_DIR} --target check-mlir llvm-symbolizer
)

#####################################################################
# 2. Allow MLIR project to be used for subsequent builds
#####################################################################

list(APPEND CMAKE_MODULE_PATH 
    "${CATALYST_MLIR_INSTALL_DIR}/lib/cmake"
    "${CATALYST_MLIR_INSTALL_DIR}/lib/cmake/mlir"
    "${CATALYST_MLIR_INSTALL_DIR}/lib/cmake/llvm")
list(APPEND CMAKE_PREFIX_PATH 
    "${CATALYST_MLIR_INSTALL_DIR}/lib/cmake"
    "${CATALYST_MLIR_INSTALL_DIR}/lib/cmake/mlir"
    "${CATALYST_MLIR_INSTALL_DIR}/lib/cmake/llvm")


find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

#####################################################################
# 3. Build MHLO project
#####################################################################

set(CATALYST_MHLO_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/BuildMHLO" CACHE PATH "Path to build directory for MLIR-HLO project")
set(MHLO_SRC_DIR "${PROJECT_SOURCE_DIR}/mlir-hlo")
set(MHLO_PATCH_TARGET "${MHLO_DIR}/mhlo/transforms/CMakeLists.txt")
set(MHLO_PATCH_SRC "${PROJECT_SOURCE_DIR}/patches/mhlo-Add-PassesIncGen-in-transforms-CMakeList.patch")

# Conditionally apply patch if it is applicable
execute_process(COMMAND bash -c "if [ patch --dry-run -p1 -N ${MHLO_PATCH_TARGET} ${MHLO_PATCH_SRC} > /dev/null 2>&1 ]; then patch -p1 ${MHLO_PATCH_TARGET} ${MHLO_PATCH_SRC}; fi"
)

# Configure the MLIR-HLO project
execute_process(
    COMMAND ${CMAKE_COMMAND} -G Ninja
                            -B${CATALYST_MHLO_BUILD_DIR} 
                            -S ${MHLO_SRC_DIR}
                            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                            -DLLVM_ENABLE_ASSERTIONS=ON
                            -DMLIR_DIR=${CATALYST_MLIR_INSTALL_DIR}/lib/cmake/mlir
                            -DCMAKE_C_COMPILER_LAUNCHER=${CCACHE_PATH}
                            -DCMAKE_CXX_COMPILER_LAUNCHER=${CCACHE_PATH}
                            -DPython3_EXECUTABLE=${Python_EXECUTABLE}
                            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                            -DCMAKE_EXE_LINKER_FLAGS=${USE_SANITIZER_FLAGS}
                            -DLLVM_ENABLE_LLD=${ENABLE_LLD}
                            -DLLVM_ENABLE_ZLIB=${ENABLE_ZLIB}
                            -DLLVM_ENABLE_ZSTD=${ENABLE_ZSTD}
                            -DCMAKE_CXX_VISIBILITY_PRESET=${SYMBOL_VISIBILITY}

)

# Build the MLIR binaries and libraries
execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${CATALYST_MHLO_BUILD_DIR} --verbose
)
# Install the built tooling into the given directory
execute_process(
    COMMAND ${CMAKE_COMMAND} --install ${CATALYST_MHLO_BUILD_DIR} --prefix ${CATALYST_PKG_INSTALL_DIR}
)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E env LIT_FILTER_OUT="chlo_legalize_to_mhlo"
    --build ${CATALYST_MHLO_BUILD_DIR} --target check-mlir-hlo
)

#####################################################################
# 4. Build Enzyme project
#####################################################################

set(ENZYME_SRC_DIR "${CMAKE_CURRENT_BINARY_DIR}/Enzyme/enzyme")

# Configure the MLIR-HLO project
execute_process(
    COMMAND ${CMAKE_COMMAND} -G Ninja
                            -B${CATALYST_MHLO_BUILD_DIR} 
                            -S ${MHLO_SRC_DIR}
                            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                            -DLLVM_ENABLE_ASSERTIONS=ON
                            -DMLIR_DIR=${CATALYST_MLIR_INSTALL_DIR}/lib/cmake/mlir
                            -DCMAKE_C_COMPILER_LAUNCHER=${CCACHE_PATH}
                            -DCMAKE_CXX_COMPILER_LAUNCHER=${CCACHE_PATH}
                            -DPython3_EXECUTABLE=${Python_EXECUTABLE}
                            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                            -DCMAKE_EXE_LINKER_FLAGS=${USE_SANITIZER_FLAGS}
                            -DLLVM_ENABLE_LLD=${ENABLE_LLD}
                            -DLLVM_ENABLE_ZLIB=${ENABLE_ZLIB}
                            -DLLVM_ENABLE_ZSTD=${ENABLE_ZSTD}
                            -DCMAKE_CXX_VISIBILITY_PRESET=${SYMBOL_VISIBILITY}

)

# Build the MLIR binaries and libraries
execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${CATALYST_MHLO_BUILD_DIR} --verbose
)
# Install the built tooling into the given directory
execute_process(
    COMMAND ${CMAKE_COMMAND} --install ${CATALYST_MHLO_BUILD_DIR} --prefix ${CATALYST_PKG_INSTALL_DIR}
)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E env LIT_FILTER_OUT="chlo_legalize_to_mhlo"
    --build ${CATALYST_MHLO_BUILD_DIR} --target check-mlir-hlo
)


#####################################################################
# X. Add required modules for building our extensions and passes
#####################################################################

include(AddLLVM REQUIRED)
include(TableGen REQUIRED)
include(AddMLIR REQUIRED)
include(HandleLLVMOptions REQUIRED)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)



if (NOT DEFINED MLIR_DIR)
  message(FATAL_ERROR "MLIR_DIR is not configured but it is required. "
    "Set the cmake option MLIR_DIR, e.g.,\n"
    "    cmake -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir ..\n"
    )
endif()

#llvm_map_components_to_libnames(llvm_libs support core mlir)
add_subdirectory(mlir-hlo)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MHLOConfig.cmake in: ${MHLO_DIR}")

# Required so as not to always use the cached option from the mlir build.
option(QUANTUM_ENABLE_BINDINGS_PYTHON "Enable quantum dialect python bindings" OFF)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_CURRENT_BINARY_DIR}/InstallMLIR/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_CURRENT_BINARY_DIR}/InstallMLIR/lib)
set(MLIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

if(LLVM_USE_SANITIZER)
  if ("${LLVM_USE_SANITIZER}" STREQUAL "Address")
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
  endif()
endif()


#######################################################################

# Configure and build MLIR HLO project
execute_process(
    COMMAND ${CMAKE_COMMAND} -B${CMAKE_CURRENT_BINARY_DIR}/BuildMHLO 
                             ${CMAKE_SOURCE_DIR}/llvm-project/llvm
                            -DLLVM_TARGETS_TO_BUILD=X86
                            -DLLVM_ENABLE_PROJECTS=mlir
                            -DCMAKE_BUILD_TYPE=Release
                            -DCMAKE_CXX_COMPILER_LAUNCHER=${CCACHE_PATH}
                            -G Ninja
                            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR}/BuildMLIR --verbose
)
execute_process(
    COMMAND ${CMAKE_COMMAND} --install ${CMAKE_CURRENT_BINARY_DIR}/BuildMLIR --prefix ${CMAKE_CURRENT_BINARY_DIR}/InstallMLIR
)


#######################################################################

set(MHLO_EXTERNAL_PROJECT_BUILD 0)

find_package(MHLO REQUIRED CONFIG)

# Taken from mlir-hlo/mhlo/transforms/CMakeLists.txt.
# Unfortunately, AllMhloPasses doesn't appear to be exported.
set(ALL_MHLO_PASSES
  ChloPasses
  MhloPasses
  StablehloPasses
  MhloToArithmeticConversion
  MhloToMemrefConversion
  MhloToStandard
  HloToLinalgUtils
  MhloToLinalg
  MhloToStablehlo
  MhloQuantToIntConversion
  StablehloToMhlo
)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MHLO_CMAKE_DIR}")

if(QUANTUM_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

list(GET MHLO_INCLUDE_DIRS 1 MLIRHLO_DIR)
list(GET MHLO_INCLUDE_DIRS 2 MLIRHLO_BUILD_DIR)

include_directories(${LLVM_INCLUDE_DIRS}
                    ${MLIR_INCLUDE_DIRS}
                    ${MHLO_INCLUDE_DIRS}
                    ${MLIRHLO_DIR}/stablehlo
                    ${MLIRHLO_BUILD_DIR}/stablehlo
                    ${PROJECT_SOURCE_DIR}/include
                    ${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

if(APPLE)
  set(CMAKE_CXX_VISIBILITY_PRESET hidden)
else()
  set(CMAKE_CXX_VISIBILITY_PRESET protected)
endif()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)

if(QUANTUM_ENABLE_BINDINGS_PYTHON)
  message(STATUS "Enabling Python API")
  add_subdirectory(python)
endif()
