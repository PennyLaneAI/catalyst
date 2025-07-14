cmake_minimum_required(VERSION 3.20)
project(Catalyst LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Required so as not to always use the cached option from the mlir build.
option(QUANTUM_ENABLE_BINDINGS_PYTHON "Enable quantum dialect python bindings" OFF)
option(CATALYST_ENABLE_WARNINGS "Enable -Wall and -Werror" ON)

message(STATUS "LLVM_USE_SANITIZER is ${LLVM_USE_SANITIZER}.")
message(STATUS "CATALYST_ENABLE_WARNINGS is ${CATALYST_ENABLE_WARNINGS}.")

if ("${LLVM_USE_SANITIZER}" STREQUAL "Address")
  add_compile_options(-fsanitize=address)
  add_link_options(-fsanitize=address)
endif()

if(CATALYST_ENABLE_WARNINGS)
    set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
    add_compile_options(-Wall)
endif()

#########################
## Handle Dependencies ##
#########################

find_package(MLIR REQUIRED CONFIG)
find_package(MHLO REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MHLOConfig.cmake in: ${MHLO_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# Taken from mlir-hlo/mhlo/transforms/CMakeLists.txt.
# Unfortunately, AllMhloPasses doesn't appear to be exported.
set(ALL_MHLO_PASSES
  ChloPasses
  MhloPasses
  StablehloPasses
  MhloToArithmeticConversion
  MhloToMemrefConversion
  HloToLinalgUtils
  MhloToLinalg
  MhloToStablehlo
  StablehloToMhlo
)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MHLO_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

# Policy CMP0175 was introduced in CMake 4.31 and raises warnings in the upstream CMake modules.
# Policy CMP0177 was introduced in CMake 4.31 and raises warnings in the upstream CMake modules.
# TODO: Remove once they (and us) have updated their code to deal with it.
if(POLICY CMP0175)
  cmake_policy(SET CMP0175 OLD)
endif()
if(POLICY CMP0177)
  cmake_policy(SET CMP0177 OLD)
endif()

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# LLVM's CMake options generate linker warnings on macOS when building modules (e.g. the Python
# bindings). We need to filter out those linker flags for now. TODO: remove once we update past
# https://github.com/llvm/llvm-project/commit/1f0e0da3af783fd2bb5e23bc2b97141abac68926
if(APPLE)
  string(REPLACE "-Wl,-undefined -Wl,suppress" "" CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS}")
endif()

if(QUANTUM_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

list(GET MHLO_INCLUDE_DIRS 1 MLIRHLO_DIR)
list(GET MHLO_INCLUDE_DIRS 2 MLIRHLO_BUILD_DIR)

set(CATALYST_MAIN_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(CATALYST_GEN_INCLUDE_DIR ${PROJECT_BINARY_DIR}/include)
set(CATALYST_LIB_DIR ${PROJECT_BINARY_DIR})

include_directories(SYSTEM
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${MHLO_INCLUDE_DIRS}
  ${MLIRHLO_DIR}/stablehlo
  ${MLIRHLO_BUILD_DIR}/stablehlo
)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})


# Include json and toml dependency
include(FetchContent)

FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH  SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
    DOWNLOAD_EXTRACT_TIMESTAMP true
    SYSTEM
)
FetchContent_MakeAvailable(json)

FetchContent_Declare(tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG v3.4.0
    SYSTEM
)
FetchContent_MakeAvailable(tomlplusplus)


######################
## Catalyst Sources ##
######################

include_directories(
  ${CATALYST_MAIN_INCLUDE_DIR}
  ${CATALYST_GEN_INCLUDE_DIR}
)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

if(QUANTUM_ENABLE_BINDINGS_PYTHON)
  message(STATUS "Enabling Python API")
  add_subdirectory(python)
endif()

add_subdirectory(cmake/modules)

####################
## Catalyst Tests ##
####################

##############################################################################
# The following subsection of code                                           #
# was taken from the CIRCT project: https://github.com/llvm/circt            #
# Small alteration were made for Catalyst                                    #
# The CIRCT project has the following license:                               #
#                                                                            #
#   The LLVM Project is under the Apache License v2.0 with LLVM Exceptions:  #
#   As an incubator project with ambition to become part of the LLVM Project,#
#   CIRCT is under the same license.                                         #
#                                                                            #
# https://github.com/llvm/circt/blob/                                        #
#  930aabe70fe78965a153f6e936c5853e948873eb/CMakeLists.txt#L85-L126          #
##############################################################################

# Handle unittests when building out-of-tree against an installed version of
# LLVM/MLIR (not a build tree). Adapted from `llvm/flang/CMakeLists.txt`.
set(CATALYST_GTEST_AVAILABLE 0)
if (TARGET llvm_gtest)
  # Installed gtest, via LLVM_INSTALL_GTEST.  Preferred.
  message(STATUS "LLVM GTest found, enabling unittests")
  set(CATALYST_GTEST_AVAILABLE 1)
else()
    find_package(Threads REQUIRED)
    set(LLVM_THIRD_PARTY_DIR llvm-project/third-party)
    set(UNITTEST_DIR ${LLVM_THIRD_PARTY_DIR}/unittest)
    if (NOT EXISTS ${UNITTEST_DIR}/googletest/include/gtest/gtest.h)
      set(UNITTEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llvm/third-party/unittest)
    endif()
    if (EXISTS ${UNITTEST_DIR}/googletest/include/gtest/gtest.h)
      add_llvm_library(llvm_gtest
        ${UNITTEST_DIR}/googletest/src/gtest-all.cc
        ${UNITTEST_DIR}/googlemock/src/gmock-all.cc
        LINK_COMPONENTS Support # llvm::raw_ostream
        BUILDTREE_ONLY
      )
      target_include_directories(llvm_gtest SYSTEM
        PUBLIC
        "${UNITTEST_DIR}/googletest/include"
        "${UNITTEST_DIR}/googlemock/include"
        PRIVATE
        "${UNITTEST_DIR}/googletest"
        "${UNITTEST_DIR}/googlemock"
      )
      target_link_libraries(llvm_gtest PUBLIC Threads::Threads)
      add_llvm_library(llvm_gtest_main
        ${UNITTEST_DIR}/UnitTestMain/TestMain.cpp
        LINK_LIBS llvm_gtest
        LINK_COMPONENTS Support # llvm::cl
        BUILDTREE_ONLY
      )
      set(CATALYST_GTEST_AVAILABLE 1)
    else()
      message(WARNING "Skipping unittests since LLVM install does not include \
        gtest headers and libraries")
      set(CATALYST_GTEST_AVAILABLE 0)
    endif()
endif()
if (CATALYST_GTEST_AVAILABLE)
  add_subdirectory(unittests)
endif()

######################
# End of CIRCT code  #
######################

add_subdirectory(test)
