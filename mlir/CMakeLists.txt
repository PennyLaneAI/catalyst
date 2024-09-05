cmake_minimum_required(VERSION 3.20)
project(catalyst LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)
find_package(MHLO REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MHLOConfig.cmake in: ${MHLO_DIR}")

# Required so as not to always use the cached option from the mlir build.
option(QUANTUM_ENABLE_BINDINGS_PYTHON "Enable quantum dialect python bindings" OFF)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

if(LLVM_USE_SANITIZER)
  if ("${LLVM_USE_SANITIZER}" STREQUAL "Address")
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
  endif()
endif()

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
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

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
      target_include_directories(llvm_gtest
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

add_subdirectory(test)

if(QUANTUM_ENABLE_BINDINGS_PYTHON)
  message(STATUS "Enabling Python API")
  add_subdirectory(python)
endif()
