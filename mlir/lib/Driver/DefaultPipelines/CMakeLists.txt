if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.20)
    project(DefaultPipelines LANGUAGES CXX C)

    set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

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
find_package(Python COMPONENTS Interpreter Development REQUIRED)
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

set(MLIR_PROJECT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../..")

target_include_directories(default_pipelines PRIVATE
    ${MLIR_PROJECT_SOURCE_DIR}/include/Driver
)
