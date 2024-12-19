cmake_minimum_required(VERSION 3.26)

project(catalyst_frontend)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# nanobind suggests including these lines to configure CMake to perform an optimized release build
# by default unless another build type is specified. Without this addition, binding code may run
# slowly and produce large binaries.
# See https://nanobind.readthedocs.io/en/latest/building.html#preliminaries
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Locate Python & nanobind
find_package(Python REQUIRED
    COMPONENTS Interpreter Development.Module NumPy
    OPTIONAL_COMPONENTS Development.SABIModule
)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE nanobind_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(nanobind CONFIG REQUIRED)

# Source file list for `wrapper` module
set(WRAPPER_SRC_FILES
    wrapper.cpp
)

# Create the Python `wrapper` module
# Target the stable ABI for Python 3.12+, which reduces the number of binary wheels that must be
# built (`STABLE_ABI` does nothing on older Python versions).
nanobind_add_module(wrapper STABLE_ABI ${WRAPPER_SRC_FILES})

# Add the NumPy include directory to the library's include paths
target_include_directories(wrapper PRIVATE ${Python_NumPy_INCLUDE_DIRS})

# Use suffix ".so" rather than ".abi3.so" for library file using Stable ABI
# This is necessary for compatibility with setuptools build extensions
set_target_properties(wrapper PROPERTIES SUFFIX ".so")
