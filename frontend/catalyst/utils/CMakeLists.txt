set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3
    REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)

# nanobind suggests including these lines to configure CMake to perform an optimized release build
# by default unless another build type is specified. Without this addition, binding code may run
# slowly and produce large binaries.
# See https://nanobind.readthedocs.io/en/latest/building.html#preliminaries
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)

find_package(nanobind CONFIG REQUIRED)

# Get the NumPy include directory
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Source file list for `wrapper` module
set(WRAPPER_SRC_FILES
    wrapper.cpp
)

# Create the Python `wrapper` module
# Target the stable ABI for Python 3.12+, which reduces the number of binary wheels that must be
# built (`STABLE_ABI` does nothing on older Python versions).
nanobind_add_module(wrapper STABLE_ABI ${WRAPPER_SRC_FILES})

# Add the NumPy include directory to the library's include paths
target_include_directories(wrapper PRIVATE ${NUMPY_INCLUDE_DIR})

# Install `wrapper` library to catalyst/utils/ so that we can import it
install(TARGETS wrapper
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/catalyst/utils
)
