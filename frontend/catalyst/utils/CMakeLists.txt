project(catalyst_frontend_utils)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the Python interpreter, development headers and NumPy headers
find_package(Python 3
    REQUIRED COMPONENTS Interpreter Development.Module NumPy
    OPTIONAL_COMPONENTS Development.SABIModule)
include_directories(${PYTHON_INCLUDE_DIRS})
    
# Build the custom calls library and make `catalyst_jax_cpu_lapack_kernels` 
# target visible in the current project directory.
add_subdirectory(jax_cpu_lapack_kernels)

add_library(custom_calls SHARED
    ${PROJECT_SOURCE_DIR}/libcustom_calls.cpp
)
target_link_libraries(custom_calls PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,catalyst_jax_cpu_lapack_kernels>")
set_property(TARGET custom_calls PROPERTY ENABLE_EXPORTS OFF)

# If aiming to build additional shared submodules, setting the RPATH $ORIGIN
# value may be useful to avoid explicit library paths.
set_target_properties(custom_calls PROPERTIES BUILD_RPATH "$ORIGIN/")

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

# Allow custom calls to be loaded alongside wrapper
target_link_libraries(wrapper INTERFACE custom_calls)
set_target_properties(wrapper PROPERTIES BUILD_RPATH "$ORIGIN/")