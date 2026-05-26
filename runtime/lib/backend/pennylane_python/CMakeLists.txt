if(RUNTIME_CLANG_TIDY)
    if(NOT DEFINED CLANG_TIDY_BINARY)
        set(CLANG_TIDY_BINARY clang-tidy)
    endif()
    message(STATUS "Using CLANG_TIDY_BINARY=${CLANG_TIDY_BINARY}")
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_BINARY};
        -extra-arg=-std=c++20;
        --use-color;
    )
endif()

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

# Create the Python `pennylane_python_module` module
# Target the stable ABI for Python 3.12+, which reduces the number of binary wheels that must be
# built (`STABLE_ABI` does nothing on older Python versions).
nanobind_add_module(pennylane_python_module STABLE_ABI pennylane_python_module.cpp)

# Use a consistent suffix ".so" rather than, e.g. ".abi3.so" (when using the Stable ABI) or
# ".cpython-3xx-darwin.so". Doing so simplifies the process to locate it when calling
# `dlopen(PLPYTHON_PY)` in PLPythonRunner.hpp.
set_target_properties(pennylane_python_module PROPERTIES SUFFIX ".so")


message("I AM BEING BUILT!!!!!!!!!!!")
add_library(rtd_pennylane_python SHARED PLPythonDevice.cpp)

target_include_directories(rtd_pennylane_python
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
    ${runtime_includes}
    ${backend_utils_includes}
)

set_property(TARGET rtd_pennylane_python PROPERTY POSITION_INDEPENDENT_CODE ON)

# We don't directly link against the PL Python module, but we do dlopen it so make dep explicit.
add_dependencies(rtd_pennylane_python pennylane_python_module)
target_compile_definitions(rtd_pennylane_python PUBLIC -DPLPYTHON_PY=\"$<TARGET_FILE_NAME:pennylane_python_module>\")

# The pennylane_python_module.so will be in the same directory as the device library:
if(NOT APPLE)
    set_property(TARGET rtd_pennylane_python APPEND PROPERTY BUILD_RPATH $ORIGIN)
else()
    set_property(TARGET rtd_pennylane_python APPEND PROPERTY BUILD_RPATH @loader_path)
endif()
