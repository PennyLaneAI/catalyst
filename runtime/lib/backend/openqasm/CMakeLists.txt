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

# Create the Python `openqasm_python_module` module
# Target the stable ABI for Python 3.12+, which reduces the number of binary wheels that must be
# built (`STABLE_ABI` does nothing on older Python versions).
nanobind_add_module(openqasm_python_module STABLE_ABI openqasm_python_module.cpp)

# Use a consistant suffix ".so" rather than, e.g. ".abi3.so" (when using the Stable ABI) or
# ".cpython-3xx-darwin.so". Doing so simplifies the process to locate it when calling
# `dlopen(OPENQASM_PY)` in runtime/lib/backend/openqasm/OpenQasmRunner.hpp.
set_target_properties(openqasm_python_module PROPERTIES SUFFIX ".so")

add_library(rtd_openqasm SHARED OpenQasmDevice.cpp)

target_include_directories(rtd_openqasm PRIVATE .
    ${runtime_includes}
    ${backend_includes}
)


add_dependencies(rtd_openqasm openqasm_python_module)
target_compile_definitions(rtd_openqasm PUBLIC -DOPENQASM_PY=\"$<TARGET_FILE_NAME:openqasm_python_module>\")
set_property(TARGET rtd_openqasm PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET rtd_openqasm APPEND PROPERTY BUILD_RPATH "$<TARGET_FILE_DIR:openqasm_python_module>")
if(NOT APPLE)
    set_property(TARGET rtd_openqasm APPEND PROPERTY BUILD_RPATH $ORIGIN)
else()
    set_property(TARGET rtd_openqasm APPEND PROPERTY BUILD_RPATH @loader_path)
endif()
