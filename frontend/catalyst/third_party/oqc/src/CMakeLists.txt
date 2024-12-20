cmake_minimum_required(VERSION 3.20)

project(catalyst_oqc)

set(runtime_includes "${PROJECT_SOURCE_DIR}/../../../../../runtime/include")
set(backend_includes "${PROJECT_SOURCE_DIR}/../../../../../runtime/lib/backend/common")
set(runtime_lib "${RUNTIME_BUILD_DIR}/lib")
set(oqc_backend_dir "${OQC_BUILD_DIR}/backend")

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

# Locate Python
# The optional component is only used for the C++ test suite (to spin up its own interpreter),
# and requires libpython.so to be available on the system.
find_package(Python REQUIRED
    COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.Embed Development.SABIModule
)

# Locate nanobind
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE nanobind_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(nanobind CONFIG REQUIRED)

# Create the Python `oqc_python_module` module
# Target the stable ABI for Python 3.12+, which reduces the number of binary wheels that must be
# built (`STABLE_ABI` does nothing on older Python versions).
nanobind_add_module(oqc_python_module STABLE_ABI oqc_python_module.cpp)

# Use a consistant suffix ".so" rather than, e.g. ".abi3.so" (when using the Stable ABI) or
# ".cpython-3xx-darwin.so". Doing so simplifies the process to locate it when calling
# `dlopen(OQC_PY)` in frontend/catalyst/third_party/oqc/src/OQCRunner.hpp.
set_target_properties(oqc_python_module PROPERTIES SUFFIX ".so")

message(STATUS "Building the OQC device.")

add_library(rtd_oqc SHARED OQCDevice.cpp)

target_include_directories(rtd_oqc PUBLIC .
    ${runtime_includes}
    ${backend_includes}
)

set(OQC_LIBRARIES
    rtd_oqc
)

set_target_properties(rtd_oqc PROPERTIES BUILD_RPATH "$ORIGIN/../utils")
target_link_directories(rtd_oqc PRIVATE ${runtime_lib})

target_include_directories(oqc_python_module PRIVATE ${runtime_includes})

add_dependencies(rtd_oqc oqc_python_module)
target_compile_definitions(rtd_oqc PUBLIC -DOQC_PY=\"$<TARGET_FILE_NAME:oqc_python_module>\")

set_property(TARGET rtd_oqc PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET rtd_oqc APPEND PROPERTY BUILD_RPATH "$<TARGET_FILE_DIR:oqc_python_module>")
if(NOT APPLE)
    set_property(TARGET rtd_oqc APPEND PROPERTY BUILD_RPATH $ORIGIN)
else()
    set_property(TARGET rtd_oqc APPEND PROPERTY BUILD_RPATH @loader_path)
endif()

file(COPY ${PROJECT_SOURCE_DIR}/oqc.toml DESTINATION ./backend)

add_subdirectory(tests)
