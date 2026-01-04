cmake_minimum_required(VERSION 3.26)

project(catalyst_runtime)
include(FetchContent)
include(ExternalProject)

set(CMAKE_CXX_STANDARD  20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler options
option(ENABLE_CODE_COVERAGE "Enable code coverage" OFF)
option(ENABLE_ADDRESS_SANITIZER "Enable address sanitizer" OFF)
option(RUNTIME_CLANG_TIDY "Enable Clang Tidy" OFF)
option(RUNTIME_ENABLE_WARNINGS "Enable -Wall and -Werror" ON)

option(ENABLE_OPENQASM "Build OpenQasm backend device" OFF)
option(ENABLE_OQD "Build OQD backend device" OFF)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(runtime_includes ${PROJECT_SOURCE_DIR}/include)
set(capi_utils_includes ${PROJECT_SOURCE_DIR}/lib/capi)
set(backend_utils_includes
    ${PROJECT_SOURCE_DIR}/lib/backend/common
    ${MLIR_INCLUDE_DIRS}                      # RunnerUtils.h
)

# Get LLVM hash to target from source tree.
file(READ ../.dep-versions DEPENDENCY_VERSIONS)
string(REGEX MATCH "llvm=([0-9a-f]+)" _ ${DEPENDENCY_VERSIONS})
set(LLVM_HASH ${CMAKE_MATCH_1})
message(STATUS "Detected LLVM version - ${LLVM_HASH}")

# Note on pybind11 vs python discovery order:
# If Python is looked for first, then we have to look for all the components needed by pybind11.
# In particular, if pybind11::embed is used, then we need to find both headers (Development.Module)
# and the shared library (Development.Embed) before pybind11 is discovered.
# With the other order PyBind will discover everything it needs.
# Note on flags:
# - PYTHON_EXECUTABLE is a pybind11 specific flag used by its own (legacy) Python discovery process,
#   it will not affect find_package(Python) calls.
# - Python_EXECUTABLE is a cmake flag used in find_package(Python) to guide the discovery.
# Note that pybind11 can be made to use find_python (instead of its legacy discovery), and thus
# respect Python_EXECUTABLE), via the PYBIND11_FINDPYTHON flag.

# Here, we look for the desired Python version early to avoid any problems with mismatched packages.
# The desired Python environment should be specified ahead of time via -DPython_EXECUTABLE=...
# The optional component is only used for the C++ test suite (to spin up its own interpreter),
# and requires libpython.so to be available on the system.
find_package(Python REQUIRED
    COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.Embed Development.SABIModule
)

message(STATUS "ENABLE_OPENQASM is ${ENABLE_OPENQASM}.")
message(STATUS "ENABLE_OQD is ${ENABLE_OQD}.")

message(STATUS "ENABLE_CODE_COVERAGE is ${ENABLE_CODE_COVERAGE}.")
message(STATUS "ENABLE_ADDRESS_SANITIZER is ${ENABLE_ADDRESS_SANITIZER}.")
message(STATUS "RUNTIME_ENABLE_WARNINGS is ${RUNTIME_ENABLE_WARNINGS}.")

if(ENABLE_ADDRESS_SANITIZER)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif()

if(RUNTIME_ENABLE_WARNINGS)
    set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
    add_compile_options(-Wall)
endif()

add_subdirectory(lib)
add_subdirectory(tests)

if(APPLE AND (${CMAKE_SYSTEM_PROCESSOR} STREQUAL arm64))
# Don't rerun external project everytime we configure the runtime build.
if(NOT EXISTS ${CMAKE_BINARY_DIR}/lib/liblapacke.3.dylib)
    if(CMAKE_OSX_SYSROOT)
        get_filename_component(OSX_SDK_PATH ${CMAKE_OSX_SYSROOT} REALPATH)
    else()
        message(STATUS "CMAKE_OSX_SYSROOT is not set, using xcrun to get the official SDK path")

        # Use xcrun to get the official SDK path
        execute_process(
            COMMAND xcrun --sdk macosx --show-sdk-path
            OUTPUT_VARIABLE XCRUN_SDK_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE XCRUN_RESULT
        )

        if(XCRUN_RESULT EQUAL 0)
            # get the realpath of the SDK path
            get_filename_component(OSX_SDK_PATH ${XCRUN_SDK_PATH} REALPATH)
        else()
            message(FATAL_ERROR "Failed to get the official SDK path since xcrun failed, try to run "
                    "`xcrun --sdk macosx --show-sdk-path` for more information")
        endif()
    endif()

    ExternalProject_Add(lapacke-accelerate
        GIT_REPOSITORY https://github.com/lepus2589/accelerate-lapacke.git
        GIT_TAG 6cf628bd8c169fe3753555f593a8a1ddf0a124a1  # v2.0.0
        PREFIX _lapacke-accelerate
        CMAKE_ARGS "--preset accelerate-lapacke32"
                   "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/_lapacke-accelerate/install"
                   "-DCMAKE_OSX_SYSROOT=${OSX_SDK_PATH}"
        INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
        COMMAND cp ${CMAKE_BINARY_DIR}/_lapacke-accelerate/install/lib/liblapacke.3.dylib ${CMAKE_BINARY_DIR}/lib
        LOG_CONFIGURE ON
        LOG_INSTALL ON
        LOG_BUILD ON
    )
    add_dependencies(rt_capi lapacke-accelerate)  # automatically build with the runtime
endif()
endif()
