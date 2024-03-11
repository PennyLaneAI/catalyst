cmake_minimum_required(VERSION 3.20)

project(catalyst_runtime)
include(FetchContent)

set(CMAKE_CXX_STANDARD  20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler options
option(ENABLE_CODE_COVERAGE "Enable code coverage" OFF)
option(ENABLE_ADDRESS_SANITIZER "Enable address sanitizer" OFF)
option(RUNTIME_CLANG_TIDY "Enable Clang Tidy" OFF)

option(ENABLE_LIGHTNING "Build Lightning backend device" ON)
option(ENABLE_LIGHTNING_KOKKOS "Build Lightning-Kokkos backend device" OFF)
option(ENABLE_OPENQASM "Build OpenQasm backend device" OFF)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(runtime_includes "${PROJECT_SOURCE_DIR}/include")
set(capi_utils_includes "${PROJECT_SOURCE_DIR}/lib/capi")
set(backend_includes "${PROJECT_SOURCE_DIR}/lib/backend/common")


# Get LLVM hash to target from source tree.
file(READ ../.dep-versions DEPENDENCY_VERSIONS)
string(REGEX MATCH "llvm=([0-9a-f]+)" _ ${DEPENDENCY_VERSIONS})
set(LLVM_HASH ${CMAKE_MATCH_1})
message(STATUS "Detected LLVM version - ${LLVM_HASH}")

FetchContent_Declare(
    MLIRRunnerUtils
    URL                 https://raw.githubusercontent.com/llvm/llvm-project/${LLVM_HASH}/mlir/include/mlir/ExecutionEngine/RunnerUtils.h
    DOWNLOAD_NO_EXTRACT True
    SOURCE_DIR        mlir/ExecutionEngine
)

FetchContent_Declare(
    MLIRCRunnerUtils
    URL                 https://raw.githubusercontent.com/llvm/llvm-project/${LLVM_HASH}/mlir/include/mlir/ExecutionEngine/CRunnerUtils.h
    DOWNLOAD_NO_EXTRACT True
    SOURCE_DIR          mlir/ExecutionEngine
)

function(fetch_pybind11)
    find_package(pybind11 CONFIG)
    if (pybind11_FOUND)
        message(STATUS, "FOUND pybind11")
    else()
        message(STATUS "Could not find existing pybind11-dev package. Building from source.")
        set(CMAKE_POLICY_DEFAULT_CMP0127 NEW) # To suppress pybind11 CMP0127 warning

        # https://cmake.org/cmake/help/latest/module/FindPython.html
        find_package(Python COMPONENTS Interpreter Development)

        FetchContent_Declare(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.10.1
        )

        FetchContent_MakeAvailable(pybind11)
    endif()
endfunction()

if(RUNTIME_ENABLE_WARNINGS)
    message(STATUS "Building with compiler warnings as errors enabled.")
    add_compile_options(-Werror -Wall)
endif()

message(STATUS "ENABLE_LIGHTNING is ${ENABLE_LIGHTNING}.")
message(STATUS "ENABLE_LIGHTNING_KOKKOS is ${ENABLE_LIGHTNING_KOKKOS}.")
message(STATUS "ENABLE_OPENQASM is ${ENABLE_OPENQASM}.")

set(devices_list)

if(ENABLE_LIGHTNING OR ENABLE_LIGHTNING_KOKKOS)
    list(APPEND devices_list pennylane_lightning rtd_lightning)
    if(ENABLE_LIGHTNING)
        list(APPEND backend_includes "${PROJECT_SOURCE_DIR}/lib/backend/lightning/lightning_dynamic")
    endif()
    if(ENABLE_LIGHTNING_KOKKOS)
        list(APPEND backend_includes "${PROJECT_SOURCE_DIR}/lib/backend/lightning/lightning_kokkos")
    endif()
endif()

if(ENABLE_OPENQASM)
    list(APPEND backend_includes "${PROJECT_SOURCE_DIR}/lib/backend/openqasm")
    list(APPEND devices_list rtd_openqasm)
endif()

# On macOS libomp is typically installed via brew, which doesn't make the package discoverable by
# default to avoid conflicting with GCC's OpenMP library.
if(APPLE)
    execute_process (
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE BREW_OMP
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(OpenMP_ROOT ${BREW_OMP})
endif()

add_library(catalyst_qir_runtime INTERFACE)

target_link_libraries(catalyst_qir_runtime INTERFACE ${devices_list} rt_capi)

target_include_directories(catalyst_qir_runtime INTERFACE
    ${runtime_includes}
    ${backend_includes}
    )

if(ENABLE_CODE_COVERAGE)
	message(STATUS "ENABLE_CODE_COVERAGE is ON.")
    target_compile_options(catalyst_qir_runtime INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-fprofile-arcs;-ftest-coverage>)
    target_link_libraries(catalyst_qir_runtime INTERFACE gcov)
endif()


if(ENABLE_ADDRESS_SANITIZER)
    message(STATUS "ENABLE_ADDRESS_SANITIZER is ON.")
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif()

add_subdirectory(lib)
add_subdirectory(tests)
