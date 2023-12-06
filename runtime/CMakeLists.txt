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
option(BUILD_QIR_STDLIB_FROM_SRC "Build qir-stdlib from source" OFF)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(runtime_includes "${PROJECT_SOURCE_DIR}/include")
set(capi_utils_includes "${PROJECT_SOURCE_DIR}/lib/capi")
set(backend_includes "${PROJECT_SOURCE_DIR}/lib/backend/common")

set(QIRSTDLIB_VERSION "0.4.0")
set(QIRSTDLIB_REPO "https://github.com/qir-alliance/qir-runner")

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

if(RUNTIME_ENABLE_WARNINGS)
    message(STATUS "Building with compiler warnings as errors enabled.")
    add_compile_options(-Werror -Wall)
endif()

if(BUILD_QIR_STDLIB_FROM_SRC OR APPLE)
    message(STATUS "Building qir-stdlib from source.")

    include(FetchContent)

    FetchContent_Declare(
        qir_runner
        GIT_REPOSITORY ${QIRSTDLIB_REPO}.git
        GIT_TAG        v${QIRSTDLIB_VERSION}
    )

    FetchContent_MakeAvailable(qir_runner)

    execute_process(
        COMMAND cargo build --release --package qir-stdlib --target-dir ${qir_runner_BINARY_DIR} RESULT_VARIABLE cargo_found_status
        COMMAND ${CMAKE_COMMAND} -E copy ./stdlib/include/qir_stdlib.h ${qir_runner_BINARY_DIR}/release/qir_stdlib.h
        WORKING_DIRECTORY ${qir_runner_SOURCE_DIR}
    )

    if(NOT cargo_found_status EQUAL "0")
        message(FATAL_ERROR
            "Missing cargo installation\n"
            "Please install \"Rust\" and the \"llvm-tools-preview\" rustup component "
            "for building qir-stdlib from source. See the installation guideline for details: "
            "https://docs.pennylane.ai/projects/catalyst/en/stable/dev/installation.html"
            )
    endif()

    set(QIR_STDLIB_PATH ${qir_runner_BINARY_DIR}/release)
elseif(NOT DEFINED QIR_STDLIB_PATH)
    if(UNIX)
        set(QIRSTDLIB_PLATFORM "ubuntu-20.04-amd64")
    elseif(MSVC)
        set(QIRSTDLIB_PLATFORM "windows-2019-amd64")
    endif()

    set(QIRSTDLIB_URL "${QIRSTDLIB_REPO}/releases/download/v${QIRSTDLIB_VERSION}/qir-stdlib-${QIRSTDLIB_VERSION}-${QIRSTDLIB_PLATFORM}.zip")
    set(QIRSTDLIB_PATH "${CMAKE_BINARY_DIR}/qir-stdlib/lib.zip")

    file(DOWNLOAD ${QIRSTDLIB_URL} ${QIRSTDLIB_PATH} STATUS QIRSTDLIB_STATUS)
    message(STATUS "Downloading qir-stdlib status: ${QIRSTDLIB_STATUS}")

    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xf ${CMAKE_BINARY_DIR}/qir-stdlib/lib.zip
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/qir-stdlib
    )

    set(QIR_STDLIB_PATH "${CMAKE_BINARY_DIR}/qir-stdlib")
endif()

message(STATUS "QIR_STDLIB: ${QIR_STDLIB_PATH}")

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

add_library(catalyst_qir_runtime INTERFACE)

target_link_libraries(catalyst_qir_runtime INTERFACE ${devices_list} rt_capi)

target_include_directories(catalyst_qir_runtime INTERFACE
    ${runtime_includes}
    ${backend_includes}
    ${QIR_STDLIB_PATH}
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
