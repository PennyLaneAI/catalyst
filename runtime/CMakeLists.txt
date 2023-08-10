cmake_minimum_required(VERSION 3.20)

project(catalyst_runtime)

set(CMAKE_CXX_STANDARD  20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler options
option(ENABLE_CODE_COVERAGE "Enable code coverage" OFF)
option(ENABLE_SANITIZER "Enable address sanitizer" OFF)
option(RUNTIME_CLANG_TIDY "Enable Clang Tidy" OFF)

option(ENABLE_LIGHTNING "Build Lightning backend device" ON)
option(ENABLE_LIGHTNING_KOKKOS "Build Lightning-Kokkos backend device" OFF)
option(ENABLE_OPENQASM "Build OpenQasm backend device" OFF)
option(BUILD_QIR_STDLIB_FROM_SRC "Build qir-stdlib from source" OFF)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(runtime_includes "${PROJECT_SOURCE_DIR}/include")
set(capi_utils_includes "${PROJECT_SOURCE_DIR}/lib/capi")
set(extensions_includes "${PROJECT_SOURCE_DIR}/extensions")
set(backend_includes "${PROJECT_SOURCE_DIR}/lib/backend/common")

set(QIRSTDLIB_VERSION "0.4.0")
set(QIRSTDLIB_REPO "https://github.com/qir-alliance/qir-runner")

if(BUILD_QIR_STDLIB_FROM_SRC)
    message(STATUS "Building qir-stdlib from source.")
    include(FetchContent)

    FetchContent_Declare(
        qir_runner
        GIT_REPOSITORY ${QIRSTDLIB_REPO}.git
        GIT_TAG        v${QIRSTDLIB_VERSION}
    )

    FetchContent_MakeAvailable(qir_runner)

    execute_process(
    COMMAND cargo build --release --package qir-stdlib --target-dir ${qir_runner_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ./stdlib/include/qir_stdlib.h ${qir_runner_BINARY_DIR}/release/qir_stdlib.h
    WORKING_DIRECTORY ${qir_runner_SOURCE_DIR}
    )

    set(QIR_STDLIB_PATH ${qir_runner_BINARY_DIR}/release)
elseif(NOT DEFINED QIR_STDLIB_PATH)
    if (APPLE)
        set(QIRSTDLIB_PLATFORM "macos-11-amd64")
    elseif(UNIX)
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

if(ENABLE_LIGHTNING)
    message(STATUS "ENABLE_LIGHTNING is ${ENABLE_LIGHTNING}.")
    list(APPEND backend_includes "${PROJECT_SOURCE_DIR}/lib/backend/lightning")
endif()

if(ENABLE_LIGHTNING_KOKKOS)
    message(STATUS "ENABLE_LIGHTNING_KOKKOS is ${ENABLE_LIGHTNING_KOKKOS}.")
    list(APPEND backend_includes "${PROJECT_SOURCE_DIR}/lib/backend/lightning-kokkos")
endif()

if(ENABLE_OPENQASM)
    message(STATUS "ENABLE_OPENQASM is ${ENABLE_OPENQASM}.")
    list(APPEND backend_includes "${PROJECT_SOURCE_DIR}/lib/backend/openqasm")
endif()

add_library(catalyst_qir_runtime INTERFACE)

target_link_libraries(catalyst_qir_runtime INTERFACE rt_backend rt_capi)

target_include_directories(catalyst_qir_runtime INTERFACE runtime_includes ${QIR_STDLIB_PATH})

if(ENABLE_CODE_COVERAGE)
	message(STATUS "ENABLE_CODE_COVERAGE is ON.")
    target_compile_options(catalyst_qir_runtime INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-fprofile-arcs;-ftest-coverage>)
    target_link_libraries(catalyst_qir_runtime INTERFACE gcov)
endif()

if(ENABLE_SANITIZER)
    message(STATUS "ENABLE_SANITIZER is ON.")
    if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        add_compile_options(-fsanitize=address)
        add_link_options(-fsanitize=address -static-libasan)
    elseif(CMAKE_C_COMPILER_ID STREQUAL "Clang")
        target_link_libraries(catalyst_qir_runtime INTERFACE -fno-omit-frame-pointer -fsanitize=address)
    endif()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
ENDIF()

add_subdirectory(extensions)
add_subdirectory(lib)

add_subdirectory(tests)
