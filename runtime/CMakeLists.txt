cmake_minimum_required(VERSION 3.20)

project(catalyst_runtime)

set(CMAKE_CXX_STANDARD  20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler options
option(ENABLE_CODE_COVERAGE "Enable code coverage" OFF)
option(ENABLE_LIGHTNING_KOKKOS "Build Lightning-Kokkos backend device" OFF)
option(ENABLE_OPENQASM "Build OpenQasm backend device" OFF)
option(RUNTIME_CLANG_TIDY "Enable Clang Tidy" OFF)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(runtime_includes "${PROJECT_SOURCE_DIR}/include")
set(capi_utils_includes "${PROJECT_SOURCE_DIR}/lib/capi")
set(backend_includes "${PROJECT_SOURCE_DIR}/lib/backend")
set(extensions_includes "${PROJECT_SOURCE_DIR}/extensions")

add_library(catalyst_qir_runtime INTERFACE)

message(--\ ENABLE_OPENQASM="${ENABLE_OPENQASM}")
message(--\ QIR_STDLIB_LIB="${QIR_STDLIB_LIB}")
message(--\ QIR_STDLIB_INCLUDES="${QIR_STDLIB_INCLUDES}")

target_link_libraries(catalyst_qir_runtime INTERFACE rt_backend rt_capi)

target_include_directories(catalyst_qir_runtime INTERFACE runtime_includes ${QIR_STDLIB_INCLUDES})

if(ENABLE_CODE_COVERAGE)
	message(STATUS "ENABLE_CODE_COVERAGE is ON.")
    target_compile_options(catalyst_qir_runtime INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-fprofile-arcs;-ftest-coverage>)
    target_link_libraries(catalyst_qir_runtime INTERFACE gcov)
endif()

add_subdirectory(extensions)
add_subdirectory(lib)

add_subdirectory(tests)
