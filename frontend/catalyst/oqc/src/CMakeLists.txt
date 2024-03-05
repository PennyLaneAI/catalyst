cmake_minimum_required(VERSION 3.21)

project(catalyst_oqc)
option(ENABLE_OQC "Build OQC backend device" ON)

set(runtime_includes "${PROJECT_SOURCE_DIR}/../../../../runtime/include")
set(capi_utils_includes "${PROJECT_SOURCE_DIR}/../../../../runtime/lib/capi")
set(backend_includes "${PROJECT_SOURCE_DIR}/../../../../runtime/lib/backend/common")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

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

find_package(pybind11 CONFIG)

if(pybind11_FOUND)
    message(STATUS "Found pybind11")
else()
    message(STATUS "Cound not find existing pybind11-dev package. Building from source.")
    set(CMAKE_POLICY_DEFAULT_CMP0127 NEW) # To suppress pybind11 CMP0127 warning

    # https://cmake.org/cmake/help/latest/module/FindPython.html
    find_package(Python COMPONENTS Interpreter Development)

    FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.10.1
    )

    FetchContent_MakeAvailable(pybind11)
endif()

add_library(rtd_oqc SHARED OQCDevice.cpp)

target_include_directories(rtd_oqc PUBLIC .
    ${runtime_includes}
    ${backend_includes}
    )

target_link_libraries(rtd_oqc PRIVATE pybind11::module)

set_property(TARGET rtd_oqc PROPERTY POSITION_INDEPENDENT_CODE ON)

add_subdirectory(tests)
