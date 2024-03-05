cmake_minimum_required(VERSION 3.21)

project(oqc_runtime_tests)

set(CMAKE_CXX_STANDARD 20)

Include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v2.13.9
)

FetchContent_MakeAvailable(Catch2)

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

# Required for catch_discover_tests().
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)

# Modify `ctest` to only run the supported subset of tests.
include(CTest)
include(Catch)

add_executable(runner_tests_oqc runner_main.cpp)

# To avoid link to libpython, we use pybind11::module interface library.
target_link_libraries(runner_tests_oqc PRIVATE
    Catch2::Catch2
    pybind11::embed
    rtd_oqc
    )

#     message(${PROJECT_SOURCE_DIR})
# target_include_directories(runner_tests_oqc PRIVATE
#     "${PROJECT_SOURCE_DIR}/../"
# )

target_sources(runner_tests_oqc PRIVATE
    Test_OpenQASM2Builder.cpp
    )

catch_discover_tests(runner_tests_oqc)
