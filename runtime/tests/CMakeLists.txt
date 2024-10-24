cmake_minimum_required(VERSION 3.20)

project(lightning_qir_runtime_tests)


set(CMAKE_CXX_STANDARD 20)

Include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v2.13.9
)

FetchContent_MakeAvailable(Catch2)

fetch_pybind11()

# Required for catch_discover_tests().
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)

# Modify `ctest` to only run the supported subset of tests.
include(CTest)
include(Catch)

add_executable(runner_tests_qir_runtime runner_main.cpp)

# To avoid link to libpython, we use pybind11::module interface library.
target_link_libraries(runner_tests_qir_runtime PRIVATE
    Catch2::Catch2
    pybind11::embed
    catalyst_qir_runtime
    rtd_null_qubit
    )

target_sources(runner_tests_qir_runtime PRIVATE
    Test_NullQubit.cpp
    )

catch_discover_tests(runner_tests_qir_runtime)

if(ENABLE_OPENQASM)
    add_executable(runner_tests_openqasm runner_main.cpp)

    # To avoid link to libpython, we use pybind11::module interface library.
    target_link_libraries(runner_tests_openqasm PRIVATE
        Catch2::Catch2
        pybind11::embed
        catalyst_qir_runtime
        )

    target_sources(runner_tests_openqasm PRIVATE
        Test_OpenQasmBuilder.cpp
        Test_OpenQasmDevice.cpp
        )

    catch_discover_tests(runner_tests_openqasm)
endif()
