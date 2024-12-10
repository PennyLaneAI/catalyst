cmake_minimum_required(VERSION 3.20)

project(oqd_runtime_tests)

set(CMAKE_CXX_STANDARD 20)

Include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v2.13.9
)

FetchContent_MakeAvailable(Catch2)

# Required for catch_discover_tests().
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)

# Modify `ctest` to only run the supported subset of tests.
include(CTest)
include(Catch)

add_executable(runner_tests_oqd runner_main.cpp)

target_include_directories(runner_tests_oqd PRIVATE
    ${OQD_LIBRARIES}
    )

target_link_directories(runner_tests_oqd PRIVATE ${runtime_lib})
target_compile_definitions(runner_tests_oqd PUBLIC INITIALIZE_PYTHON)
target_link_libraries(runner_tests_oqd PRIVATE
    Catch2::Catch2
    ${OQD_LIBRARIES}
    )

target_sources(runner_tests_oqd PRIVATE
    Test_OQDDevice.cpp
    )

catch_discover_tests(runner_tests_oqd)
