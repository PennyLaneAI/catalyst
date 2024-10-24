cmake_minimum_required(VERSION 3.20)

project(oqc_runtime_tests)

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

add_executable(runner_tests_oqc runner_main.cpp)
add_dependencies(runner_tests_oqc oqc_python_module)
set_property(TARGET runner_tests_oqc APPEND PROPERTY BUILD_RPATH "$<TARGET_FILE_DIR:oqc_python_module>")
if(NOT APPLE)
    set_property(TARGET runner_tests_oqc APPEND PROPERTY BUILD_RPATH $ORIGIN)
    set_property(TARGET runner_tests_oqc APPEND PROPERTY BUILD_RPATH $ORIGIN/../lib)
else()
    set_property(TARGET runner_tests_oqc APPEND PROPERTY BUILD_RPATH @loader_path)
    set_property(TARGET runner_tests_oqc APPEND PROPERTY BUILD_RPATH @loader_path/../lib)
endif()

target_include_directories(runner_tests_oqc PRIVATE
    ${OQC_LIBRARIES}
    )

target_link_directories(runner_tests_oqc PRIVATE ${runtime_lib})
# To avoid link to libpython, we use pybind11::module interface library.
target_compile_definitions(runner_tests_oqc PUBLIC INITIALIZE_PYTHON)
target_link_libraries(runner_tests_oqc PRIVATE
    Catch2::Catch2
    pybind11::embed
    ${OQC_LIBRARIES}
    )

target_sources(runner_tests_oqc PRIVATE
    Test_OpenQASM2Builder.cpp
    Test_OQCDevice.cpp
    )

catch_discover_tests(runner_tests_oqc)
