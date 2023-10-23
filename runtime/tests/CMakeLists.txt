cmake_minimum_required(VERSION 3.21)

project(lightning_qir_runtime_tests)


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

add_executable(runner_tests runner_main.cpp)

add_subdirectory(third_party)

if (ENABLE_SANITIZER)
    set(third_party_tests "")
else()
    # Third party tests are incompatible with sanitizers.
    # They produce the following error message:
    # You are trying to dlopen a this-file-does-not-exist.so shared
    # library with RTLD_DEEPBIND flag which is incompatible with sanitizer
    # runtime (see https://github.com/google/sanitizers/issues/611 for
    # details). If you want to run this-file-does-not-exist.so library under
    # sanitizers please remove RTLD_DEEPBIND from dlopen flags.
    add_dependencies(runner_tests dummy_device)
    set(third_party_tests Test_LoadingThirdParty.cpp)
endif()

if(ENABLE_OPENQASM)
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

# To avoid link to libpython, we use pybind11::module interface library.
target_link_libraries(runner_tests PRIVATE pybind11::embed)

endif()

target_link_libraries(runner_tests PRIVATE  catalyst_qir_runtime
            rt_interfaces
            Catch2::Catch2)

if(ENABLE_CODE_COVERAGE)
    set(cov_helper_src  coverage_helper.cpp)
else()
    set(cov_helper_src  "")
endif()

if (ENABLE_OPENQASM)
    set(openqasm_tests  Test_OpenQasmBuilder.cpp
            Test_OpenQasmDevice.cpp)
else()
    set(openqasm_tests "")
endif()

target_sources(runner_tests PRIVATE ${cov_helper_src}
            ${openqasm_tests}
            ${third_party_tests}
            Test_QubitManager.cpp
            Test_CacheManager.cpp
            Test_LightningDriver.cpp
            Test_LightningGateSet.cpp
            Test_LightningCoreQIS.cpp
            Test_LightningMeasures.cpp
            Test_LightningGradient.cpp
            Test_SVDynamicCPU_Core.cpp
            Test_SVDynamicCPU_Allocation.cpp)

if(KOKKOS_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(runner_tests INTERFACE OpenMP::OpenMP_CXX)
endif()

if(PLKOKKOS_ENABLE_NATIVE)
    message(STATUS "ENABLE_NATIVE is ON. Use -march=native for cpptests.")
    target_compile_options(runner_tests PRIVATE -march=native)
endif()

catch_discover_tests(runner_tests)
