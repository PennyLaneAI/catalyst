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

target_include_directories(runner_tests PRIVATE 
            ${pennylane_lightning_SOURCE_DIR}/pennylane_lightning/src/tests)

target_link_libraries(runner_tests PRIVATE  catalyst_qir_runtime
            lightning_interfaces
            Catch2::Catch2)

if(ENABLE_CODE_COVERAGE)
    set(cov_helper_src  coverage_helper.cpp)
else()
    set(cov_helper_src  "")
endif()

target_sources(runner_tests PRIVATE ${cov_helper_src}
            Test_QubitManager.cpp
            Test_CacheManager.cpp
            Test_LightningDriver.cpp
            Test_LightningGateSet.cpp
            Test_LightningCoreQIS.cpp
            Test_LightningMeasures.cpp
            Test_LightningGradient.cpp
            Test_SVDynamicCPU_Core.cpp
            Test_SVDynamicCPU_Allocation.cpp)

if(PLKOKKOS_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(runner_tests INTERFACE OpenMP::OpenMP_CXX)
endif()

if(PLKOKKOS_ENABLE_NATIVE)
    message(STATUS "ENABLE_NATIVE is ON. Use -march=native for cpptests.")
    target_compile_options(runner_tests PRIVATE -march=native)
endif()

catch_discover_tests(runner_tests)
