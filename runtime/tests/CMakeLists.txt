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

if(ENABLE_LIGHTNING OR ENABLE_LIGHTNING_KOKKOS)
    add_executable(runner_tests_lightning runner_main.cpp)

    add_subdirectory(third_party)

    # Skip dynamic device loading test when sanitizers are enabled or when building on mac,
    # due to the fact that our current support is not fully compatible on macOS.
    # RTLD_DEEPBIND incompatibility.
    if (ENABLE_ADDRESS_SANITIZER OR APPLE)
        set(dl_manager_tests "")
    else()
        # Tests which use dlopen and RTLD_DEEPBIND are incompatible with sanitizers.
        # All tests which go through dlopen need to be tested without sanitizers.
        # They produce the following error message:
        # You are trying to dlopen a this-file-does-not-exist.so shared
        # library with RTLD_DEEPBIND flag which is incompatible with sanitizer
        # runtime (see https://github.com/google/sanitizers/issues/611 for
        # details). If you want to run this-file-does-not-exist.so library under
        # sanitizers please remove RTLD_DEEPBIND from dlopen flags.
        add_dependencies(runner_tests_lightning dummy_device)
        set(dl_manager_tests 
            Test_DLManager.cpp
            )
    endif()

    target_link_libraries(runner_tests_lightning PRIVATE
        Catch2::Catch2
        pybind11::embed
        catalyst_qir_runtime
        )

    target_sources(runner_tests_lightning PRIVATE ${cov_helper_src}
        ${dl_manager_tests}
        Test_QubitManager.cpp
        Test_CacheManager.cpp
        Test_LightningDriver.cpp
        Test_LightningGateSet.cpp
        Test_LightningCoreQIS.cpp
        Test_LightningMeasures.cpp
        Test_LightningGradient.cpp
        Test_SVDynamicCPU_Core.cpp
        Test_SVDynamicCPU_Allocation.cpp
        )

    if(KOKKOS_ENABLE_OPENMP)
        find_package(OpenMP REQUIRED)
        target_link_libraries(runner_tests_lightning INTERFACE OpenMP::OpenMP_CXX)
    endif()

    if(PLKOKKOS_ENABLE_NATIVE)
        message(STATUS "ENABLE_NATIVE is ON. Use -march=native for cpptests.")
        target_compile_options(runner_tests_lightning PRIVATE -march=native)
    endif()

    catch_discover_tests(runner_tests_lightning)
endif()

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
