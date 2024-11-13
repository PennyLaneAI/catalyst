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

add_executable(runner_tests_lightning runner_main.cpp)

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
    )

catch_discover_tests(runner_tests_lightning)

if(ENABLE_OPENQASM)
    add_executable(runner_tests_openqasm runner_main.cpp)
    add_dependencies(runner_tests_openqasm openqasm_python_module)
    set_property(TARGET runner_tests_openqasm APPEND PROPERTY BUILD_RPATH "$<TARGET_FILE_DIR:openqasm_python_module>")
    if(NOT APPLE)
        set_property(TARGET runner_tests_openqasm APPEND PROPERTY BUILD_RPATH $ORIGIN)
        set_property(TARGET runner_tests_openqasm APPEND PROPERTY BUILD_RPATH $ORIGIN/../lib)
    else()
        set_property(TARGET runner_tests_openqasm APPEND PROPERTY BUILD_RPATH @loader_path)
        set_property(TARGET runner_tests_openqasm APPEND PROPERTY BUILD_RPATH @loader_path/../lib)
    endif()

    # To avoid link to libpython, we use pybind11::module interface library.
    target_compile_definitions(runner_tests_openqasm PUBLIC INITIALIZE_PYTHON)
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
