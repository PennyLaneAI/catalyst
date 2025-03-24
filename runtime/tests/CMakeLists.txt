Include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v2.13.9
)

# Raised by Catch2 with GCC-13, so we disable it here before including their project.
if(RUNTIME_ENABLE_WARNINGS AND (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    add_compile_options("-Wno-maybe-uninitialized")
endif()

FetchContent_MakeAvailable(Catch2)

# Locate PyBind11
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)

# Required for catch_discover_tests().
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)

# Modify `ctest` to only run the supported subset of tests.
include(CTest)
include(Catch)

add_executable(runner_tests_qir_runtime runner_main.cpp)

# For tests we do require libpython in order to embed a Python interpreter.
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
    add_dependencies(runner_tests_openqasm openqasm_python_module)
    set_property(TARGET runner_tests_openqasm APPEND PROPERTY BUILD_RPATH "$<TARGET_FILE_DIR:openqasm_python_module>")
    if(NOT APPLE)
        set_property(TARGET runner_tests_openqasm APPEND PROPERTY BUILD_RPATH $ORIGIN)
        set_property(TARGET runner_tests_openqasm APPEND PROPERTY BUILD_RPATH $ORIGIN/../lib)
    else()
        set_property(TARGET runner_tests_openqasm APPEND PROPERTY BUILD_RPATH @loader_path)
        set_property(TARGET runner_tests_openqasm APPEND PROPERTY BUILD_RPATH @loader_path/../lib)
    endif()

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

if(ENABLE_OQD)
    add_executable(runner_tests_oqd runner_main.cpp)

    if(NOT APPLE)
        set_property(TARGET runner_tests_oqd APPEND PROPERTY BUILD_RPATH $ORIGIN)
        set_property(TARGET runner_tests_oqd APPEND PROPERTY BUILD_RPATH $ORIGIN/../lib)
    else()
        set_property(TARGET runner_tests_oqd APPEND PROPERTY BUILD_RPATH @loader_path)
        set_property(TARGET runner_tests_oqd APPEND PROPERTY BUILD_RPATH @loader_path/../lib)
    endif()

    target_link_libraries(runner_tests_oqd PRIVATE
        Catch2::Catch2
        catalyst_qir_runtime
        rt_OQD_capi
        PRIVATE nlohmann_json::nlohmann_json
    )

    target_sources(runner_tests_oqd PRIVATE
        Test_OQDDevice.cpp
    )

    catch_discover_tests(runner_tests_oqd)
endif()
