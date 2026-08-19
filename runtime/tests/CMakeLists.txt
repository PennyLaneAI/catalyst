Include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.11.0
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
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

# Modify `ctest` to only run the supported subset of tests.
include(CTest)
include(Catch)

# Set up targets for testing.
add_library(catalyst_runtime_testing INTERFACE)

target_include_directories(catalyst_runtime_testing INTERFACE
    ${runtime_includes}
    ${capi_utils_includes}
    ${backend_utils_includes}
)

target_link_libraries(catalyst_runtime_testing INTERFACE
    ${CMAKE_DL_LIBS}  # dlopen from ExecutionContext.h
    ${devices_list}
    rt_capi
)

if(ENABLE_CODE_COVERAGE)
    if(APPLE)
        target_compile_options(catalyst_runtime_testing INTERFACE -fprofile-instr-generate -fcoverage-mapping)
        target_link_options(catalyst_runtime_testing INTERFACE -fprofile-instr-generate -fcoverage-mapping)
    else()
        target_compile_options(catalyst_runtime_testing INTERFACE -fprofile-arcs -ftest-coverage)
        target_link_libraries(catalyst_runtime_testing INTERFACE gcov)
    endif()
endif()

add_executable(runner_tests_qir_runtime)
target_sources(runner_tests_qir_runtime PRIVATE
    Test_DataView.cpp
    Test_NullQubit.cpp
    Test_ResourceTracker.cpp
    Test_SharedLibraryManager.cpp
)

# For tests we do require libpython in order to embed a Python interpreter.
target_link_libraries(runner_tests_qir_runtime PRIVATE
    Catch2WithMain
    pybind11::embed
    catalyst_runtime_testing
    rtd_null_qubit
)

# Test_SharedLibraryManager dlopens
if(NOT APPLE)
    set_property(TARGET runner_tests_qir_runtime APPEND PROPERTY BUILD_RPATH $ORIGIN)
    set_property(TARGET runner_tests_qir_runtime APPEND PROPERTY BUILD_RPATH $ORIGIN/../lib)
else()
    set_property(TARGET runner_tests_qir_runtime APPEND PROPERTY BUILD_RPATH @loader_path)
    set_property(TARGET runner_tests_qir_runtime APPEND PROPERTY BUILD_RPATH @loader_path/../lib)
endif()

target_compile_definitions(runner_tests_qir_runtime PRIVATE
    RTD_NULL_QUBIT_LIB="$<TARGET_FILE:rtd_null_qubit>"
)

catch_discover_tests(runner_tests_qir_runtime)

if(ENABLE_OPENQASM)
    add_executable(runner_tests_openqasm)
    target_sources(runner_tests_openqasm PRIVATE
        Test_OpenQasmBuilder.cpp
        Test_OpenQasmDevice.cpp
    )

    if(NOT APPLE)
        set_property(TARGET runner_tests_openqasm APPEND PROPERTY BUILD_RPATH $ORIGIN)
        set_property(TARGET runner_tests_openqasm APPEND PROPERTY BUILD_RPATH $ORIGIN/../lib)
    else()
        set_property(TARGET runner_tests_openqasm APPEND PROPERTY BUILD_RPATH @loader_path)
        set_property(TARGET runner_tests_openqasm APPEND PROPERTY BUILD_RPATH @loader_path/../lib)
    endif()

    target_link_libraries(runner_tests_openqasm PRIVATE
        Catch2WithMain
        pybind11::embed
        catalyst_runtime_testing
        rtd_openqasm
    )

    catch_discover_tests(runner_tests_openqasm)
endif()

if(ENABLE_OQD)
    add_executable(runner_tests_oqd)
    target_sources(runner_tests_oqd PRIVATE
        Test_OQDDevice.cpp
    )

    if(NOT APPLE)
        set_property(TARGET runner_tests_oqd APPEND PROPERTY BUILD_RPATH $ORIGIN)
        set_property(TARGET runner_tests_oqd APPEND PROPERTY BUILD_RPATH $ORIGIN/../lib)
    else()
        set_property(TARGET runner_tests_oqd APPEND PROPERTY BUILD_RPATH @loader_path)
        set_property(TARGET runner_tests_oqd APPEND PROPERTY BUILD_RPATH @loader_path/../lib)
    endif()

    target_link_libraries(runner_tests_oqd PRIVATE
        Catch2WithMain
        catalyst_runtime_testing
        rt_OQD_capi
        rtd_oqd_device
        nlohmann_json::nlohmann_json
    )

    catch_discover_tests(runner_tests_oqd)
endif()


# MBQC test suite
add_executable(runner_tests_mbqc_runtime)
target_sources(runner_tests_mbqc_runtime PRIVATE
    Test_MBQC.cpp
)

target_link_libraries(runner_tests_mbqc_runtime PRIVATE
    Catch2WithMain
    catalyst_runtime_testing
    rtd_null_qubit
)

catch_discover_tests(runner_tests_mbqc_runtime)

# Decoder test suite
add_executable(runner_tests_decoder_runtime)
target_sources(runner_tests_decoder_runtime PRIVATE
    Test_DecoderLUTDecoderUtils.cpp
    Test_DecoderLUTDecoder.cpp
)
target_link_libraries(runner_tests_decoder_runtime PRIVATE
    Catch2WithMain
    catalyst_runtime_testing
    rt_decoder
)

catch_discover_tests(runner_tests_decoder_runtime)

# RS Decomposition test suite
add_executable(runner_tests_rsdecomp_runtime)
target_sources(runner_tests_rsdecomp_runtime PRIVATE
    Test_RSDecompUtils.cpp
    Test_RSDecompRings.cpp
    Test_RSDecompEllipse.cpp
    Test_RSDecompGridProblems.cpp
    Test_RSDecompNormalForms.cpp
    Test_RSDecompNormSolver.cpp
    Test_RSDecomp.cpp
)
target_link_libraries(runner_tests_rsdecomp_runtime PRIVATE
    Catch2WithMain
    catalyst_runtime_testing
    rt_rsdecomp
)

catch_discover_tests(runner_tests_rsdecomp_runtime)

# Transport test suites
if(ENABLE_TRANSPORT)
    # CAPI + loader, exercised against a stub backend plugin.
    add_library(stub_transport_backend SHARED stubs/stub_transport_backend.cpp)
    target_include_directories(stub_transport_backend PRIVATE ${runtime_includes})
    set_property(TARGET stub_transport_backend PROPERTY POSITION_INDEPENDENT_CODE ON)

    add_executable(runner_tests_transport)
    target_sources(runner_tests_transport PRIVATE Test_Transport.cpp)
    target_link_libraries(runner_tests_transport PRIVATE
        Catch2WithMain
        rt_transport
    )
    target_compile_definitions(runner_tests_transport PRIVATE
        STUB_BACKEND_PATH="$<TARGET_FILE:stub_transport_backend>"
        MEMCPY_CONTROLLER_BACKEND_PATH="$<TARGET_FILE:catalyst_transport_memcpy_controller>"
        MEMCPY_COPROCESSOR_BACKEND_PATH="$<TARGET_FILE:catalyst_transport_memcpy_coprocessor>")
    add_dependencies(runner_tests_transport
        stub_transport_backend
        catalyst_transport_memcpy_controller
        catalyst_transport_memcpy_coprocessor)
    catch_discover_tests(runner_tests_transport)

    if(IBVERBS_LIB)
        # CPU-verbs backend: device-agnostic common primitives (RDMA-backed; SKIP without rxe0).
        add_executable(runner_tests_transport_common)
        target_sources(runner_tests_transport_common PRIVATE
            Test_TransportCommon.cpp
            Test_TransportWireProtocol.cpp
        )
        target_link_libraries(runner_tests_transport_common PRIVATE
            Catch2WithMain
            catalyst_runtime_testing
            transport_common
        )
        catch_discover_tests(runner_tests_transport_common)

        # CPU-verbs backend: both controller and coprocessor (requires RDMA - skip without rxe0).
        add_executable(runner_tests_transport_cpu)
        target_sources(runner_tests_transport_cpu PRIVATE
            Test_TransportCpuVerbs.cpp
        )
        target_link_libraries(runner_tests_transport_cpu PRIVATE
            Catch2WithMain
            catalyst_runtime_testing
            cpu_verbs_impl
        )
        catch_discover_tests(runner_tests_transport_cpu)

        if(TRANSPORT_HAS_HIP)
            add_executable(runner_tests_transport_gpu)
            target_sources(runner_tests_transport_gpu PRIVATE
                Test_TransportGpu.cpp
            )
            target_link_libraries(runner_tests_transport_gpu PRIVATE
                Catch2WithMain
                catalyst_runtime_testing
                catalyst_transport_gpu_verbs_coprocessor
            )
            catch_discover_tests(runner_tests_transport_gpu)
        endif()
    endif()

    # Memcpy backend: in-process request/reply copies via memcpy.
    add_executable(runner_tests_transport_memcpy)
    target_sources(runner_tests_transport_memcpy PRIVATE
        Test_TransportMemcpyCpu.cpp
    )
    target_link_libraries(runner_tests_transport_memcpy PRIVATE
        Catch2WithMain
        memcpy_cpu_impl
        steane_coprocessor_cpu
    )
    if(TRANSPORT_HAS_HIP)
        target_sources(runner_tests_transport_memcpy PRIVATE
            Test_TransportMemcpyGpu.cpp
        )
        target_link_libraries(runner_tests_transport_memcpy PRIVATE
            memcpy_gpu_impl
        )
    endif()
    catch_discover_tests(runner_tests_transport_memcpy)

    add_custom_target(transport_tests_all)
    add_dependencies(transport_tests_all runner_tests_transport
                     runner_tests_transport_memcpy)
    if(IBVERBS_LIB)
        add_dependencies(transport_tests_all runner_tests_transport_common
                         runner_tests_transport_cpu)
        if(TRANSPORT_HAS_HIP)
            add_dependencies(transport_tests_all runner_tests_transport_gpu)
        endif()
    endif()
endif()

# Executor test suite: spawns the real catalyst-executor binary and drives it
# over a loopback TCP connection with the host-side session API.
if(ENABLE_EXECUTOR)
    add_executable(runner_tests_executor)
    target_sources(runner_tests_executor PRIVATE
        Test_Executor.cpp
    )
    target_link_libraries(runner_tests_executor PRIVATE
        Catch2WithMain
        catalyst_runtime_testing
        catalyst_executor_session
    )
    target_compile_definitions(runner_tests_executor PRIVATE
        CATALYST_EXECUTOR_PATH="$<TARGET_FILE:catalyst-executor>")
    add_dependencies(runner_tests_executor catalyst-executor)

    if(NOT APPLE)
        set_property(TARGET runner_tests_executor APPEND PROPERTY BUILD_RPATH $ORIGIN)
        set_property(TARGET runner_tests_executor APPEND PROPERTY BUILD_RPATH $ORIGIN/../lib)
    else()
        set_property(TARGET runner_tests_executor APPEND PROPERTY BUILD_RPATH @loader_path)
        set_property(TARGET runner_tests_executor APPEND PROPERTY BUILD_RPATH @loader_path/../lib)
    endif()

    catch_discover_tests(runner_tests_executor)
endif()
