##################################
# Object Lib catalyst_qir_qis_obj
##################################

add_library(catalyst_qir_qis_obj OBJECT RuntimeCAPI.cpp)

# include external MLIR runner utils
FetchContent_MakeAvailable(MLIRRunnerUtils)
FetchContent_MakeAvailable(MLIRCRunnerUtils)

# link to rt_backend
target_link_libraries(catalyst_qir_qis_obj ${CMAKE_DL_LIBS})

if(ENABLE_OPENQASM)
    include(FetchContent)

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

    target_link_libraries(catalyst_qir_qis_obj pybind11::module)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(LINKER_FLAGS_1, "-Wl,-all_load")
  # We can technically use "-Wl,-noall_load", but the linker
  # raises a warning saying that it is a no-op.
  set(LINKER_FLAGS_2, "")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(LINKER_FLAGS_1, "-Wl,--whole-archive")
  set(LINKER_FLAGS_2, "-Wl,--no-whole-archive")
endif()

target_link_libraries(catalyst_qir_qis_obj "-L${QIR_STDLIB_PATH}"
    ${LINKER_FLAGS_1}
    qir_stdlib
    ${LINKER_FLAGS_2}
    pthread
    dl)

target_include_directories(catalyst_qir_qis_obj PUBLIC .
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${QIR_STDLIB_PATH}
    ${runtime_includes}
    ${mlirrunnerutils_SOURCE_DIR}/../..  # includes are relative to mlir/ExecutionEngine
    )

# The MLIR Runner Utils raises this warning so we need to disable it for our -Werror builds.
if(RUNTIME_ENABLE_WARNINGS)
    target_compile_options(catalyst_qir_qis_obj PRIVATE "-Wno-unused-parameter")
endif()

set_property(TARGET catalyst_qir_qis_obj PROPERTY POSITION_INDEPENDENT_CODE ON)

#####################
# Shared Lib rt_capi
#####################

add_library(rt_capi SHARED)

target_link_libraries(rt_capi ${CMAKE_DL_LIBS} catalyst_qir_qis_obj registry)

target_include_directories(rt_capi PUBLIC .
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${QIR_STDLIB_PATH}
    ${runtime_includes}
    ${capi_utils_includes}
    )

set_property(TARGET rt_capi PROPERTY POSITION_INDEPENDENT_CODE ON)
