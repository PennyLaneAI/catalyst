cmake_minimum_required(VERSION 3.20)

project(catalyst_oqc)

set(runtime_includes "${PROJECT_SOURCE_DIR}/../../../../../runtime/include")
set(backend_includes "${PROJECT_SOURCE_DIR}/../../../../../runtime/lib/backend/common")
set(runtime_lib "${RUNTIME_BUILD_DIR}/lib")
set(oqc_backend_dir "${OQC_BUILD_DIR}/backend")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Avoid warning raised by pybind11 on newer cmake versions. PYBIND11_FINDPYTHON=ON caused issues.
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.27")
 cmake_policy(SET CMP0148 OLD)
endif()

include(FetchContent)

function(fetch_pybind11)
    find_package(pybind11 CONFIG)
    if (pybind11_FOUND)
        message(STATUS, "FOUND pybind11")
    else()
        message(STATUS "Could not find existing pybind11-dev package. Building from source.")
        set(CMAKE_POLICY_DEFAULT_CMP0127 NEW) # To suppress pybind11 CMP0127 warning

        FetchContent_Declare(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.10.1
        )

        FetchContent_MakeAvailable(pybind11)
    endif()
endfunction()

fetch_pybind11()

message(STATUS "Building the OQC device.")

add_library(rtd_oqc SHARED OQCDevice.cpp)

target_include_directories(rtd_oqc PUBLIC .
    ${runtime_includes}
    ${backend_includes}
    )

set(OQC_LIBRARIES
    rtd_oqc
)

set_target_properties(rtd_oqc PROPERTIES BUILD_RPATH "$ORIGIN/../utils")
target_link_directories(rtd_oqc PRIVATE ${runtime_lib})

pybind11_add_module(oqc_python_module SHARED oqc_python_module.cpp)
target_include_directories(oqc_python_module PRIVATE ${runtime_includes})

add_dependencies(rtd_oqc oqc_python_module)
target_compile_definitions(rtd_oqc PUBLIC -DOQC_PY=\"$<TARGET_FILE_NAME:oqc_python_module>\")

set_property(TARGET rtd_oqc PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET rtd_oqc APPEND PROPERTY BUILD_RPATH "$<TARGET_FILE_DIR:oqc_python_module>")
if(NOT APPLE)
    set_property(TARGET rtd_oqc APPEND PROPERTY BUILD_RPATH $ORIGIN)
else()
    set_property(TARGET rtd_oqc APPEND PROPERTY BUILD_RPATH @loader_path)
endif()

file(COPY ${PROJECT_SOURCE_DIR}/oqc.toml DESTINATION ./backend)

add_subdirectory(tests)
