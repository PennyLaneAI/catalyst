cmake_minimum_required(VERSION 3.20)

project(catalyst_oqc)

set(runtime_includes "${PROJECT_SOURCE_DIR}/../../../../../runtime/include")
set(backend_includes "${PROJECT_SOURCE_DIR}/../../../../../runtime/lib/backend/common")
set(runtime_lib "${RUNTIME_BUILD_DIR}/lib")
set(oqc_backend_dir "${OQC_BUILD_DIR}/backend")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Locate Python & PyBind11
# The optional component is only used for the C++ test suite (to spin up their its interpreter),
# and requires libpython.so to be available on the system.
find_package(Python REQUIRED
    COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.Embed
)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)

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

pybind11_add_module(oqc_python_module oqc_python_module.cpp)
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
