cmake_minimum_required(VERSION 3.20)

project(catalyst_oqd)

set(runtime_includes "${PROJECT_SOURCE_DIR}/../../../../../runtime/include")
set(backend_includes "${PROJECT_SOURCE_DIR}/../../../../../runtime/lib/backend/common")
set(runtime_lib "${RUNTIME_BUILD_DIR}/lib")
set(oqd_backend_dir "${OQD_BUILD_DIR}/backend")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Building the OQD device.")

add_library(rtd_oqd SHARED OQDDevice.cpp)

target_include_directories(rtd_oqd PUBLIC .
    ${runtime_includes}
    ${backend_includes}
)

set(OQD_LIBRARIES
    rtd_oqd
)

set_target_properties(rtd_oqd PROPERTIES BUILD_RPATH "$ORIGIN/../utils")
target_link_directories(rtd_oqd PRIVATE ${runtime_lib})

file(COPY ${PROJECT_SOURCE_DIR}/oqd.toml DESTINATION ./backend)

add_subdirectory(tests)
