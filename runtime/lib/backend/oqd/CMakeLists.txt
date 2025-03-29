cmake_minimum_required(VERSION 3.20)

project(rtd_oqd_device LANGUAGES CXX)

message(STATUS "Building the OQD device.")

add_library(rtd_oqd_device SHARED OQDDevice.cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_include_directories(rtd_oqd_device
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
    ${runtime_includes}
    ${backend_utils_includes}
)

set_property(TARGET rtd_oqd_device PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(rtd_oqd_device PRIVATE rt_OQD_capi)
