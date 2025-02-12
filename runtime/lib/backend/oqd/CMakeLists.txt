cmake_minimum_required(VERSION 3.20)

project(rtd_oqd_device LANGUAGES CXX)

add_library(rtd_oqd_device SHARED OQDDevice.cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_include_directories(rtd_oqd_device PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${runtime_includes}
    ${backend_includes}
)

set_property(TARGET rtd_oqd_device PROPERTY POSITION_INDEPENDENT_CODE ON)
