cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(rtd_dummy SHARED dummy_device.cpp)
target_include_directories(rtd_dummy PUBLIC ${runtime_includes})
set_property(TARGET rtd_dummy PROPERTY POSITION_INDEPENDENT_CODE ON)