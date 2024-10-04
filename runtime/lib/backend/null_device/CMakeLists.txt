cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(rtd_null_device SHARED null_device.cpp)
target_include_directories(rtd_null_device PUBLIC ${runtime_includes})
set_property(TARGET rtd_null_device PROPERTY POSITION_INDEPENDENT_CODE ON)