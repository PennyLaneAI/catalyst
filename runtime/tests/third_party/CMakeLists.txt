cmake_minimum_required(VERSION 3.20)

project(third_party_device)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(dummy_device SHARED dummy_device.cpp)
target_include_directories(dummy_device PUBLIC ${runtime_includes})
set_property(TARGET dummy_device PROPERTY POSITION_INDEPENDENT_CODE ON)
configure_file(dummy.device.toml ${CMAKE_BINARY_DIR}/lib/backend/)
