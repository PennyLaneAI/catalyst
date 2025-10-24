cmake_minimum_required(VERSION 3.20)

project(catalyst_ftqc)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build the shared library
add_library(rtd_ftqc SHARED FTQCDevice.cpp)

target_include_directories(rtd_ftqc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set_property(TARGET rtd_ftqc PROPERTY POSITION_INDEPENDENT_CODE ON)

# Copy TOML config file
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ftqc.toml DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
