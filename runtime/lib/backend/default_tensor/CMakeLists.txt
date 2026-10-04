cmake_minimum_required(VERSION 3.20)

project(rtd_default_tensor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(rtd_default_tensor SHARED DefaultTensor.cpp)

target_include_directories(rtd_default_tensor
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
    ${runtime_includes}
    ${backend_utils_includes}
)

set_property(TARGET rtd_default_tensor PROPERTY POSITION_INDEPENDENT_CODE ON)
