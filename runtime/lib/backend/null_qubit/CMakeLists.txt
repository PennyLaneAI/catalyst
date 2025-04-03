cmake_minimum_required(VERSION 3.20)

project(rtd_null_qubit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(rtd_null_qubit SHARED NullQubit.cpp)

target_include_directories(rtd_null_qubit
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
    ${runtime_includes}
    ${backend_utils_includes}
)

set_property(TARGET rtd_null_qubit PROPERTY POSITION_INDEPENDENT_CODE ON)
