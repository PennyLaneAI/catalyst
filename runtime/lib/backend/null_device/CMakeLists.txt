cmake_minimum_required(VERSION 3.20)

project(rtd_null_qubit LANGUAGES CXX)

add_library(rtd_null_qubit SHARED NullQubit.cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_include_directories(rtd_null_qubit PUBLIC   ${CMAKE_CURRENT_SOURCE_DIR}
                                                    ${runtime_includes}
                                                    )

set_property(TARGET rtd_null_qubit PROPERTY POSITION_INDEPENDENT_CODE ON)
