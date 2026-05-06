
add_library(rt_decoder SHARED LUTDecoder.cpp)

target_include_directories(rt_decoder
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set_property(TARGET rt_decoder PROPERTY POSITION_INDEPENDENT_CODE ON)
