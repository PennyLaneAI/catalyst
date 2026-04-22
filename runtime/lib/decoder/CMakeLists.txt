set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(rt_decoder SHARED lut_decoder.cpp)

# Boost targets are already defined by the parent runtime/lib/CMakeLists.txt.
# rt_decoder is a SHARED library, so PUBLIC linking ensures Boost symbols
# are resolved at load time.

target_include_directories(rt_decoder
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set_property(TARGET rt_decoder PROPERTY POSITION_INDEPENDENT_CODE ON)
