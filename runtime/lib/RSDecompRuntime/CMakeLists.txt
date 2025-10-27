set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(rtd_rsdecomp SHARED testaa.cpp)

target_include_directories(rtd_rsdecomp
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_property(TARGET rtd_rsdecomp PROPERTY POSITION_INDEPENDENT_CODE ON)
