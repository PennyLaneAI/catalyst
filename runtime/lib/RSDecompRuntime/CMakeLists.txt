set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Boost REQUIRED)

add_library(rt_rsdecomp SHARED RSDecomp.cpp Rings.cpp CliffordData.cpp)

target_include_directories(rt_rsdecomp
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
target_link_libraries(rt_rsdecomp PUBLIC Boost::headers)

set_property(TARGET rt_rsdecomp PROPERTY POSITION_INDEPENDENT_CODE ON)
