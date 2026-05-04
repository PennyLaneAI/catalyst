add_library(rt_qec_tanner_graph SHARED TannerGraph.cpp)

target_include_directories(rt_qec_tanner_graph
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
)

set_property(TARGET rt_qec_tanner_graph PROPERTY POSITION_INDEPENDENT_CODE ON)
