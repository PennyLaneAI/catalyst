set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(decompsolver STATIC DGBuilder.cpp DGSolver.cpp)

target_link_libraries(decompsolver
    PRIVATE Boost::graph
)

target_include_directories(decompsolver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set_property(TARGET decompsolver PROPERTY POSITION_INDEPENDENT_CODE ON)
