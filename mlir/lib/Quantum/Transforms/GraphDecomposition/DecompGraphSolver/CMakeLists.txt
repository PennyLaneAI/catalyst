add_library(decompsolver OBJECT DGBuilder.cpp DGSolver.cpp)

target_include_directories(decompsolver PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_property(TARGET decompsolver PROPERTY POSITION_INDEPENDENT_CODE ON)
