###############################################
# library catalyst_executor_session + rt_executor #
###############################################

find_package(LLVM CONFIG REQUIRED)

llvm_map_components_to_libnames(_executor_llvm_libs
    core support orcjit jitlink object passes
    ${LLVM_TARGETS_TO_BUILD}
)

#  catalyst_executor_session
add_library(catalyst_executor_session SHARED ExecutorSession.cpp)

target_include_directories(catalyst_executor_session
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(catalyst_executor_session SYSTEM PRIVATE
    ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(catalyst_executor_session PRIVATE
    rt_capi
    ${_executor_llvm_libs}
    pthread
    ${CMAKE_DL_LIBS}
)

target_compile_options(catalyst_executor_session PRIVATE -fno-rtti)

set_property(TARGET catalyst_executor_session PROPERTY POSITION_INDEPENDENT_CODE ON)

if(NOT APPLE)
    set_property(TARGET catalyst_executor_session APPEND PROPERTY BUILD_RPATH $ORIGIN)
else()
    set_property(TARGET catalyst_executor_session APPEND PROPERTY BUILD_RPATH @loader_path)
endif()

# rt_executor

add_library(rt_executor SHARED ExecutorRuntime.cpp)

target_include_directories(rt_executor
    PRIVATE
    ${runtime_includes}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(rt_executor PRIVATE
    catalyst_executor_session
    rt_capi
    pthread
    ${CMAKE_DL_LIBS}
)

set_property(TARGET rt_executor PROPERTY POSITION_INDEPENDENT_CODE ON)

if(NOT APPLE)
    set_property(TARGET rt_executor APPEND PROPERTY BUILD_RPATH $ORIGIN)
else()
    set_property(TARGET rt_executor APPEND PROPERTY BUILD_RPATH @loader_path)
endif()
