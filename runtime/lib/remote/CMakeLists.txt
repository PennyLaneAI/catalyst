###############################################
# library catalyst_remote_session + rt_remote #
###############################################

find_package(LLVM CONFIG REQUIRED)

llvm_map_components_to_libnames(_remote_llvm_libs
    core support orcjit jitlink object passes
    ${LLVM_TARGETS_TO_BUILD}
)

#  catalyst_remote_session
add_library(catalyst_remote_session SHARED RemoteSession.cpp)

target_include_directories(catalyst_remote_session
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(catalyst_remote_session SYSTEM PRIVATE
    ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(catalyst_remote_session PRIVATE
    rt_capi
    ${_remote_llvm_libs}
    pthread
    ${CMAKE_DL_LIBS}
)

target_compile_options(catalyst_remote_session PRIVATE -fno-rtti)

set_property(TARGET catalyst_remote_session PROPERTY POSITION_INDEPENDENT_CODE ON)

if(NOT APPLE)
    set_property(TARGET catalyst_remote_session APPEND PROPERTY BUILD_RPATH $ORIGIN)
else()
    set_property(TARGET catalyst_remote_session APPEND PROPERTY BUILD_RPATH @loader_path)
endif()

# rt_remote

add_library(rt_remote SHARED RemoteRuntime.cpp)

target_include_directories(rt_remote
    PRIVATE
    ${runtime_includes}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(rt_remote PRIVATE
    catalyst_remote_session
    rt_capi
    pthread
    ${CMAKE_DL_LIBS}
)

set_property(TARGET rt_remote PROPERTY POSITION_INDEPENDENT_CODE ON)

if(NOT APPLE)
    set_property(TARGET rt_remote APPEND PROPERTY BUILD_RPATH $ORIGIN)
else()
    set_property(TARGET rt_remote APPEND PROPERTY BUILD_RPATH @loader_path)
endif()

##############################
# catalyst-executor (server) #
##############################
# The out-of-process executor a local/remote Executor() dispatches to: an ORC SimpleRemoteEPCServer
# over a socket that dlopen's plugins and runs their entries. Exported dynamically so plugin symbols
# are visible to the JIT.

llvm_map_components_to_libnames(_executor_libs orcjit orctargetprocess orcshared support)

add_executable(catalyst-executor executor.cpp)

target_include_directories(catalyst-executor SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

target_link_libraries(catalyst-executor PRIVATE
    ${_executor_libs}
    pthread
    ${CMAKE_DL_LIBS}
)

target_compile_options(catalyst-executor PRIVATE -fno-rtti)

if(NOT APPLE)
    target_link_options(catalyst-executor PRIVATE -Wl,--export-dynamic)
else()
    target_link_options(catalyst-executor PRIVATE -Wl,-export_dynamic)
endif()
