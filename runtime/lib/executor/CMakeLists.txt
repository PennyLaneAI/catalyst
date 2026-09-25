###############################################
# catalyst_executor_session + rt_executor + catalyst-executor
#   HOST  rt_executor                 the C ABI a compiled program calls, over handles
#         catalyst_executor_session   the ORC client it drives
#   NODE  catalyst-executor           the ORC server: receives objects, maps them, calls them
###############################################

find_package(LLVM CONFIG REQUIRED)

llvm_map_components_to_libnames(_executor_llvm_libs
    core support orcjit jitlink object passes
    AllTargetsCodeGens AllTargetsDescs AllTargetsInfos
)

# GCC false-positives -Wmaybe-uninitialized in LLVM's unique_function (ADT/FunctionExtras.h),
# reached via ORC headers; -Werror makes it fatal. GCC-only: the executor is built and tested
# against clang in the backline project, which has no such warning and rejects the flag as unknown.
# Applied per target rather than to the directory: only the two that include ORC headers need it,
# and rt_executor keeps the warning fatal (ExecutorSession.hpp pulls in no LLVM).
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(_executor_orc_warning_flags -Wno-error=maybe-uninitialized)
endif()

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

target_compile_options(catalyst_executor_session PRIVATE
    -fno-rtti
    ${_executor_orc_warning_flags}
)

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

# catalyst-executor

llvm_map_components_to_libnames(_executor_bin_llvm_libs
    orcjit orctargetprocess orcshared support
)

add_executable(catalyst-executor executor.cpp)

target_include_directories(catalyst-executor SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

target_link_libraries(catalyst-executor PRIVATE
    ${_executor_bin_llvm_libs}
    pthread
    ${CMAKE_DL_LIBS}
)

target_compile_options(catalyst-executor PRIVATE
    -fno-rtti
    ${_executor_orc_warning_flags}
)

if(APPLE)
    target_link_options(catalyst-executor PRIVATE -Wl,-export_dynamic)
else()
    target_link_options(catalyst-executor PRIVATE -Wl,--export-dynamic)
endif()

set_target_properties(catalyst-executor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
