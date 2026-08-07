###############################################
# Local-copy transport backend                #
###############################################

add_library(local_copy_cpu_impl STATIC
    cpu_controller/LocalCpuControllerSession.cpp
    cpu_coprocessor/LocalCpuCoprocessorSession.cpp
)
target_include_directories(local_copy_cpu_impl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_controller
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_coprocessor
    ${runtime_includes}
)
set_property(TARGET local_copy_cpu_impl PROPERTY POSITION_INDEPENDENT_CODE ON)

if(TRANSPORT_HAS_HIP)
    add_library(local_copy_gpu_impl STATIC
        gpu_coprocessor/LocalGpuCoprocessorSession.cpp
    )
    set_source_files_properties(
        gpu_coprocessor/LocalGpuCoprocessorSession.cpp
        PROPERTIES LANGUAGE HIP)
    target_include_directories(local_copy_gpu_impl PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_coprocessor
        ${runtime_includes}
    )
    target_link_libraries(local_copy_gpu_impl PUBLIC
        catalyst_transport_gpu_coprocessor
        hip::host
    )
    set_property(TARGET local_copy_gpu_impl PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()
