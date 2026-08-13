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
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${runtime_includes}
)
set_property(TARGET local_copy_cpu_impl PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(catalyst_transport_local_controller SHARED
    cpu_controller/LocalCpuControllerFactory.cpp
)
target_link_libraries(catalyst_transport_local_controller PRIVATE local_copy_cpu_impl)

add_library(catalyst_transport_local_coprocessor SHARED
    cpu_coprocessor/LocalCpuCoprocessorFactory.cpp
)
target_link_libraries(catalyst_transport_local_coprocessor PRIVATE local_copy_cpu_impl)

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
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../gpu_verbs/coprocessor
        ${CMAKE_CURRENT_SOURCE_DIR}/../gpu_verbs/coprocessor/coprocessor_functions
        ${runtime_includes}
    )
    target_link_libraries(local_copy_gpu_impl PUBLIC
        catalyst_transport_gpu_verbs_coprocessor
        hip::host
    )
    if(CMAKE_HIP_COMPILER_ID STREQUAL "NVIDIA")
        target_link_libraries(local_copy_gpu_impl PUBLIC CUDA::cudart)
    endif()
    set_property(TARGET local_copy_gpu_impl PROPERTY POSITION_INDEPENDENT_CODE ON)

    add_library(catalyst_transport_local_gpu_coprocessor SHARED
        gpu_coprocessor/LocalGpuCoprocessorFactory.cpp
    )
    target_link_libraries(catalyst_transport_local_gpu_coprocessor PRIVATE local_copy_gpu_impl)
endif()
