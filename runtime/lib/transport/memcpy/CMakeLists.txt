###############################################
# Memcpy transport backend                    #
###############################################

add_library(memcpy_cpu_impl STATIC
    cpu_controller/CpuControllerSession.cpp
    cpu_coprocessor/CpuCoprocessorSession.cpp
)
target_include_directories(memcpy_cpu_impl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_controller
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_coprocessor
    ${CMAKE_CURRENT_SOURCE_DIR}/../rdma/common
    ${runtime_includes}
)
set_property(TARGET memcpy_cpu_impl PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(catalyst_transport_memcpy_controller SHARED
    cpu_controller/CpuControllerFactory.cpp
)
target_link_libraries(catalyst_transport_memcpy_controller PRIVATE memcpy_cpu_impl)

add_library(catalyst_transport_memcpy_coprocessor SHARED
    cpu_coprocessor/CpuCoprocessorFactory.cpp
)
target_link_libraries(catalyst_transport_memcpy_coprocessor PRIVATE memcpy_cpu_impl)

if(TRANSPORT_HAS_HIP)
    add_library(memcpy_gpu_impl STATIC
        gpu_coprocessor/GpuCoprocessorSession.cpp
    )
    set_source_files_properties(
        gpu_coprocessor/GpuCoprocessorSession.cpp
        PROPERTIES LANGUAGE HIP)
    target_include_directories(memcpy_gpu_impl PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_coprocessor
        ${CMAKE_CURRENT_SOURCE_DIR}/../rdma/common
        ${CMAKE_CURRENT_SOURCE_DIR}/../rdma/gpu_verbs/coprocessor
        ${CMAKE_CURRENT_SOURCE_DIR}/../rdma/gpu_verbs/coprocessor/coprocessor_functions
        ${runtime_includes}
    )
    target_link_libraries(memcpy_gpu_impl PUBLIC
        catalyst_transport_gpu_verbs_coprocessor
        hip::host
    )
    if(CMAKE_HIP_COMPILER_ID STREQUAL "NVIDIA")
        target_link_libraries(memcpy_gpu_impl PUBLIC CUDA::cudart)
    endif()
    set_property(TARGET memcpy_gpu_impl PROPERTY POSITION_INDEPENDENT_CODE ON)

    add_library(catalyst_transport_memcpy_gpu_coprocessor SHARED
        gpu_coprocessor/GpuCoprocessorFactory.cpp
    )
    target_link_libraries(catalyst_transport_memcpy_gpu_coprocessor PRIVATE memcpy_gpu_impl)
endif()
