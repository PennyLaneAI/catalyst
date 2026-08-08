###############################################
# GPU-verbs transport backend                 #
###############################################

add_library(catalyst_transport_gpu_coprocessor SHARED
    coprocessor/GpuCoprocessorSession.cpp
    coprocessor/GpuRuntime.hip
    coprocessor/coprocessor_functions/GpuLaunchers.hip
    coprocessor/GpuCoprocessorFactory.cpp
)
set_source_files_properties(
    coprocessor/GpuRuntime.hip
    coprocessor/coprocessor_functions/GpuLaunchers.hip
    PROPERTIES LANGUAGE HIP)

target_include_directories(catalyst_transport_gpu_coprocessor
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/coprocessor
    ${runtime_includes}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/coprocessor/coprocessor_functions)

target_link_libraries(catalyst_transport_gpu_coprocessor
    PUBLIC transport_common
    PRIVATE ${IBVERBS_LIB} Threads::Threads hip::host)
if(CMAKE_HIP_COMPILER_ID STREQUAL "NVIDIA")
    target_link_libraries(catalyst_transport_gpu_coprocessor PRIVATE CUDA::cuda_driver CUDA::cudart)
endif()
set_property(TARGET catalyst_transport_gpu_coprocessor PROPERTY POSITION_INDEPENDENT_CODE ON)
# HIP kernel handoff store needs optimization even in Debug.
target_compile_options(catalyst_transport_gpu_coprocessor PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O2>)

# Self-contained loopback application for testing
add_executable(gpu_verbs_selftest gpu_verbs_selftest.cpp)
target_link_libraries(gpu_verbs_selftest PRIVATE catalyst_transport_gpu_coprocessor)
