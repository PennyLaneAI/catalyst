###############################################
# GPU-verbs transport backend                 #
###############################################

add_library(catalyst_transport_gpu_verbs_coprocessor SHARED
    coprocessor/GpuCoprocessorSession.cpp
    coprocessor/GpuCoprocessorFactory.cpp
)

target_include_directories(catalyst_transport_gpu_verbs_coprocessor
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/coprocessor
    ${runtime_includes})

target_link_libraries(catalyst_transport_gpu_verbs_coprocessor
    PUBLIC transport_common catalyst_transport_coproc_gpu
    PRIVATE ${IBVERBS_LIB} Threads::Threads hip::host)
if(CMAKE_HIP_COMPILER_ID STREQUAL "NVIDIA")
    # CUDA runtime is injected automatically by hip however hipMemGetHandleForAddressRange
    # requires the CUDA driver symbol cuMemGetHandleForAddressRange so link it here
    target_link_libraries(catalyst_transport_gpu_verbs_coprocessor PRIVATE CUDA::cuda_driver)
endif()
set_property(TARGET catalyst_transport_gpu_verbs_coprocessor PROPERTY POSITION_INDEPENDENT_CODE ON)

# Self-contained loopback application for testing
add_executable(gpu_verbs_selftest gpu_verbs_selftest.cpp)
target_link_libraries(gpu_verbs_selftest PRIVATE catalyst_transport_gpu_verbs_coprocessor)
