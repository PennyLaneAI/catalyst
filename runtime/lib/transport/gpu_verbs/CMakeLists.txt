###############################################
# GPU-verbs transport backend                 #
###############################################

if(DEFINED ENV{ROCM_PATH})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{ROCM_PATH}")
elseif(EXISTS "/opt/rocm")
    list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
endif()

find_package(hip REQUIRED)
enable_language(HIP)
if(CMAKE_HIP_COMPILER_ID STREQUAL "NVIDIA")
    find_package(CUDAToolkit REQUIRED)
endif()

add_library(catalyst_transport_gpu_verbs_coprocessor SHARED
    coprocessor/GpuCoprocessorSession.cpp
    coprocessor/GpuRuntime.hip
    coprocessor/coprocessor_functions/GpuLaunchers.hip
    coprocessor/GpuCoprocessorFactory.cpp
)
set_source_files_properties(
    coprocessor/GpuRuntime.hip
    coprocessor/coprocessor_functions/GpuLaunchers.hip
    PROPERTIES LANGUAGE HIP)

target_include_directories(catalyst_transport_gpu_verbs_coprocessor
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/coprocessor
    ${runtime_includes}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/coprocessor/coprocessor_functions)

target_link_libraries(catalyst_transport_gpu_verbs_coprocessor
    PUBLIC transport_common
    PRIVATE ${IBVERBS_LIB} Threads::Threads hip::host)
if(CMAKE_HIP_COMPILER_ID STREQUAL "NVIDIA")
    target_link_libraries(catalyst_transport_gpu_verbs_coprocessor PRIVATE CUDA::cuda_driver)
endif()
set_property(TARGET catalyst_transport_gpu_verbs_coprocessor PROPERTY POSITION_INDEPENDENT_CODE ON)
# HIP kernel handoff store needs optimization even in Debug.
target_compile_options(catalyst_transport_gpu_verbs_coprocessor PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O2>)

# Self-contained loopback application for testing
add_executable(gpu_verbs_selftest gpu_verbs_selftest.cpp)
target_link_libraries(gpu_verbs_selftest PRIVATE catalyst_transport_gpu_verbs_coprocessor)
