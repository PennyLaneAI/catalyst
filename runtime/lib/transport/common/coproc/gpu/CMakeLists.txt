###############################################
# Shared GPU coprocessor plumbing             #
###############################################
# HIP-only. Used by both memcpy_gpu and gpu_verbs.

add_library(catalyst_transport_coproc_gpu STATIC
    GpuRuntime.hip
    GpuLaunchers.hip
)

set_source_files_properties(
    GpuRuntime.hip
    GpuLaunchers.hip
    PROPERTIES LANGUAGE HIP
)

target_include_directories(catalyst_transport_coproc_gpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..   # SteaneLut.hpp (used by steane_decoder.hpp)
    ${runtime_includes}
)

target_link_libraries(catalyst_transport_coproc_gpu PUBLIC
    catalyst_transport_common
    hip::host)
if(CMAKE_HIP_COMPILER_ID STREQUAL "NVIDIA")
    # hipMemGetHandleForAddressRange dispatches through the CUDA driver symbol
    # cuMemGetHandleForAddressRange, which isn't in the HIP runtime alone.
    target_link_libraries(catalyst_transport_coproc_gpu PUBLIC CUDA::cuda_driver)
endif()

set_property(TARGET catalyst_transport_coproc_gpu PROPERTY POSITION_INDEPENDENT_CODE ON)
# HIP kernel handoff store needs optimization even in Debug.
target_compile_options(catalyst_transport_coproc_gpu PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O2>)
