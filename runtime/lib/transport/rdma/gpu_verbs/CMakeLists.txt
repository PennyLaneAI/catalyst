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
    PUBLIC verbs_links catalyst_transport_coproc_gpu
    PRIVATE ${IBVERBS_LIB} Threads::Threads hip::host)
set_property(TARGET catalyst_transport_gpu_verbs_coprocessor PROPERTY POSITION_INDEPENDENT_CODE ON)

catalyst_runtime_install(catalyst_transport_gpu_verbs_coprocessor)

# Self-contained loopback application for testing
add_executable(gpu_verbs_selftest gpu_verbs_selftest.cpp)
target_link_libraries(gpu_verbs_selftest PRIVATE catalyst_transport_gpu_verbs_coprocessor)
