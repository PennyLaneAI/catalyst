###############################################
# Memcpy transport backend                    #
###############################################

# memcpy_cpu_impl is SHARED (not STATIC) so both memcpy plugins loaded into the same process
# reference one copy of `acquire_memcpy_link` and its static registry map. Linking a STATIC
# impl into two SHARED plugins would give each plugin its own private map, and a controller
# and coprocessor dlopen'd from different .so files could not pair via (peer, oob_port).
add_library(memcpy_cpu_impl SHARED
    MemcpyLink.cpp
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

# Plugins are dlopen'd by absolute path; $ORIGIN / @loader_path lets them find sibling
# libmemcpy_cpu_impl.so at load time without LD_LIBRARY_PATH help.
function(_memcpy_set_plugin_rpath tgt)
    if(APPLE)
        set_property(TARGET ${tgt} APPEND PROPERTY BUILD_RPATH @loader_path)
        set_property(TARGET ${tgt} APPEND PROPERTY INSTALL_RPATH @loader_path)
    else()
        set_property(TARGET ${tgt} APPEND PROPERTY BUILD_RPATH $ORIGIN)
        set_property(TARGET ${tgt} APPEND PROPERTY INSTALL_RPATH $ORIGIN)
    endif()
endfunction()

add_library(catalyst_transport_memcpy_controller SHARED
    cpu_controller/CpuControllerFactory.cpp
)
target_link_libraries(catalyst_transport_memcpy_controller PRIVATE memcpy_cpu_impl)
_memcpy_set_plugin_rpath(catalyst_transport_memcpy_controller)

add_library(catalyst_transport_memcpy_coprocessor SHARED
    cpu_coprocessor/CpuCoprocessorFactory.cpp
)
target_link_libraries(catalyst_transport_memcpy_coprocessor PRIVATE memcpy_cpu_impl)
_memcpy_set_plugin_rpath(catalyst_transport_memcpy_coprocessor)

# memcpy_gpu reuses shared GPU plumbing (GpuRuntime, HipCheck, GpuLaunchers) that currently
# lives under rdma/gpu_verbs/coprocessor/. So today it can only build when both a HIP toolchain
# and libibverbs are present (the latter guards gpu_verbs itself). Extracting the shared GPU
# code into a common library is a follow-up that would let memcpy_gpu build with HIP alone.
if(TRANSPORT_HAS_HIP AND TARGET catalyst_transport_gpu_verbs_coprocessor)
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
        memcpy_cpu_impl
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
    _memcpy_set_plugin_rpath(catalyst_transport_memcpy_gpu_coprocessor)
endif()
