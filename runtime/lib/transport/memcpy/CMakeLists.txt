###############################################
# Memcpy transport backend                    #
###############################################

# memcpy_cpu_impl is SHARED so both memcpy plugins loaded into the same process
# reference one copy of `acquire_memcpy_link` and its static registry map.
add_library(memcpy_cpu_impl SHARED
    MemcpyLink.cpp
    cpu_controller/CpuControllerSession.cpp
    cpu_coprocessor/CpuCoprocessorSession.cpp
)
target_include_directories(memcpy_cpu_impl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_controller
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_coprocessor
    ${runtime_includes}
)
target_link_libraries(memcpy_cpu_impl PUBLIC catalyst_transport_common)
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

# memcpy_gpu depends only on HIP (via the shared catalyst_transport_coproc_gpu).
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
        ${runtime_includes}
    )
    target_link_libraries(memcpy_gpu_impl PUBLIC
        memcpy_cpu_impl
        catalyst_transport_coproc_gpu
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
