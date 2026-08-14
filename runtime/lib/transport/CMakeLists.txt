###############################################
# library rt_transport                        #
###############################################

add_library(rt_transport SHARED TransportCAPI.cpp TransportWrappers.cpp)

target_include_directories(rt_transport
    PUBLIC
    ${runtime_includes}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/rdma/common  # WireProtocol.hpp
    ${PROJECT_SOURCE_DIR}/lib/backend/common  # DynamicLibraryLoader.hpp
)

target_link_libraries(rt_transport PRIVATE
    ${CMAKE_DL_LIBS}  # dlopen/dlsym for backend plugins
)

set_property(TARGET rt_transport PROPERTY POSITION_INDEPENDENT_CODE ON)

###############################################
# transport backends                          #
###############################################
#
# Build matrix:
#   * memcpy CPU       — always built (no external deps).
#   * memcpy GPU       — built when HIP + libibverbs are both present. The dependency on
#                        libibverbs is transitive via gpu_verbs (memcpy_gpu currently reuses
#                        GpuRuntime/HipCheck/GpuLaunchers from rdma/gpu_verbs); extracting those
#                        into a shared GPU lib is a follow-up.
#   * cpu_verbs (RDMA) — built when libibverbs is found.
#   * gpu_verbs (RDMA) — built when both libibverbs and HIP are available.

find_package(Threads REQUIRED)
find_library(IBVERBS_LIB ibverbs)

add_custom_target(transport_all)
add_custom_target(transport_backends)
add_dependencies(transport_all rt_transport)

# HIP toolchain detection is independent of libibverbs. Set TRANSPORT_HAS_HIP once, up front,
# so both memcpy_gpu and gpu_verbs can consult it.
if(DEFINED ENV{ROCM_PATH})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{ROCM_PATH}")
elseif(EXISTS "/opt/rocm")
    list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
endif()

include(CheckLanguage)
check_language(HIP)
if(CMAKE_HIP_COMPILER)
    message(STATUS "transport: HIP found (${CMAKE_HIP_COMPILER}) -> GPU backends enabled")
    enable_language(HIP)
    if(DEFINED CMAKE_HIP_PLATFORM)
        set(HIP_PLATFORM "${CMAKE_HIP_PLATFORM}" CACHE STRING "HIP platform" FORCE)
    endif()
    find_package(hip REQUIRED)
    if(CMAKE_HIP_COMPILER_ID STREQUAL "NVIDIA")
        find_package(CUDAToolkit REQUIRED)
    endif()
    set(TRANSPORT_HAS_HIP TRUE CACHE INTERNAL "transport: HIP toolchain available")
else()
    message(STATUS "transport: no HIP compiler -> GPU backends skipped")
endif()

# --- RDMA (cpu_verbs / gpu_verbs) --------------------------------------------
# Requires libibverbs. gpu_verbs additionally needs a HIP toolchain.

if(IBVERBS_LIB)
    add_subdirectory(rdma/common)
    add_subdirectory(rdma/cpu_verbs)
    add_dependencies(transport_all
        catalyst_transport_cpu_verbs_controller
        catalyst_transport_cpu_verbs_coprocessor
        steane_coprocessor_cpu)
    add_dependencies(transport_backends
        catalyst_transport_cpu_verbs_controller
        catalyst_transport_cpu_verbs_coprocessor
        steane_coprocessor_cpu)

    if(TRANSPORT_HAS_HIP)
        add_subdirectory(rdma/gpu_verbs)
        add_dependencies(transport_all catalyst_transport_gpu_verbs_coprocessor)
        add_dependencies(transport_backends catalyst_transport_gpu_verbs_coprocessor)
    endif()
else()
    message(STATUS
        "transport: libibverbs not found -> skipping the RDMA backends (cpu_verbs, gpu_verbs); "
        "rt_transport and the memcpy CPU backend still build. "
        "Install libibverbs-dev / rdma-core to get the RDMA backends.")
endif()

# --- Memcpy (in-process shim) -------------------------------------------------
# Always built. memcpy_gpu is added here (memcpy is processed after rdma/gpu_verbs so it can
# `TARGET catalyst_transport_gpu_verbs_coprocessor` when deciding whether to build memcpy_gpu).

add_subdirectory(memcpy)
add_dependencies(transport_all
    catalyst_transport_memcpy_controller
    catalyst_transport_memcpy_coprocessor)
add_dependencies(transport_backends
    catalyst_transport_memcpy_controller
    catalyst_transport_memcpy_coprocessor)
if(TARGET catalyst_transport_memcpy_gpu_coprocessor)
    add_dependencies(transport_all catalyst_transport_memcpy_gpu_coprocessor)
    add_dependencies(transport_backends catalyst_transport_memcpy_gpu_coprocessor)
endif()
