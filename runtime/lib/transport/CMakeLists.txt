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

find_package(Threads REQUIRED)
find_library(IBVERBS_LIB ibverbs)

add_custom_target(transport_all)
add_dependencies(transport_all rt_transport)

if(IBVERBS_LIB)
    add_subdirectory(rdma/common)
    add_subdirectory(rdma/cpu_verbs)
    add_dependencies(transport_all catalyst_transport_cpu_verbs_controller
                     catalyst_transport_cpu_verbs_coprocessor steane_coprocessor_cpu)

    if(DEFINED ENV{ROCM_PATH})
        list(APPEND CMAKE_PREFIX_PATH "$ENV{ROCM_PATH}")
    elseif(EXISTS "/opt/rocm")
        list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
    endif()

    # GPU-backed transport pieces — only when a HIP toolchain is present.
    include(CheckLanguage)
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
        message(STATUS "transport: HIP found (${CMAKE_HIP_COMPILER}) -> building gpu_verbs")
        enable_language(HIP)
        if(DEFINED CMAKE_HIP_PLATFORM)
            set(HIP_PLATFORM "${CMAKE_HIP_PLATFORM}" CACHE STRING "HIP platform" FORCE)
        endif()
        find_package(hip REQUIRED)
        if(CMAKE_HIP_COMPILER_ID STREQUAL "NVIDIA")
            find_package(CUDAToolkit REQUIRED)
        endif()
        set(TRANSPORT_HAS_HIP TRUE CACHE INTERNAL "transport: HIP toolchain available")
        add_subdirectory(rdma/gpu_verbs)
        add_dependencies(transport_all catalyst_transport_gpu_verbs_coprocessor)
    else()
        message(STATUS "transport: no HIP compiler -> skipping gpu_verbs")
    endif()
else()
    message(STATUS
        "transport: libibverbs not found -> skipping the RDMA backends (cpu_verbs, gpu_verbs, "
        "and the shared transport_common); rt_transport and the memcpy backend still build. "
        "Install libibverbs-dev / rdma-core to get the RDMA backends.")
endif()

add_custom_target(transport_backends
    DEPENDS
    catalyst_transport_cpu_verbs_controller
    catalyst_transport_cpu_verbs_coprocessor
    steane_coprocessor_cpu
)
if(TRANSPORT_HAS_HIP)
    add_dependencies(transport_backends catalyst_transport_gpu_verbs_coprocessor)
endif()

add_subdirectory(memcpy)
add_dependencies(transport_all catalyst_transport_memcpy_controller
                 catalyst_transport_memcpy_coprocessor)
if(TRANSPORT_HAS_HIP)
    add_dependencies(transport_all catalyst_transport_memcpy_gpu_coprocessor)
endif()
