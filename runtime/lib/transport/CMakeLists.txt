###############################################
# library rt_transport                        #
###############################################

add_library(rt_transport SHARED TransportCAPI.cpp TransportWrappers.cpp)

target_include_directories(rt_transport
    PUBLIC
    ${runtime_includes}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common       # WireProtocol.hpp
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
    add_subdirectory(common)
    add_subdirectory(cpu_verbs)
    add_dependencies(transport_all catalyst_transport_cpu_controller
                     catalyst_transport_cpu_coprocessor steane_coprocessor)

    # GPU coprocessor backend — only when a HIP toolchain is present.
    include(CheckLanguage)
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
        message(STATUS "transport: HIP found (${CMAKE_HIP_COMPILER}) -> building gpu_verbs")
        set(TRANSPORT_HAS_HIP TRUE CACHE INTERNAL "transport: HIP toolchain available")
        add_subdirectory(gpu_verbs)
        add_dependencies(transport_all catalyst_transport_gpu_coprocessor)
    else()
        message(STATUS "transport: no HIP compiler -> skipping gpu_verbs")
    endif()
else()
    message(STATUS
        "transport: libibverbs not found -> building rt_transport only. The bundled backends "
        "are skipped; install libibverbs-dev / rdma-core to build them.")
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
