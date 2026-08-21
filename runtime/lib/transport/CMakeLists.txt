###############################################
# library rt_transport                        #
###############################################

add_library(rt_transport SHARED TransportCAPI.cpp TransportWrappers.cpp)

target_include_directories(rt_transport
    PUBLIC
    ${runtime_includes}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common/interface  # WireProtocol.hpp, ConfigParser.hpp
    ${PROJECT_SOURCE_DIR}/lib/backend/common  # DynamicLibraryLoader.hpp
)

target_link_libraries(rt_transport PRIVATE
    ${CMAKE_DL_LIBS}  # dlopen/dlsym for backend plugins
)

set_property(TARGET rt_transport PROPERTY POSITION_INDEPENDENT_CODE ON)

###############################################
# transport backends                          #
###############################################
# Backends:            built when
#   memcpy             always
#   memcpy_gpu         HIP
#   cpu_verbs (RDMA)   libibverbs
#   gpu_verbs (RDMA)   libibverbs + HIP
#   fpga_verbs (RDMA)  libibverbs + ENABLE_TRANSPORT_FPGA
#   fpga_hwhs (RDMA)   libibverbs + ENABLE_TRANSPORT_FPGA
#
# Shared libs under common/:
#   common/interface   headers (WireProtocol, ConfigParser)         ::common
#   common/coproc/cpu  CoprocessorFn plugins (steane, echo)         ::coproc
#   common/coproc/gpu  HIP plumbing + device decoders               ::coproc

find_package(Threads REQUIRED)
find_library(IBVERBS_LIB ibverbs)

add_custom_target(transport_all)
add_custom_target(transport_backends)
add_dependencies(transport_all rt_transport)

# Detect HIP before add_subdirectory(common) so coproc/gpu can gate on TRANSPORT_HAS_HIP.
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

add_subdirectory(common)

# CoprocessorFn plugins are always built (no RDMA deps); any backend dlopens them at runtime.
add_dependencies(transport_all steane_coprocessor_cpu echo_coprocessor_cpu)
add_dependencies(transport_backends steane_coprocessor_cpu echo_coprocessor_cpu)

# --- RDMA (cpu_verbs / gpu_verbs) --------------------------------------------
if(IBVERBS_LIB)
    add_subdirectory(rdma/common)
    add_subdirectory(rdma/cpu_verbs)
    add_dependencies(transport_all
        catalyst_transport_cpu_verbs_controller
        catalyst_transport_cpu_verbs_coprocessor)
    add_dependencies(transport_backends
        catalyst_transport_cpu_verbs_controller
        catalyst_transport_cpu_verbs_coprocessor)

    if(TRANSPORT_HAS_HIP)
        add_subdirectory(rdma/gpu_verbs)
        add_dependencies(transport_all catalyst_transport_gpu_verbs_coprocessor)
        add_dependencies(transport_backends catalyst_transport_gpu_verbs_coprocessor)
    endif()

    # The FPGA controllers are for one board and are off by default: they compile
    # against patched verbs headers, and at run time they need /dev/mem plus a
    # character device from an out-of-tree kernel module. Instructions for this
    # cross-compilation is in the backline repo.
    if(ENABLE_TRANSPORT_FPGA)
        message(STATUS "transport: ENABLE_TRANSPORT_FPGA -> building FPGA controllers")
        add_subdirectory(rdma/fpga_verbs)
        add_dependencies(transport_all catalyst_transport_fpga_verbs_controller)
        add_dependencies(transport_backends catalyst_transport_fpga_verbs_controller)

        # The hardware-handshake session writes the START doorbell through an aarch64
        # Device-MMIO barrier and #errors out on anything else, so it is the board's
        # architecture or nothing. fpga_verbs above has no such restriction.
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            add_subdirectory(rdma/fpga_hwhs)
            add_dependencies(transport_all catalyst_transport_hwhs_controller)
            add_dependencies(transport_backends catalyst_transport_hwhs_controller)
        else()
            message(STATUS
                "transport: fpga_hwhs skipped -- aarch64 only, this is ${CMAKE_SYSTEM_PROCESSOR}")
        endif()
    endif()
else()
    message(STATUS "transport: libibverbs not found -> RDMA backends skipped")
endif()

# --- Memcpy (in-process shim) -------------------------------------------------
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
