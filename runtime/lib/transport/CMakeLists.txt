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
#
# "HIP" above is the HIP language: headers plus a compiler. On AMD both come from an installed
# ROCm; on NVIDIA the compiler is nvcc and the headers are fetched below.

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

# On NVIDIA the HIP language is nvcc with `-x cu`, and ROCm contributes only headers -- hip::host
# is an empty interface target there. So with no headers found above but a CUDA toolkit present,
# fetch just hip-dev (2.5 MB) rather than requiring all of ROCm. An install is used unchanged.
find_path(_hip_include_dir hip/hip_runtime.h PATH_SUFFIXES include)
find_program(_nvcc nvcc HINTS ENV CUDA_PATH ENV CUDA_HOME /usr/local/cuda PATH_SUFFIXES bin)
if(NOT _hip_include_dir AND _nvcc)
    # Pinned, so an upstream release cannot change what we compile against. Set the URL to "" to
    # disable the fetch, e.g. offline builds.
    set(TRANSPORT_HIP_HEADERS_URL
        "https://repo.radeon.com/rocm/apt/6.4.4/pool/main/h/hip-dev/hip-dev_6.4.43484.60404-129~24.04_amd64.deb"
        CACHE STRING "hip-dev package supplying the HIP headers when no ROCm is installed")
    set(TRANSPORT_HIP_HEADERS_SHA256
        "bcbc87aae32ad00db72b33e0b67a66bbfe963d8a248e1ac3600d5ae3076916af"
        CACHE STRING "Expected SHA256 of TRANSPORT_HIP_HEADERS_URL")

    set(_hip_stage "${CMAKE_BINARY_DIR}/hip-headers")
    file(GLOB _hip_root "${_hip_stage}/payload/opt/rocm-*")
    if(NOT _hip_root AND TRANSPORT_HIP_HEADERS_URL)
        message(STATUS "transport: no ROCm installed, fetching the HIP headers")
        # A .deb is an ar archive wrapping data.tar.gz, hence two extractions.
        file(DOWNLOAD "${TRANSPORT_HIP_HEADERS_URL}" "${_hip_stage}/hip-dev.deb"
             EXPECTED_HASH "SHA256=${TRANSPORT_HIP_HEADERS_SHA256}" TLS_VERIFY ON)
        file(ARCHIVE_EXTRACT INPUT "${_hip_stage}/hip-dev.deb" DESTINATION "${_hip_stage}/deb")
        file(ARCHIVE_EXTRACT INPUT "${_hip_stage}/deb/data.tar.gz" DESTINATION "${_hip_stage}/payload")
        file(GLOB _hip_root "${_hip_stage}/payload/opt/rocm-*")
    endif()
    if(_hip_root)
        list(APPEND CMAKE_PREFIX_PATH "${_hip_root}")
        # CMakeDetermineHIPCompiler reads both only when unset, asking hipconfig otherwise --
        # which a fetched header tree has no copy of. No architecture flag belongs here: on this
        # platform CMake takes nvcc's own default, and CUDAARCHS overrides it.
        set(CMAKE_HIP_COMPILER_ROCM_ROOT "${_hip_root}")
        set(CMAKE_HIP_PLATFORM "nvidia" CACHE STRING "HIP platform" FORCE)
    endif()
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

    if(ENABLE_TRANSPORT_FPGA)
        message(STATUS "transport: ENABLE_TRANSPORT_FPGA -> building FPGA controllers")
        add_subdirectory(rdma/fpga_verbs)
        add_dependencies(transport_all catalyst_transport_fpga_verbs_controller)
        add_dependencies(transport_backends catalyst_transport_fpga_verbs_controller)

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
