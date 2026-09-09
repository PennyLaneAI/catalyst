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

# Build the GPU backends against host memory when no HIP toolchain is present, so the session, the
# ring/handoff protocol and gpu_verbs_selftest stay runnable off GPU hosts. Never both: HIP wins.
option(TRANSPORT_GPU_STUB
       "Build the GPU coprocessor against host memory instead of HIP (testing only)" OFF)

# Detect HIP before add_subdirectory(common) so coproc/gpu can gate on TRANSPORT_HAS_HIP.
if(DEFINED ENV{ROCM_PATH})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{ROCM_PATH}")
elseif(EXISTS "/opt/rocm")
    list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
endif()

# On NVIDIA nvcc compiles and ROCm supplies only headers -- fetch hip-dev (2.5 MB), not all of ROCm.
find_path(_hip_include_dir hip/hip_runtime.h PATH_SUFFIXES include)
find_program(_nvcc nvcc HINTS ENV CUDA_PATH ENV CUDA_HOME /usr/local/cuda PATH_SUFFIXES bin)
if(NOT _hip_include_dir AND _nvcc)
    # 7.1.x or newer required: hipMemGetHandleForAddressRange and hipMemRangeHandleTypeDmaBufFd
    # are absent from 6.4.4, and 7.1.1 maps them onto cuMemGetHandleForAddressRange.
    # Both packages are MIT (c) AMD; fetched, not redistributed, so no NOTICE entry is needed.
    set(TRANSPORT_HIP_PACKAGES
        "https://repo.radeon.com/rocm/apt/7.1.1/pool/main/h/hip-dev/hip-dev_7.1.52802.70101-38~24.04_amd64.deb|736acec734396969e11e8813a55742de536c53324fab3a836d75aa3dbf0d9651"
        "https://repo.radeon.com/rocm/apt/7.1.1/pool/main/h/hipcc-nvidia/hipcc-nvidia_1.1.1.70101-38~24.04_amd64.deb|221a4a0923862cd236d3b9e5b76fd829a87ba171ad08c75b69a6f6984a18ad30"
        CACHE STRING "url|sha256 of the ROCm HIP packages; empty disables the fetch")

    set(_hip_stage "${CMAKE_BINARY_DIR}/hip-headers")
    file(GLOB _hip_root "${_hip_stage}/payload/opt/rocm-*")
    if(NOT _hip_root AND TRANSPORT_HIP_PACKAGES)
        message(STATUS "transport: no ROCm installed, fetching the HIP headers")
        set(_hip_pkg_n 0)
        foreach(_hip_pkg IN LISTS TRANSPORT_HIP_PACKAGES)
            string(REPLACE "|" ";" _hip_pair "${_hip_pkg}")
            list(GET _hip_pair 0 _hip_url)
            list(GET _hip_pair 1 _hip_sha)
            math(EXPR _hip_pkg_n "${_hip_pkg_n} + 1")
            # A .deb is an ar archive wrapping data.tar.gz, hence two extractions.
            file(DOWNLOAD "${_hip_url}" "${_hip_stage}/pkg${_hip_pkg_n}.deb"
                 EXPECTED_HASH "SHA256=${_hip_sha}" TLS_VERIFY ON)
            file(ARCHIVE_EXTRACT INPUT "${_hip_stage}/pkg${_hip_pkg_n}.deb"
                 DESTINATION "${_hip_stage}/deb${_hip_pkg_n}")
            file(ARCHIVE_EXTRACT INPUT "${_hip_stage}/deb${_hip_pkg_n}/data.tar.gz"
                 DESTINATION "${_hip_stage}/payload")
        endforeach()
        file(GLOB _hip_root "${_hip_stage}/payload/opt/rocm-*")
    endif()
    if(_hip_root)
        list(APPEND CMAKE_PREFIX_PATH "${_hip_root}")
        # CMakeDetermineHIPCompiler reads these only when unset, asking hipconfig otherwise --
        # which a fetched header tree has no copy of. No architecture flag belongs here: on this
        # platform CMake takes nvcc's own default, and CUDAARCHS overrides it.
        set(CMAKE_HIP_COMPILER_ROCM_ROOT "${_hip_root}")
        set(CMAKE_HIP_PLATFORM "nvidia" CACHE STRING "HIP platform" FORCE)
        # Naming the compiler matters for the same reason. check_language(HIP) probes in a nested
        # project that inherits CMAKE_HIP_PLATFORM but not the root above, so it falls through to
        # hipconfig and aborts with "Failed to find HIP root directory". Setting it here makes
        # check_language a no-op -- it only probes when undefined -- and leaves enable_language to
        # run in this scope, where the root is set. nvcc is what CMake picks on this platform.
        set(CMAKE_HIP_COMPILER "${_nvcc}" CACHE FILEPATH "HIP compiler")
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
elseif(TRANSPORT_GPU_STUB)
    message(STATUS "transport: TRANSPORT_GPU_STUB -> GPU backends built against host memory")
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

    if(TRANSPORT_HAS_HIP OR TRANSPORT_GPU_STUB)
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
