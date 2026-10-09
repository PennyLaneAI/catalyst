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

# Search multiple layers for the HIP tooling: the path env-var, a venv install, and system install
if(DEFINED ENV{ROCM_PATH})
    set(ROCM_ROOT "$ENV{ROCM_PATH}")
elseif(DEFINED ENV{VIRTUAL_ENV} AND EXISTS "$ENV{VIRTUAL_ENV}/bin/rocm-sdk")
    message("!!!!!!!!!!!!!!HERE 2")
    execute_process(
        COMMAND rocm-sdk path --root
        OUTPUT_VARIABLE ROCM_SDK_ROOT_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE ROCM_SDK_RESULT
    )
    if(ROCM_SDK_RESULT EQUAL 0 AND EXISTS "${ROCM_SDK_ROOT_DIR}")
        message(STATUS "Found rocm-sdk root inside venv: ${ROCM_SDK_ROOT_DIR}")

        # Force CMake to find hip-lang-config.cmake inside the venv
        set(CMAKE_HIP_COMPILER_ROCM_ROOT "${ROCM_SDK_ROOT_DIR}" CACHE PATH "ROCm SDK root")
        set(CMAKE_PREFIX_PATH "${ROCM_SDK_ROOT_DIR}" ${CMAKE_PREFIX_PATH})

        # Force where amdclang++ lives
        set(CMAKE_HIP_COMPILER "${ROCM_SDK_ROOT_DIR}/llvm/bin/amdclang++" CACHE FILEPATH "HIP Compiler")

        set(ENV{HIP_PATH} "${ROCM_SDK_ROOT_DIR}")
        set(ENV{ROCM_PATH} "${ROCM_SDK_ROOT_DIR}")
        set(ENV{HIPCXX} "${ROCM_SDK_ROOT_DIR}/llvm/bin/clang++")
        
        # Also seed the standard cache variables just in case
        set(CMAKE_HIP_COMPILER "${ROCM_SDK_ROOT_DIR}/llvm/bin/clang++" CACHE FILEPATH "HIP Compiler" FORCE)
        set(CMAKE_HIP_COMPILER_ROCM_ROOT "${ROCM_SDK_ROOT_DIR}" CACHE PATH "ROCm SDK root" FORCE)
        set(CMAKE_PREFIX_PATH "${ROCM_SDK_ROOT_DIR}" ${CMAKE_PREFIX_PATH})

        list(APPEND CMAKE_BUILD_RPATH "${ROCM_SDK_ROOT_DIR}/lib")
    else()
        message(WARNING "rocm-sdk CLI failed or was not found in the active environment.")
    endif()

elseif(EXISTS "/opt/rocm")
    set(ROCM_ROOT "/opt/rocm")
endif()


# If value is set, force the compiler bin path to the above
if(ROCM_ROOT AND NOT CMAKE_HIP_COMPILER)
    list(APPEND CMAKE_BUILD_RPATH "${ROCM_ROOT}/lib")
    if(NOT CMAKE_HIP_COMPILER) # SDK found, but nothing set yet
        list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")
        set(CMAKE_HIP_COMPILER_ROCM_ROOT "${ROCM_ROOT}" CACHE PATH "ROCm root directory")

        if(NOT CMAKE_HIP_COMPILER AND EXISTS "${ROCM_ROOT}/bin/amdclang++")
            set(CMAKE_HIP_COMPILER "${ROCM_ROOT}/bin/amdclang++" CACHE FILEPATH "HIP compiler")
        endif()
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
# The ONNX CoprocessorFn library is absent when CATALYST_TRANSPORT_ONNX is OFF or its header is
# unavailable (see common/coproc/onnx).
if(TARGET catalyst_onnx_coprocessor)
    add_dependencies(transport_all catalyst_onnx_coprocessor)
    add_dependencies(transport_backends catalyst_onnx_coprocessor)
endif()

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
