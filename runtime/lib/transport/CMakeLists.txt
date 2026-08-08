###############################################
# transport backends: shared dependencies     #
###############################################

find_package(Threads REQUIRED)
find_library(IBVERBS_LIB ibverbs)
if(NOT IBVERBS_LIB)
    message(FATAL_ERROR
        "ENABLE_TRANSPORT requires libibverbs (e.g. libibverbs-dev / rdma-core), but it was not found.")
endif()

###############################################
# library rt_transport                        #
###############################################

add_library(rt_transport SHARED TransportCAPI.cpp)

target_include_directories(rt_transport
    PUBLIC
    ${runtime_includes}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/lib/backend/common  # DynamicLibraryLoader.hpp
)

target_link_libraries(rt_transport PRIVATE
    ${CMAKE_DL_LIBS}  # dlopen/dlsym for backend plugins
)

set_property(TARGET rt_transport PROPERTY POSITION_INDEPENDENT_CODE ON)

###############################################
# transport backends                          #
###############################################

add_subdirectory(common)
add_subdirectory(cpu_verbs)

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
    add_subdirectory(gpu_verbs)
else()
    message(STATUS "transport: no HIP compiler -> skipping gpu_verbs")
endif()

add_subdirectory(local_copy)
