###############################################
# Coprocessor-shared code                     #
###############################################
# Split by side:
#   * cpu/ — pluggable CoprocessorFn implementations (steane, echo) as SHARED plugins.
#   * gpu/ — HIP-only GPU coprocessor plumbing (GpuRuntime, HipCheck, GpuLaunchers) and
#            device-side decoders. Only built when the HIP language is available; see
#            lib/transport/CMakeLists.txt for what supplies it.

add_subdirectory(cpu)

if(TRANSPORT_HAS_HIP OR TRANSPORT_GPU_STUB)
    add_subdirectory(gpu)
endif()
