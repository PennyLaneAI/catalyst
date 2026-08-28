###############################################
# Coprocessor-shared code                     #
###############################################
# Split by side:
#   * cpu/ — pluggable CoprocessorFn implementations (steane, echo) as SHARED plugins.
#   * gpu/ — HIP-only GPU coprocessor plumbing (GpuRuntime, HipCheck, GpuLaunchers) and
#            device-side decoders. Only built when a HIP toolchain is present.

add_subdirectory(cpu)

if(TRANSPORT_HAS_HIP)
    add_subdirectory(gpu)
endif()
