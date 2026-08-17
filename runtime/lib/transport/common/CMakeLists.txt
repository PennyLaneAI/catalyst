###############################################
# Transport common — transport-agnostic bits  #
# No RDMA / ibverbs / HIP dependencies.       #
###############################################
# Header-only: WireProtocol.hpp and ConfigParser.hpp. Every transport backend can depend
# on this without dragging in libibverbs or HIP.

add_library(catalyst_transport_common INTERFACE)
target_include_directories(catalyst_transport_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/interface
    ${runtime_includes}
)

# Coprocessor-shared code (CPU plugin implementations + GPU shared plumbing).
add_subdirectory(coproc)
