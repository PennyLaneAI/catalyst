###############################################
# library transport_common (RDMA plumbing)    #
# Device-agnostic RDMA verbs primitives       #
# shared by cpu_verbs and gpu_verbs.          #
###############################################
# Transport-agnostic bits (WireProtocol, ConfigParser) live in ../../common/.

add_library(transport_common STATIC
    CompletionQueue.cpp
    Context.cpp
    MemoryRegion.cpp
    OobSocket.cpp
    ProtectionDomain.cpp
    QueuePair.cpp
)

target_include_directories(transport_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(transport_common
    PUBLIC catalyst_transport_common ${IBVERBS_LIB} Threads::Threads)

set_property(TARGET transport_common PROPERTY POSITION_INDEPENDENT_CODE ON)
