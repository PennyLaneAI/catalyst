###############################################
# library transport_common (RDMA plumbing)    #
# Device-agnostic RDMA verbs primitives       #
# shared by cpu_verbs and gpu_verbs.          #
###############################################
# Transport-agnostic bits (WireProtocol, ConfigParser) live in ../../common/.

set(TRANSPORT_COMMON_SOURCES
    CompletionQueue.cpp
    Context.cpp
    MemoryRegion.cpp
    OobSocket.cpp
    ProtectionDomain.cpp
    QueuePair.cpp
)

add_library(transport_common STATIC ${TRANSPORT_COMMON_SOURCES})

target_include_directories(transport_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(transport_common
    PUBLIC catalyst_transport_common ${IBVERBS_LIB} Threads::Threads)

set_property(TARGET transport_common PROPERTY POSITION_INDEPENDENT_CODE ON)

if(ENABLE_TRANSPORT_FPGA)
    add_library(transport_common_fpga STATIC ${TRANSPORT_COMMON_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/../vendor/VendorVerbsCheck.cpp)
    target_include_directories(transport_common_fpga SYSTEM BEFORE PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../vendor)
    target_include_directories(transport_common_fpga PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(transport_common_fpga
        PUBLIC catalyst_transport_common ${IBVERBS_LIB} Threads::Threads)
    set_property(TARGET transport_common_fpga PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()
