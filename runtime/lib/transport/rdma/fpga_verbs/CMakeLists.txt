# FPGA-verbs transport backend for the VPK120 controller (software handshake)
# Built only when ENABLE_TRANSPORT_FPGA is on. Needs the patched headers in
# ../vendor/infiniband

add_library(fpga_verbs_impl STATIC
    FpgaControllerSession.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../vendor/VendorVerbsCheck.cpp   # fails the build if the
)                                                                # stock verbs.h wins

target_include_directories(fpga_verbs_impl SYSTEM BEFORE PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../vendor
)
target_include_directories(fpga_verbs_impl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${runtime_includes}                        # Transport.hpp, TransportBackend.h
)

target_link_libraries(fpga_verbs_impl PUBLIC transport_common_fpga ${CMAKE_DL_LIBS})
set_property(TARGET fpga_verbs_impl PROPERTY POSITION_INDEPENDENT_CODE ON)

# Controller library
add_library(catalyst_transport_fpga_verbs_controller SHARED FpgaControllerFactory.cpp)
target_link_libraries(catalyst_transport_fpga_verbs_controller PRIVATE fpga_verbs_impl)

# Self-contained round-trip application for testing
add_executable(fpga_verbs_selftest fpga_verbs_selftest.cpp)
target_link_libraries(fpga_verbs_selftest PRIVATE fpga_verbs_impl)
# -g + -rdynamic: on-crash backtrace() prints function names (there is no gdb on the board).
target_compile_options(fpga_verbs_selftest PRIVATE -g)
target_link_options(fpga_verbs_selftest PRIVATE -rdynamic)
