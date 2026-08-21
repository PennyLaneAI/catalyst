###############################################
# FPGA-verbs transport backend                #
# VPK120 controller, software handshake:      #
# the CPU posts through ibverbs and polls     #
# for the reply.                              #
###############################################
# Built only when ENABLE_TRANSPORT_FPGA is on. The board's libibverbs exposes an
# extra member in struct ibv_context_ops, so this backend compiles against the
# patched headers in ../vendor/infiniband rather than the system's. That shifts
# offsets in a function-pointer table, which is why the include directory is
# added BEFORE any other and scoped to these targets alone.

add_library(fpga_verbs_impl STATIC
    FpgaControllerSession.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../vendor/VendorVerbsCheck.cpp   # fails the build if the
)                                                                # stock verbs.h wins
# SYSTEM, not a plain -I: this is a third-party header and Catalyst builds with
# -Wall -Werror, which otherwise fails on verbs.h's own deprecated enum conversion.
#
# Note SYSTEM does NOT outrank a plain -I — every -I directory is searched before
# any -isystem one, and BEFORE only orders this target's own list. The vendored copy
# wins because no -I directory here contains an infiniband/ subdirectory. That is an
# invariant, not a guarantee, which is what VendorVerbsCheck.cpp above is for.
target_include_directories(fpga_verbs_impl SYSTEM BEFORE PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../vendor      # patched infiniband/verbs.h — must win
)
target_include_directories(fpga_verbs_impl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${runtime_includes}                        # Transport.hpp, TransportBackend.h
)
# transport_common_fpga carries rdma/common and common/interface, plus ibverbs and
# Threads — and is built against the patched verbs.h, so every TU here agrees on
# struct ibv_context. See ../common/CMakeLists.txt.
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

# test/Test_FpgaController.cpp is NOT built from here. BUILD_TESTING is only defined once
# runtime/tests/ runs include(CTest), which happens after lib/ is processed, and Catch2 is
# fetched there too. It belongs beside the other transport tests in runtime/tests/, which is
# where runner_tests_transport_* live.
