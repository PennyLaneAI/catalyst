###############################################
# FPGA hardware-handshake transport backend   #
# VPK120 controller, hardware handshake: the  #
# FPGA engine posts the syndrome and detects  #
# the reply itself, out of BRAM.              #
###############################################
# Built only when ENABLE_TRANSPORT_FPGA is on. Needs the patched headers in
# ../vendor/infiniband for the same reason fpga_verbs does, and at run time it
# needs /dev/mem and /dev/xib0 — the latter from an out-of-tree kernel module
# that is not carried here.
#
# Two shared libraries, not one: a deployed bundle carries the session beside the
# backend and resolves it through LD_LIBRARY_PATH, so the split is part of the
# artifact contract rather than an accident of how it is built.

# The session, as its own shared object.
add_library(hwhs_controller_session SHARED
    HwhsControllerSession.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../vendor/VendorVerbsCheck.cpp   # fails the build if the
)                                                                # stock verbs.h wins
set_target_properties(hwhs_controller_session PROPERTIES PREFIX "lib")
# SYSTEM, and see the note in ../fpga_verbs/CMakeLists.txt on why that does not by
# itself outrank a plain -I.
target_include_directories(hwhs_controller_session SYSTEM BEFORE PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../vendor      # patched infiniband/verbs.h — must win
)
target_include_directories(hwhs_controller_session PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${runtime_includes}                        # Transport.hpp, TransportBackend.h
)
# transport_common_fpga: the patched-verbs.h build of rdma/common, so every TU here
# agrees on struct ibv_context. See ../common/CMakeLists.txt.
target_link_libraries(hwhs_controller_session PUBLIC transport_common_fpga ${CMAKE_DL_LIBS} rt)
set_property(TARGET hwhs_controller_session PROPERTY POSITION_INDEPENDENT_CODE ON)

# The backend the compiler names, linked against the session above.
add_library(catalyst_transport_hwhs_controller SHARED HwhsBackend.cpp)
set_target_properties(catalyst_transport_hwhs_controller PROPERTIES PREFIX "lib")
target_link_libraries(catalyst_transport_hwhs_controller PRIVATE hwhs_controller_session)
set_property(TARGET catalyst_transport_hwhs_controller PROPERTY POSITION_INDEPENDENT_CODE ON)
