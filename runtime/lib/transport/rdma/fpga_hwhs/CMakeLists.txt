# FPGA hardware-handshake transport backend for VPK120 controller.
# Built only when ENABLE_TRANSPORT_FPGA is on. Needs the patched headers in
# ../vendor/infiniband

add_library(hwhs_controller_session SHARED
    HwhsControllerSession.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../vendor/VendorVerbsCheck.cpp   # fails the build if the
)                                                                # stock verbs.h wins
set_target_properties(hwhs_controller_session PROPERTIES PREFIX "lib")
target_include_directories(hwhs_controller_session SYSTEM BEFORE PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../vendor
)
target_include_directories(hwhs_controller_session PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${runtime_includes}                        # Transport.hpp, TransportBackend.h
)

target_link_libraries(hwhs_controller_session PUBLIC transport_common_fpga ${CMAKE_DL_LIBS} rt)
set_property(TARGET hwhs_controller_session PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(catalyst_transport_hwhs_controller SHARED HwhsBackend.cpp)
set_target_properties(catalyst_transport_hwhs_controller PROPERTIES PREFIX "lib")
target_link_libraries(catalyst_transport_hwhs_controller PRIVATE hwhs_controller_session)
set_property(TARGET catalyst_transport_hwhs_controller PROPERTY POSITION_INDEPENDENT_CODE ON)
