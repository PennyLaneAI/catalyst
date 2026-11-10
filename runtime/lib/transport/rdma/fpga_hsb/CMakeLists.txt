# FPGA HSB transport backend for VPK120 controller.
# Built only when ENABLE_TRANSPORT_FPGA is on.

add_library(catalyst_transport_hsb_hwhs_controller SHARED
    HsbBackend.cpp
    HsbControllerSession.cpp
    HsbDevice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/OobSocket.cpp
)
set_target_properties(catalyst_transport_hsb_hwhs_controller PROPERTIES PREFIX "lib")
target_include_directories(catalyst_transport_hsb_hwhs_controller PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common      # OobSocket.hpp
)
target_link_libraries(catalyst_transport_hsb_hwhs_controller PRIVATE catalyst_transport_common)
set_property(TARGET catalyst_transport_hsb_hwhs_controller PROPERTY POSITION_INDEPENDENT_CODE ON)

catalyst_runtime_install(catalyst_transport_hsb_hwhs_controller)
