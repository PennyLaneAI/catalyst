###############################################
# CPU-verbs transport backend                 #
###############################################

add_library(cpu_verbs_impl STATIC
    base/CpuSessionBase.cpp
    controller/CpuControllerSession.cpp
    coprocessor/CpuCoprocessorSession.cpp
)
target_include_directories(cpu_verbs_impl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/base
    ${CMAKE_CURRENT_SOURCE_DIR}/controller
    ${CMAKE_CURRENT_SOURCE_DIR}/coprocessor
    ${runtime_includes}                      # Transport.hpp, TransportBackend.h
)
target_link_libraries(cpu_verbs_impl PUBLIC transport_common)
set_property(TARGET cpu_verbs_impl PROPERTY POSITION_INDEPENDENT_CODE ON)

# Controller library
add_library(catalyst_transport_cpu_verbs_controller SHARED controller/CpuControllerFactory.cpp)
target_link_libraries(catalyst_transport_cpu_verbs_controller PRIVATE cpu_verbs_impl)

# Coprocessor library
add_library(catalyst_transport_cpu_verbs_coprocessor SHARED coprocessor/CpuCoprocessorFactory.cpp)
target_link_libraries(catalyst_transport_cpu_verbs_coprocessor PRIVATE cpu_verbs_impl)


# Self-contained loopback application for testing
add_executable(cpu_verbs_selftest cpu_verbs_selftest.cpp)
target_link_libraries(cpu_verbs_selftest PRIVATE cpu_verbs_impl)
