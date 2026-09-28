###############################################
# CPU coprocessor function libraries          #
###############################################
# Pluggable `CoprocessorFn` implementations that any transport backend can dlopen at run
# time. Pure C++ decoder implementations against the
# CoprocessorFn signature in Transport.hpp.

add_library(steane_coprocessor_cpu SHARED steane_decoder_fn.cpp)
target_include_directories(steane_coprocessor_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..) # SteaneLut.hpp
target_link_libraries(steane_coprocessor_cpu PRIVATE catalyst_transport_common)
set_property(TARGET steane_coprocessor_cpu PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(echo_coprocessor_cpu SHARED echo_coprocessor_fn.cpp)
target_link_libraries(echo_coprocessor_cpu PRIVATE catalyst_transport_common)
set_property(TARGET echo_coprocessor_cpu PROPERTY POSITION_INDEPENDENT_CODE ON)
