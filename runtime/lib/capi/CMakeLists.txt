######################
# Shared Lib rt_capi #
######################

add_library(rt_capi SHARED RuntimeCAPI.cpp)

target_include_directories(rt_capi
    PUBLIC
    ${runtime_includes}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${MLIR_INCLUDE_DIRS}                      # CRunnerUtils.h
    ${PROJECT_SOURCE_DIR}/../mlir/lib/Driver  # Timer.hpp
)

target_link_libraries(rt_capi PRIVATE
    ${CMAKE_DL_LIBS}  # dlopen
    pthread
)

set_property(TARGET rt_capi PROPERTY POSITION_INDEPENDENT_CODE ON)

# We don't directly link against the RT Python module, but we do dlopen it so make dep explicit.
add_dependencies(rt_capi catalyst_callback_registry)

# The catalyst_callback_registry.so will be located in the same directory as the CAPI library:
if(NOT APPLE)
    set_property(TARGET rt_capi APPEND PROPERTY BUILD_RPATH $ORIGIN)
else()
    set_property(TARGET rt_capi APPEND PROPERTY BUILD_RPATH @loader_path)
endif()
