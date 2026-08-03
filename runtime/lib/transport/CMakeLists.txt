###############################################
# library rt_transport                        #
###############################################

add_library(rt_transport SHARED TransportCAPI.cpp)

target_include_directories(rt_transport
    PUBLIC
    ${runtime_includes}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/lib/backend/common  # DynamicLibraryLoader.hpp
)

target_link_libraries(rt_transport PRIVATE
    ${CMAKE_DL_LIBS}  # dlopen/dlsym for backend plugins
)

set_property(TARGET rt_transport PROPERTY POSITION_INDEPENDENT_CODE ON)
