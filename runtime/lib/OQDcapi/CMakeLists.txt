##################################
# Object Lib catalyst_oqd_obj
##################################

add_library(catalyst_oqd_obj OBJECT OQDRuntimeCAPI.cpp)

# link to rt_backend
target_link_libraries(catalyst_oqd_obj ${CMAKE_DL_LIBS})

target_link_libraries(catalyst_oqd_obj
    pthread
    dl
)

target_include_directories(catalyst_oqd_obj PUBLIC .
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${runtime_includes}
    ${PROJECT_SOURCE_DIR}/../mlir/lib/Driver  # Timer.hpp
)


#####################
# Shared Lib rt_OQD_capi
#####################

add_library(rt_OQD_capi SHARED)

target_link_libraries(rt_OQD_capi ${CMAKE_DL_LIBS} catalyst_oqd_obj)
add_dependencies(rt_OQD_capi catalyst_callback_registry)


target_include_directories(rt_OQD_capi PUBLIC .
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${runtime_includes}
    ${capi_utils_includes}
)

set_property(TARGET rt_OQD_capi PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET rt_OQD_capi APPEND PROPERTY BUILD_RPATH "$<TARGET_FILE_DIR:catalyst_callback_registry>")

if(NOT APPLE)
    set_property(TARGET rt_OQD_capi APPEND PROPERTY BUILD_RPATH $ORIGIN)
else()
    set_property(TARGET rt_OQD_capi APPEND PROPERTY BUILD_RPATH @loader_path)
endif()
