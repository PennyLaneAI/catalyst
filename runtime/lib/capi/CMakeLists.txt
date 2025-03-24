##################################
# Object Lib catalyst_qir_qis_obj
##################################

add_library(catalyst_qir_qis_obj OBJECT RuntimeCAPI.cpp)

# link to rt_backend
target_link_libraries(catalyst_qir_qis_obj ${CMAKE_DL_LIBS})

target_link_libraries(catalyst_qir_qis_obj
    pthread
    dl
)

target_include_directories(catalyst_qir_qis_obj PUBLIC .
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${runtime_includes}
    ${MLIR_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/../mlir/lib/Driver  # Timer.hpp
)

set_property(TARGET catalyst_qir_qis_obj PROPERTY POSITION_INDEPENDENT_CODE ON)

#####################
# Shared Lib rt_capi
#####################

add_library(rt_capi SHARED)

target_link_libraries(rt_capi ${CMAKE_DL_LIBS} catalyst_qir_qis_obj)
add_dependencies(rt_capi catalyst_callback_registry)


target_include_directories(rt_capi PUBLIC .
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${runtime_includes}
    ${capi_utils_includes}
)

set_property(TARGET rt_capi PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET rt_capi APPEND PROPERTY BUILD_RPATH "$<TARGET_FILE_DIR:catalyst_callback_registry>")

if(NOT APPLE)
    set_property(TARGET rt_capi APPEND PROPERTY BUILD_RPATH $ORIGIN)
else()
    set_property(TARGET rt_capi APPEND PROPERTY BUILD_RPATH @loader_path)
endif()
