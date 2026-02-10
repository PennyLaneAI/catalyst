##################################
# Object Lib catalyst_oqd_obj
##################################

add_library(catalyst_oqd_obj OBJECT OQDRuntimeCAPI.cpp)

# Fetch json utilities
include(FetchContent)
FetchContent_Declare(
   json
   URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
   URL_HASH  SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
)
FetchContent_MakeAvailable(json)

# link to rt_backend and json utils
target_link_libraries(catalyst_oqd_obj
    PUBLIC ${CMAKE_DL_LIBS}
    PRIVATE nlohmann_json::nlohmann_json
)

target_include_directories(catalyst_oqd_obj PUBLIC .
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${runtime_includes}
)

set_property(TARGET catalyst_oqd_obj PROPERTY POSITION_INDEPENDENT_CODE ON)

#####################
# Shared Lib rt_OQD_capi
#####################

add_library(rt_OQD_capi SHARED)

target_link_libraries(rt_OQD_capi ${CMAKE_DL_LIBS} catalyst_oqd_obj)

target_include_directories(rt_OQD_capi PUBLIC .
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${runtime_includes}
    ${capi_utils_includes}
)

if(NOT APPLE)
    set_property(TARGET rt_OQD_capi APPEND PROPERTY BUILD_RPATH $ORIGIN)
else()
    set_property(TARGET rt_OQD_capi APPEND PROPERTY BUILD_RPATH @loader_path)
endif()
