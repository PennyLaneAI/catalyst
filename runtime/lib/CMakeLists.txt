add_subdirectory(capi)
add_subdirectory(backend)
add_subdirectory(registry)

if(ENABLE_OQD)
add_subdirectory(OQDcapi)
endif()
