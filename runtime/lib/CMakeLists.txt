add_subdirectory(capi)
add_subdirectory(backend)
add_subdirectory(registry)
add_subdirectory(RSDecompRuntime)

if(ENABLE_OQD)
add_subdirectory(OQDcapi)
endif()
