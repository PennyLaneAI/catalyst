find_package(pybind11 REQUIRED)
pybind11_add_module(registry SHARED Registry.cpp)
target_include_directories(registry PUBLIC ${runtime_includes})

pybind11_add_module(pyregistry PyRegistry.cpp)
target_link_libraries(pyregistry PRIVATE registry)

target_compile_definitions(catalyst_qir_qis_obj PUBLIC -DLIBREGISTRY=\"$<TARGET_FILE_NAME:registry>\")

