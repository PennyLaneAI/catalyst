find_package(pybind11 REQUIRED)
pybind11_add_module(registry Registry.cpp)
target_include_directories(registry PUBLIC ${runtime_includes})

pybind11_add_module(pyregistry PyRegistry.cpp)
add_dependencies(pyregistry registry)
#target_link_libraries(pyregistry PRIVATE registry)

