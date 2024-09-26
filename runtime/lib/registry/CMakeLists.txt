fetch_pybind11()
pybind11_add_module(catalyst_callback_registry SHARED Registry.cpp)
target_include_directories(catalyst_callback_registry PUBLIC ${runtime_includes})
target_compile_definitions(catalyst_qir_qis_obj PUBLIC -DLIBREGISTRY=\"$<TARGET_FILE_NAME:catalyst_callback_registry>\")

