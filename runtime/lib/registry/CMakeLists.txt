fetch_pybind11()
find_package(Python COMPONENTS Interpreter Development)
Python_add_library(catalyst_callback_registry SHARED Registry.cpp)
target_link_libraries(catalyst_callback_registry PRIVATE pybind11::headers)
target_include_directories(catalyst_callback_registry PUBLIC ${runtime_includes})
target_compile_definitions(catalyst_qir_qis_obj PUBLIC -DLIBREGISTRY=\"$<TARGET_FILE_NAME:catalyst_callback_registry>\")

