add_mlir_library(QuantumPythonCallbacks OBJECT PythonFunction.cpp)

target_link_libraries(QuantumPythonCallbacks
    PRIVATE
    pybind11::embed
)

target_include_directories(QuantumPythonCallbacks PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_source_files_properties(PythonFunction.cpp PROPERTIES COMPILE_FLAGS "-fexceptions -frtti -Wno-covered-switch-default")
set_property(TARGET decompsolver PROPERTY POSITION_INDEPENDENT_CODE ON)
