execute_process(
    COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)

add_mlir_library(QuantumPythonCallbacks STATIC
    PythonFunction.cpp
    PythonDriverUtils.cpp
)

target_link_libraries(QuantumPythonCallbacks
    PRIVATE
    pybind11::embed
    Python3::Python
)

target_include_directories(QuantumPythonCallbacks PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_source_files_properties(PythonDriverUtils.cpp PythonFunction.cpp PROPERTIES COMPILE_FLAGS "-fexceptions -frtti -Wno-covered-switch-default")
set_property(TARGET QuantumPythonCallbacks PROPERTY POSITION_INDEPENDENT_CODE ON)
