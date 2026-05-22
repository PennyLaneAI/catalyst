option(CATALYST_ENABLE_PYTHON_CALLBACKS
       "Build the embedded-interpreter callback plugin (PauliRot graph-decomposition)" ON)
if (NOT CATALYST_ENABLE_PYTHON_CALLBACKS)
    return()
endif()

execute_process(
    COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Embed)

add_mlir_library(QuantumPythonCallbacks MODULE
    PythonFunction.cpp
    PythonDriverUtils.cpp
    LINK_LIBS PRIVATE
        pybind11::embed
        Python3::Python
        MLIRIR
        MLIRFuncDialect
        MLIRParser
        quantum-transforms # for DecompCallback
)

target_include_directories(QuantumPythonCallbacks PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(QuantumPythonCallbacks PROPERTIES
    PREFIX "lib"
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

set_source_files_properties(PythonDriverUtils.cpp PythonFunction.cpp
    PROPERTIES COMPILE_FLAGS "-fexceptions -frtti -Wno-covered-switch-default")

set_property(TARGET QuantumPythonCallbacks PROPERTY POSITION_INDEPENDENT_CODE ON)
