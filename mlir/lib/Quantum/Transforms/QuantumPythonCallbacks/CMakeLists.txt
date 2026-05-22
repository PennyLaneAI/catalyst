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

# Development.Module supplies Python.h and extension-module support. We
# deliberately do NOT request Development.Embed, which would also require a
# libpython linker-name file that manylinux images do not ship.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

add_mlir_library(QuantumPythonCallbacks MODULE
    PythonFunction.cpp
    PythonDriverUtils.cpp
    LINK_LIBS PRIVATE
        pybind11::module   # headers + extension-module flags; NO libpython link
        MLIRIR
        MLIRFuncDialect
        MLIRParser
        quantum-transforms # for DecompCallback
)

# Leave libpython symbols (Py_Initialize, PyGILState_Ensure, ...) undefined at
# link time. The dynamic loader will resolve them against whichever libpython
# the loader has dlopen'd into the process by the time this plugin is loaded.
if (APPLE)
    target_link_options(QuantumPythonCallbacks PRIVATE
        "LINKER:-undefined,dynamic_lookup")
else()
    target_link_options(QuantumPythonCallbacks PRIVATE
        "LINKER:--allow-shlib-undefined")
endif()

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

# Install next to the catalyst binary
install(TARGETS QuantumPythonCallbacks
    LIBRARY DESTINATION lib)
