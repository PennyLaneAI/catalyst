add_mlir_library(QuantumUtils
    QuantumSplitting.cpp
    RemoveQuantum.cpp
    PythonFunction.cpp

    LINK_LIBS PRIVATE
    pybind11::embed
)

set_source_files_properties(PythonFunction.cpp PROPERTIES COMPILE_FLAGS "-fexceptions -frtti -Wno-covered-switch-default")
