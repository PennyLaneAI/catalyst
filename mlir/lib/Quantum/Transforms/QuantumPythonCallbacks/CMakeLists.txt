option(CATALYST_ENABLE_PYTHON_CALLBACKS
       "Build the embedded-interpreter callback plugin (PauliRot graph-decomposition)" ON)
if (NOT CATALYST_ENABLE_PYTHON_CALLBACKS)
    return()
endif()

execute_process(
    COMMAND python3 -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(nanobind CONFIG REQUIRED)
nanobind_compile_options(nanobind-static STABLE_ABI)

# Discover the active Python version
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Mirror setup.py
if (Python3_VERSION VERSION_GREATER_EQUAL 3.12)
    set(_QPC_PYTHON_COMPONENT "Development.SABIModule")
    set(_QPC_PYTHON_TARGET Python3::SABIModule)
    set(_QPC_USE_LIMITED_API ON)
    set(NANOBIND_TARGET "nanobind-static-abi3")
    message(STATUS
        "QuantumPythonCallbacks: Python ${Python3_VERSION} -> Stable ABI build "
        "(Py_LIMITED_API=0x030C0000, links Python3::SABIModule)")
else()
    set(_QPC_PYTHON_COMPONENT "Development.Module")
    set(_QPC_PYTHON_TARGET Python3::Module)
    set(_QPC_USE_LIMITED_API OFF)
    set(NANOBIND_TARGET "nanobind-static")
    message(STATUS
        "QuantumPythonCallbacks: Python ${Python3_VERSION} -> non-SABI build "
        "(plugin .so is tied to this Python's ABI)")
endif()

# Second find_package call accumulates the component selected above.
find_package(Python3 REQUIRED COMPONENTS Interpreter ${_QPC_PYTHON_COMPONENT})

if (NOT TARGET ${NANOBIND_TARGET})
    nanobind_build_library(${NANOBIND_TARGET})

    target_compile_options(${NANOBIND_TARGET} PUBLIC
        -Wno-cast-qual
        -Wno-covered-switch-default
    )
endif()

add_mlir_library(QuantumPythonCallbacks MODULE
    PythonFunction.cpp
    PythonDriverUtils.cpp
    LINK_LIBS PRIVATE
        ${NANOBIND_TARGET}
        ${_QPC_PYTHON_TARGET}
        MLIRIR
        MLIRFuncDialect
        MLIRParser
        quantum-transforms  # for DecompCallback
)

target_compile_definitions(QuantumPythonCallbacks PRIVATE NB_DOMAIN=${MLIR_BINDINGS_PYTHON_NB_DOMAIN})

if (_QPC_USE_LIMITED_API)
    target_compile_definitions(QuantumPythonCallbacks PRIVATE
        Py_LIMITED_API=0x030C0000)
endif()

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
    PROPERTIES COMPILE_FLAGS "-Wno-cast-qual -fexceptions -frtti -Wno-covered-switch-default")

# Install next to the catalyst binary
install(TARGETS QuantumPythonCallbacks
    LIBRARY DESTINATION lib)
