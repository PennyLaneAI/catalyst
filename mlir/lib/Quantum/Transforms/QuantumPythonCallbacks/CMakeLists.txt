option(CATALYST_ENABLE_PYTHON_CALLBACKS
       "Build the embedded-interpreter callback plugin (PauliRot graph-decomposition)" ON)
if (NOT CATALYST_ENABLE_PYTHON_CALLBACKS)
    return()
endif()

# nanobind suggests including these lines to configure CMake to perform an optimized release build
# by default unless another build type is specified. Without this addition, binding code may run
# slowly and produce large binaries.
# See https://nanobind.readthedocs.io/en/latest/building.html#preliminaries
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


# Discover the active Python version
find_package(Python REQUIRED
    COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule
)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE nanobind_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(nanobind CONFIG REQUIRED)

# Ignore all warnings from nanobind library sources (headers are not enough).
add_compile_options("-w")

nanobind_add_module(QuantumPythonCallbacks
    STABLE_ABI
    NB_SUPPRESS_WARNINGS
    PythonFunction.cpp
    PythonDriverUtils.cpp
)

# TODO: move MLIR parsing back to quantum-transforms, let callbacks only handle calling python
target_link_libraries(QuantumPythonCallbacks PRIVATE
    MLIRIR
    MLIRFuncDialect
    MLIRParser
    quantum-transforms  # for DecompCallback
)

target_compile_definitions(QuantumPythonCallbacks PRIVATE
    NB_DOMAIN=${MLIR_BINDINGS_PYTHON_NB_DOMAIN}
)

target_compile_options(QuantumPythonCallbacks PRIVATE
    -Wno-cast-qual
    -Wno-covered-switch-default
)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(QuantumPythonCallbacks PRIVATE "-Wno-unknown-warning-option")
endif()

if (APPLE)
    target_link_options(QuantumPythonCallbacks PRIVATE "LINKER:-undefined,dynamic_lookup")
else()
    target_link_options(QuantumPythonCallbacks PRIVATE "LINKER:--allow-shlib-undefined")
endif()

target_include_directories(QuantumPythonCallbacks PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(QuantumPythonCallbacks PROPERTIES
    PREFIX "lib"
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Install next to the catalyst binary
install(TARGETS QuantumPythonCallbacks
    LIBRARY DESTINATION lib)

