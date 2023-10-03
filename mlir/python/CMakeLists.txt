include(AddMLIRPython)
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_quantum.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(QuantumPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT QuantumPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/"
  TD_FILE dialects/QuantumOps.td
  SOURCES
  dialects/quantum.py
  DIALECT_NAME quantum)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT QuantumPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/"
  TD_FILE dialects/GradientOps.td
  SOURCES
  dialects/gradient.py
  DIALECT_NAME gradient)

declare_mlir_python_extension(QuantumPythonSources.Extension
  MODULE_NAME _quantumDialects
  ADD_TO_PARENT QuantumPythonSources
  SOURCES
    QuantumExtension.cpp
  EMBED_CAPI_LINK_LIBS
    QuantumCAPI
)

################################################################################
# Common CAPI
################################################################################


add_mlir_python_common_capi_library(QuantumPythonCAPI
  INSTALL_COMPONENT QuantumPythonModules
  INSTALL_DESTINATION python_packages/quantum/mlir_quantum/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/quantum/mlir_quantum/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    QuantumPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

add_library(compiler_driver MODULE PyCompilerDriver.cpp)
set_target_properties(compiler_driver PROPERTIES PREFIX "")

target_link_libraries(compiler_driver PRIVATE pybind11::headers pybind11::module CatalystCompilerDriver)
if("${LLVM_USE_SANITIZER}" STREQUAL "Address")
    if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        add_compile_options(-fsanitize=address)
        add_link_options(-fsanitize=address -static-libasan)
    elseif(CMAKE_C_COMPILER_ID STREQUAL "Clang")
        target_link_libraries(compiler_driver INTERFACE -fno-omit-frame-pointer -fsanitize=address)
    endif()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
ENDIF()

set_target_properties(compiler_driver PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${MLIR_BINARY_DIR}/python_packages/quantum/mlir_quantum)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(QuantumPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/quantum/mlir_quantum/"
  INSTALL_PREFIX "python_packages/quantum/mlir_quantum/"
  DECLARED_SOURCES
    QuantumPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    QuantumPythonCAPI
  )
