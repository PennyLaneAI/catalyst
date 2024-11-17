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

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT QuantumPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/"
  TD_FILE dialects/CatalystOps.td
  SOURCES
  dialects/catalyst.py
  DIALECT_NAME catalyst)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT QuantumPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/"
  TD_FILE dialects/MitigationOps.td
  SOURCES
  dialects/mitigation.py
  DIALECT_NAME mitigation)


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
