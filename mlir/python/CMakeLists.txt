include(AddMLIRPython)
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_quantum.")


################################################################################
# Declare Dialect Sources
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

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT QuantumPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/"
  TD_FILE dialects/MBQCOps.td
  SOURCES
  dialects/mbqc.py
  DIALECT_NAME mbqc)


################################################################################
# Build Python Bindings (without C extension)
################################################################################

add_mlir_python_modules(QuantumPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/quantum/mlir_quantum/"
  INSTALL_PREFIX "python_packages/quantum/mlir_quantum/"
  DECLARED_SOURCES
    QuantumPythonSources
    MLIRPythonSources.Dialects         # we also use upstream dialect bindings
    MLIRPythonSources.Core.Python      # common sources, like _ods_common.py
    MLIRPythonSources.ExecutionEngine  # for the mlir_quantum.runtime module
  )
