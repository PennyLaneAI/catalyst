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

#declare_mlir_python_extension(CatalystPythonDriver
#  MODULE_NAME _catalystDriver
#  ADD_TO_PARENT QuantumPythonSources
#  SOURCES
#    PyCompilerDriver.cpp
#  PRIVATE_LINK_LIBS
#    QuantumCAPI
#)

################################################################################
# Common CAPI
################################################################################


add_mlir_python_common_capi_library(QuantumPythonCAPI
  INSTALL_COMPONENT QuantumPythonModules
  INSTALL_DESTINATION python_packages/quantum/mlir_quantum/compiler_driver
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/quantum/mlir_quantum/compiler_driver"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    QuantumPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    MLIROptLib
    MLIRCatalyst
    MLIRCatalystTransforms
    MLIRQuantum
    quantum-transforms
    MLIRGradient
    gradient-transforms

    CatalystCompilerDriver
    QuantumPythonCAPI

    MhloRegisterDialects
    StablehloRegister
    ${ALL_MHLO_PASSES}
)

add_library(CatalystPythonDriver MODULE PyCompilerDriver.cpp)

target_link_libraries(CatalystPythonDriver PRIVATE pybind11::headers pybind11::module ${LIBS})

set_target_properties(CatalystPythonDriver PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${MLIR_BINARY_DIR}/python_packages/quantum/mlir_quantum/compiler_driver)
#set_target_properties(CatalystPythonDriver PROPERTIES BUILD_WITH_INSTALL_RPATH OFF BUILD_RPATH $ORIGIN INSTALL_RPATH $ORIGIN )
#set_target_properties(CatalystPythonDriver PROPERTIES BUILD_WITH_INSTALL_RPATH OFF BUILD_RPATH ${MLIR_BINARY_DIR}/python_packages/quantum/mlir_quantum/compiler_driver INSTALL_RPATH ${MLIR_BINARY_DIR}/python_packages/quantum/mlir_quantum/compiler_driver)
#set_target_properties(CatalystPythonDriver PROPERTIES BUILD_WITH_INSTALL_RPATH OFF BUILD_RPATH ${MLIR_BINARY_DIR}/python_packages/quantum/mlir_quantum/compiler_driver INSTALL_RPATH ${MLIR_BINARY_DIR}/python_packages/quantum/mlir_quantum/_mlir_libs)

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

add_dependencies(QuantumPythonModules CatalystPythonDriver)
