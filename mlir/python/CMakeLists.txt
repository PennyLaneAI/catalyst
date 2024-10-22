include(AddMLIRPython)
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_quantum.")


################################################################################
# Python and nanobind
################################################################################

# nanobind currently requires (as of 2.2.0) that we use the `Python` package, NOT the `Python3` package
# We still require at least Python version 3
find_package(Python 3
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Check if we have overridden the Python3_EXECUTABLE variable
# If so, set Python_EXECUTABLE to the same value for nanobind compatibility
if(DEFINED Python3_EXECUTABLE)
  set(Python_EXECUTABLE ${Python3_EXECUTABLE})
  message(STATUS "Overriding Python_EXECUTABLE to value of Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
endif()

# nanobind suggests including these lines to configure CMake to perform an optimized release build
# by default unless another build type is specified. Without this addition, binding code may run
# slowly and produce large binaries.
# See https://nanobind.readthedocs.io/en/latest/building.html#preliminaries
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)

find_package(nanobind CONFIG REQUIRED)


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

declare_mlir_python_extension(QuantumPythonSources.Extension
  MODULE_NAME _quantumDialects
  ADD_TO_PARENT QuantumPythonSources
  SOURCES
    QuantumExtension.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
    MLIRIR
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


################################################################################
# Compiler Driver Nanobind Module
################################################################################

# Source files for the Python `compiler_driver` module
set(COMPILER_DRIVER_SRC_FILES
  PyCompilerDriver.cpp
)

# Create the `compiler_driver` module
# Target the stable ABI for Python 3.12+, which reduces the number of binary wheels that must be
# built (`STABLE_ABI` does nothing on older Python versions).
nanobind_add_module(compiler_driver STABLE_ABI ${COMPILER_DRIVER_SRC_FILES})

# Output library name will be compiler_driver.so
set_target_properties(compiler_driver
  PROPERTIES
  PREFIX ""     # Removes 'lib' prefix
  SUFFIX ".so"  # Replaces '.abi3.so' suffix (for Stable ABI) with '.so'
  LIBRARY_OUTPUT_DIRECTORY ${MLIR_BINARY_DIR}/python_packages/quantum/mlir_quantum
)

target_link_libraries(compiler_driver PRIVATE CatalystCompilerDriver)


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
