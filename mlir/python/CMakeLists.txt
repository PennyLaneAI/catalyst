include(AddMLIRPython)
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_quantum.")

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

set(COMPILER_DRIVER_SRC_FILES
  PyCompilerDriver.cpp
)

# Create the Python `compiler_driver` module
# Target the stable ABI for Python 3.12+, which reduces the number of binary wheels that must be
# built (`STABLE_ABI` does nothing on older Python versions).
nanobind_add_module(compiler_driver STABLE_ABI ${COMPILER_DRIVER_SRC_FILES})

set_target_properties(compiler_driver PROPERTIES PREFIX "")

set_target_properties(compiler_driver PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${MLIR_BINARY_DIR}/python_packages/quantum/mlir_quantum)

target_link_libraries(compiler_driver PRIVATE CatalystCompilerDriver)
