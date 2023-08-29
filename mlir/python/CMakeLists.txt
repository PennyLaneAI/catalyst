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

declare_mlir_python_extension(CatalystPythonDriver
  MODULE_NAME _catalystDriver
  ADD_TO_PARENT QuantumPythonSources
  SOURCES
    PyCompilerDriver.cpp
  PRIVATE_LINK_LIBS
    QuantumCAPI
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(QuantumPythonCAPI
  INSTALL_COMPONENT QuantumPythonModules
  INSTALL_DESTINATION python_packages/quantum/mlir_quantum/_quantum_capi_
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/quantum/mlir_quantum/_quantum_capi"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    QuantumPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

# Function: add_mlir_python_modules
# Adds python modules to a project, building them from a list of declared
# source groupings (see declare_mlir_python_sources and
# declare_mlir_python_extension). One of these must be called for each
# packaging root in use.
# Arguments:
#   ROOT_PREFIX: The directory in the build tree to emit sources. This will
#     typically be something like ${MY_BINARY_DIR}/python_packages/foobar
#     for non-relocatable modules or a deeper directory tree for relocatable.
#   INSTALL_PREFIX: Prefix into the install tree for installing the package.
#     Typically mirrors the path above but without an absolute path.
#   DECLARED_SOURCES: List of declared source groups to include. The entire
#     DAG of source modules is included.
#   COMMON_CAPI_LINK_LIBS: List of dylibs (typically one) to make every
#     extension depend on (see mlir_python_add_common_capi_library).
function(add_catalyst_python_modules name)
  cmake_parse_arguments(ARG
    ""
    "ROOT_PREFIX;INSTALL_PREFIX;COMMON_CAPI_LINK_LIBS"
    "DECLARED_SOURCES;INSTALL_DIR;OUTPUT_DIRECTORY"
    ${ARGN})
  # Helper to process an individual target.
  function(_process_target modules_target sources_target)
    get_target_property(_source_type ${sources_target} mlir_python_SOURCES_TYPE)

    if(_source_type STREQUAL "pure")
      # Pure python sources to link into the tree.
      set(_pure_sources_target "${modules_target}.sources.${sources_target}")
      add_mlir_python_sources_target(${_pure_sources_target}
        INSTALL_COMPONENT ${modules_target}
        INSTALL_DIR ${ARG_INSTALL_PREFIX}
        OUTPUT_DIRECTORY ${ARG_ROOT_PREFIX}
        SOURCES_TARGETS ${sources_target}
      )
      add_dependencies(${modules_target} ${_pure_sources_target})
    elseif(_source_type STREQUAL "extension")
      # Native CPP extension.
      get_target_property(_module_name ${sources_target} mlir_python_EXTENSION_MODULE_NAME)
      # Transform relative source to based on root dir.
      set(_extension_target "${modules_target}.extension.${_module_name}.dso")
      execute_process(COMMAND ${CMAKE_COMMAND} -E echo "HERE" ${_extension_target})
      execute_process(COMMAND ${CMAKE_COMMAND} -E echo "HERE" ${sources_target})
      if(${sources_target} STREQUAL CatalystPythonDriver)
        add_mlir_python_extension(${_extension_target} "${_module_name}"
          INSTALL_COMPONENT ${modules_target}
          INSTALL_DIR "${ARG_INSTALL_PREFIX}/compiler_driver"
          OUTPUT_DIRECTORY "${ARG_ROOT_PREFIX}/compiler_driver"
          LINK_LIBS PRIVATE
            ${sources_target}
            ${ARG_COMMON_CAPI_LINK_LIBS}
        )
        mlir_python_setup_extension_rpath(${_extension_target} RELATIVE_INSTALL_ROOT "../_mlir_libs")
        mlir_python_setup_extension_rpath(${_extension_target} RELATIVE_INSTALL_ROOT "../_quantum_capi")
      else()
        add_mlir_python_extension(${_extension_target} "${_module_name}"
          INSTALL_COMPONENT ${modules_target}
          INSTALL_DIR "${ARG_INSTALL_PREFIX}/_mlir_libs"
          OUTPUT_DIRECTORY "${ARG_ROOT_PREFIX}/_mlir_libs"
          LINK_LIBS PRIVATE
            ${sources_target}
            ${ARG_COMMON_CAPI_LINK_LIBS}
        )
      endif()
      add_dependencies(${modules_target} ${_extension_target})
      mlir_python_setup_extension_rpath(${_extension_target})
    else()
      message(SEND_ERROR "Unrecognized source type '${_source_type}' for python source target ${sources_target}")
      return()
    endif()
  endfunction()

  # Build the modules target.
  add_custom_target(${name} ALL)
  _flatten_mlir_python_targets(_flat_targets ${ARG_DECLARED_SOURCES})
  foreach(sources_target ${_flat_targets})
    _process_target(${name} ${sources_target})
  endforeach()

  # Create an install target.
  if(NOT LLVM_ENABLE_IDE)
    add_llvm_install_targets(
      install-${name}
      DEPENDS ${name}
      COMPONENT ${name})
  endif()
endfunction()

################################################################################
# Instantiation of all Python modules
################################################################################

add_catalyst_python_modules(QuantumPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/quantum/mlir_quantum/"
  INSTALL_PREFIX "python_packages/quantum/mlir_quantum/"
  DECLARED_SOURCES
    QuantumPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    QuantumPythonCAPI
  )

