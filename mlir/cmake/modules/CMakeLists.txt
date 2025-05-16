# Generate a list of CMake library targets so that other CMake projects can
# link against them. LLVM calls its version of this file LLVMExports.cmake, but
# the usual CMake convention is ${Project}Targets.cmake.
# This configuration has been partly adapted from MLIR project's
# (llvm-project:mlir/cmake/modules) and from StableHLO's project
# https://github.com/llvm/llvm-project/blob/main/mlir/cmake/modules/CMakeLists.txt

##############################################################################
# The following file                                                         #
# was taken from the openxla/xla project:https://github.com/openxla/xla      #
# Small alteration were made for Catalyst                                    #
# The XLA project has the following license: Apache License v2.0             #
#                                                                            #
# https://github.com/openxla/xla/blob/main/xla/mlir_hlo/                     #
#            cmake/modules/CMakeLists.txt                                    #
##############################################################################

include(LLVMDistributionSupport)

set(CATALYST_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/catalyst)
set(catalyst_cmake_builddir "${CMAKE_BINARY_DIR}/${CATALYST_INSTALL_PACKAGE_DIR}")

# Keep this in sync with llvm/cmake/CMakeLists.txt!
set(LLVM_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm)
set(llvm_cmake_builddir "${LLVM_BINARY_DIR}/${LLVM_INSTALL_PACKAGE_DIR}")

# Since we just use MLIR's cmake functions (add_mlir_library, etc.),
# MLIR_EXPORTS would have the necessary targets to export.
get_property(MLIR_EXPORTS GLOBAL PROPERTY MLIR_EXPORTS)
# JSON for Modern C++ (nlohmann/json) https://github.com/nlohmann/json
# tomlplusplus https://github.com/marzer/tomlplusplus
# libstim https://github.com/quantumlib/Stim
# are used in Catalyst acknowledged here https://github.com/PennyLaneAI/catalyst?tab=readme-ov-file#acknowledgements
# nlohmann/json and tomlplusplus both have MIT licenses.
# libstim is Apache version 2.0
# QECUtils needs to be added manually since it is not in MLIR_EXPORTS
# todo: the following two lines removed nlohmann_json as export targets, should probably be done for the other extra targets as well
include(CMakeFindDependencyMacro)
find_dependency(nlohmann_json)
export(TARGETS ${MLIR_EXPORTS} QECUtils libstim tomlplusplus_tomlplusplus FILE ${catalyst_cmake_builddir}/CatalystTargets.cmake)

# Generate MlirConfig.cmake for the build tree.
set(CATALYST_CONFIG_CMAKE_DIR "${catalyst_cmake_builddir}")
set(CATALYST_CONFIG_LLVM_CMAKE_DIR "${llvm_cmake_builddir}")
set(CATALYST_CONFIG_INCLUDE_EXPORTS "include(\"\${CATALYST_CMAKE_DIR}/CatalystTargets.cmake\")")
set(CATALYST_CONFIG_INCLUDE_DIRS
  "${CATALYST_SOURCE_DIR}"
  "${CATALYST_MAIN_INCLUDE_DIR}"
  "${CATALYST_GEN_INCLUDE_DIR}"
)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/CatalystConfig.cmake.in
  ${catalyst_cmake_builddir}/CatalystConfig.cmake
  @ONLY)

# Unset the variables now that the file is configured.
set(CATALYST_CONFIG_CMAKE_DIR)
set(CATALYST_CONFIG_LLVM_CMAKE_DIR)
set(CATALYST_CONFIG_INCLUDE_DIRS)
