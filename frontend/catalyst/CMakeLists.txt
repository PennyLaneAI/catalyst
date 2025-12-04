#######################################
## default_pipelines nanobind module ##
#######################################

set(DEFAULT_PIPELINES_SRC_FILES
    default_pipelines.cpp
)

# Target the stable ABI for Python 3.12+,
# which reduces the number of binary wheels that must be built
# (`STABLE_ABI` does nothing on older Python versions).
nanobind_add_module(default_pipelines STABLE_ABI
    ${DEFAULT_PIPELINES_SRC_FILES}
)

# This is necessary for compatibility with setuptools build extensions
set_target_properties(default_pipelines PROPERTIES
    SUFFIX ".so"
)

set(CATALYST_ROOT_DIR "${PROJECT_SOURCE_DIR}/..")
set(CATALYST_MLIR_INCLUDE_DIR "${CATALYST_ROOT_DIR}/mlir/include")

target_include_directories(default_pipelines PRIVATE
    ${CATALYST_MLIR_INCLUDE_DIR}
)

###################
### utils subdir ##
###################

add_subdirectory(utils)
