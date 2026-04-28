#######################################
## default_pipelines nanobind module ##
#######################################

# nanobind suggests including these lines to configure CMake to perform an optimized release build
# by default unless another build type is specified. Without this addition, binding code may run
# slowly and produce large binaries.
# See https://nanobind.readthedocs.io/en/latest/building.html#preliminaries
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(Python REQUIRED
    COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule
)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE nanobind_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(nanobind CONFIG REQUIRED)

set(DEFAULT_PIPELINES_SRC_FILES
    DefaultPipelines.cpp
)

# Ignore all warnings from nanobind library sources (headers are not enough).
add_compile_options("-w")

# Target the stable ABI for Python 3.12+,
# which reduces the number of binary wheels that must be built
# (`STABLE_ABI` does nothing on older Python versions).
nanobind_add_module(default_pipelines STABLE_ABI
    ${DEFAULT_PIPELINES_SRC_FILES}
    # -Wcast-qual: llvm configures more warnings than we get from -Wall in the runtime for example,
    # so we need this additional flag here to suppress nanobind header warnings.
    NB_SUPPRESS_WARNINGS
)

# Easier discovery later (similar to the runtime nanobind module for callback registration).
set_target_properties(default_pipelines PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/lib"
)
