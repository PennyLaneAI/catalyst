set(PYTHON_VERSION_TO_FIND "3" CACHE STRING "Python version to find and use")

find_package(Python ${PYTHON_VERSION_TO_FIND} EXACT
    REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)

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

# Get the NumPy include directory
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Source file list for `wrapper` module
set(REGISTRY_SRC_FILES
    Registry.cpp
)

# Create the Python `catalyst_callback_registry` module
# Target the stable ABI for Python 3.12+, which reduces the number of binary wheels that must be
# built (`STABLE_ABI` does nothing on older Python versions).
nanobind_add_module(catalyst_callback_registry STABLE_ABI ${REGISTRY_SRC_FILES})

# Use a consistant suffix ".so" rather than, e.g. ".abi3.so" (when using the Stable ABI) or
# ".cpython-3xx-darwin.so". This ensures we can locate it when calling `dlopen(LIBREGISTRY)` in
# runtime/lib/capi/RuntimeCAPI.cpp.
set_target_properties(catalyst_callback_registry PROPERTIES SUFFIX ".so")

target_include_directories(catalyst_callback_registry PUBLIC ${runtime_includes})
target_compile_definitions(catalyst_qir_qis_obj PUBLIC -DLIBREGISTRY=\"$<TARGET_FILE_NAME:catalyst_callback_registry>\")

if(NOT APPLE)
    set_property(TARGET catalyst_qir_qis_obj APPEND PROPERTY BUILD_RPATH $ORIGIN)
else()
    set_property(TARGET catalyst_qir_qis_obj APPEND PROPERTY BUILD_RPATH @loader_path)
endif()
