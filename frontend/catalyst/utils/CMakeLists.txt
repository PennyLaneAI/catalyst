cmake_minimum_required(VERSION 3.26)

project(catalyst_frontend)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#############################
## wrapper nanobind module ##
#############################

# Source file list for `wrapper` module
set(WRAPPER_SRC_FILES
    wrapper.cpp
)

# Create the Python `wrapper` module
# Target the stable ABI for Python 3.12+, which reduces the number of binary wheels that must be
# built (`STABLE_ABI` does nothing on older Python versions).
nanobind_add_module(wrapper STABLE_ABI ${WRAPPER_SRC_FILES})

# Add the NumPy include directory to the library's include paths
target_include_directories(wrapper PRIVATE ${Python_NumPy_INCLUDE_DIRS})

# Use suffix ".so" rather than ".abi3.so" for library file using Stable ABI
# This is necessary for compatibility with setuptools build extensions
set_target_properties(wrapper PROPERTIES SUFFIX ".so")
