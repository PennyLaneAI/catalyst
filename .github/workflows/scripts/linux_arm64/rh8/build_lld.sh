#!/usr/bin/env bash

set -e -x
cd /catalyst

# Process args
export GCC_VERSION=$1
export PYTHON_VERSION=$2
export PYTHON_SUBVERSION=$3
export PYTHON_PACKAGE=$4

# Install system dependencies
dnf update -y
dnf install -y libzstd-devel gcc-toolset-${GCC_VERSION}
dnf install -y ${PYTHON_PACKAGE} ${PYTHON_PACKAGE}-devel
dnf clean all -y

# Make GCC the default compiler
source /opt/rh/gcc-toolset-${GCC_VERSION}/enable -y
export C_COMPILER=/opt/rh/gcc-toolset-${GCC_VERSION}/root/usr/bin/gcc
export CXX_COMPILER=/opt/rh/gcc-toolset-${GCC_VERSION}/root/usr/bin/g++

# Set the right Python interpreter
rm -rf /usr/bin/python3
ln -s /opt/_internal/cpython-${PYTHON_VERSION}.${PYTHON_SUBVERSION}/bin/python3 /usr/bin/python3
export PYTHON=/usr/bin/python3

# Add Python and GCC to the PATH env var
export PATH=/opt/_internal/cpython-${PYTHON_VERSION}.${PYTHON_SUBVERSION}/bin:/opt/rh/gcc-toolset-${GCC_VERSION}/root/usr/bin:$PATH

# Install python dependencies
/usr/bin/python3 -m pip install pennylane pybind11 PyYAML cmake ninja

cmake -S /catalyst/mlir/llvm-project/llvm -B llvm-build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_TARGETS_TO_BUILD="host" \
      -DLLVM_ENABLE_PROJECTS="lld"

cmake --build /catalyst/llvm-build --target lld
