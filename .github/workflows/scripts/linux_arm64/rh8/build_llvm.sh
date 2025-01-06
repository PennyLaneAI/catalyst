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
if [ "$PYTHON_VERSION" != "3.10" ]; then
    dnf install -y ${PYTHON_PACKAGE} ${PYTHON_PACKAGE}-devel
fi
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
export PATH=/opt/_internal/cpython-${PYTHON_VERSION}.${PYTHON_SUBVERSION}/bin:/opt/rh/gcc-toolset-${GCC_VERSION}/root/usr/bin:/catalyst/llvm-build/bin:$PATH

# Install python dependencies
/usr/bin/python3 -m pip install pybind11 PyYAML cmake ninja

# Build LLVM
export LLVM_BUILD_DIR="/catalyst/llvm-build"
export LLVM_PROJECTS="lld;mlir"
export LLVM_TARGETS="lld check-mlir"
export ENABLE_LLD=OFF
export ENABLE_ZLIB=FORCE_ON
make llvm
