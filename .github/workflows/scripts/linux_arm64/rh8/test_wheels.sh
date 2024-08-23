#!/usr/bin/env bash

set -e -x
cd /catalyst

# Process args
export GCC_VERSION=$1
export PYTHON_VERSION=$2
export PYTHON_SUBVERSION=$3
export PYTHON_PACKAGE=$4

# Install system dependencies (gcc gives access to c99, which is needed by some tests)
dnf update -y 
dnf install -y libzstd-devel gcc-toolset-${GCC_VERSION} gcc
if [ "$PYTHON_VERSION" != "3.10" ]; then
    dnf install -y ${PYTHON_PACKAGE} ${PYTHON_PACKAGE}-devel
else
    # Patch libpython version since we do not install python-devel for 3.10.
    dnf groupinstall "Development Tools" -y
    dnf install openssl-devel bzip2-devel libffi-devel -y
    dnf install wget -y
    cd /tmp
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}.${PYTHON_SUBVERSION}/Python-${PYTHON_VERSION}.${PYTHON_SUBVERSION}.tgz
    tar xzf Python-${PYTHON_VERSION}.${PYTHON_SUBVERSION}.tgz
    cd Python-${PYTHON_VERSION}.${PYTHON_SUBVERSION}
    ./configure --enable-optimizations --enable-shared
    make
    make altinstall
    cd /catalyst
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

# Set llvm-symbolizer
ls -la /catalyst/llvm-build/bin/llvm-symbolizer
export LLVM_SYMBOLIZER_PATH=/catalyst/llvm-build/bin/llvm-symbolizer

# Add LLVM, Python and GCC to the PATH env var
export PATH=/catalyst/llvm-build/bin:/opt/_internal/cpython-${PYTHON_VERSION}.${PYTHON_SUBVERSION}/bin:/opt/rh/gcc-toolset-${GCC_VERSION}/root/usr/bin:$PATH

# Install python dependencies
/usr/bin/python3 -m pip install pennylane pybind11 PyYAML cmake ninja pytest pytest-xdist pytest-mock autoray PennyLane-Lightning-Kokkos 'amazon-braket-pennylane-plugin>1.27.1'
/usr/bin/python3 -m pip install oqc-qcaas-client

# Install Catalyst wheel
/usr/bin/python3 -m pip install /catalyst/dist/*.whl --extra-index-url https://test.pypi.org/simple

# Test Catalyst
/usr/bin/python3 -m pytest -v /catalyst/frontend/test/pytest -n auto
/usr/bin/python3 -m pytest -v /catalyst/frontend/test/pytest --backend="lightning.kokkos" -n auto
/usr/bin/python3 -m pytest /catalyst/frontend/test/async_tests
/usr/bin/python3 -m pytest -v /catalyst/frontend/test/pytest --runbraket=LOCAL -n auto
/usr/bin/python3 -m pytest /catalyst/frontend/test/test_oqc/oqc -n auto
