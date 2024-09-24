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
/usr/bin/python3 -m pip install pennylane pybind11 PyYAML cmake ninja

export LLVM_ROOT=mlir/llvm-project
export LLVM_MODULE_PATCH_FILE=mlir/llvm-project/patches/moduleOp-bufferization.patch
export LLVM_FUNC_CALL_PATCH_FILE=mlir/llvm-project/patches/callOp-bufferization.patch
if patch --dry-run -p1 -N --directory=$(LLVM_ROOT) < $(LLVM_MODULE_PATCH_FILE) > /dev/null 2>&1; then patch -p1 --directory=$(LLVM_ROOT) < $(LLVM_MODULE_PATCH_FILE); fi
if patch --dry-run -p1 -N --directory=$(LLVM_ROOT) < $(LLVM_FUNC_CALL_PATCH_FILE) > /dev/null 2>&1; then patch -p1 --directory=$(LLVM_ROOT) < $(LLVM_FUNC_CALL_PATCH_FILE); fi

# Build LLVM
cmake -S /catalyst/mlir/llvm-project/llvm -B /catalyst/llvm-build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_ENABLE_ZLIB=OFF \
    -DLLVM_ENABLE_ZSTD=FORCE_ON \
    -DLLVM_ENABLE_LLD=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=/usr/bin/python3 \
    -DPython3_NumPy_INCLUDE_DIRS=/opt/_internal/cpython-${PYTHON_VERSION}.${PYTHON_SUBVERSION}/lib/python${PYTHON_VERSION}/site-packages/numpy/core/include \
    -DCMAKE_CXX_VISIBILITY_PRESET=protected

LIT_FILTER_OUT="Bytecode|tosa-to-tensor" cmake --build /catalyst/llvm-build --target check-mlir llvm-symbolizer
