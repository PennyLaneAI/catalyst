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

# Add LLVM, Python and GCC to the PATH env var
export PATH=/catalyst/llvm-build/bin:/opt/_internal/cpython-${PYTHON_VERSION}.${PYTHON_SUBVERSION}/bin:/opt/rh/gcc-toolset-${GCC_VERSION}/root/usr/bin:/catalyst/llvm-build/bin:$PATH

# Install python dependencies
/usr/bin/python3 -m pip install pennylane pybind11 PyYAML cmake ninja

# Patch MHLO for correct linking
sed -i -e 's/LINK_LIBS PUBLIC/LINK_LIBS PUBLIC MLIRDeallocationUtils/g' mlir/mlir-hlo/deallocation/transforms/CMakeLists.txt

export TARGET_FILE=mlir/mlir-hlo/mhlo/transforms/CMakeLists.txt
export PATCH_FILE=mlir/patches/mhlo-Add-PassesIncGen-in-transforms-CMakeList.patch
if patch --dry-run -p1 -N $TARGET_FILE $PATCH_FILE > /dev/null 2>&1; then patch -p1 $TARGET_FILE $PATCH_FILE; fi
# TODO: Remove these patches after upgrading Jax (potentailly for 0.4.34 or higher).
if patch --dry-run -p1 -N --directory=/catalyst/mlir/mlir-hlo < /catalyst/mlir/patches/FunctionOpInterface-mhlo.patch > /dev/null 2>&1; then patch -p1 --directory=/catalyst/mlir/mlir-hlo < /catalyst/mlir/patches/FunctionOpInterface-mhlo.patch; fi

# Build MHLO
cmake -S /catalyst/mlir/mlir-hlo -B /catalyst/mhlo-build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_DIR=/catalyst/llvm-build/lib/cmake/mlir \
    -DPython3_EXECUTABLE=/usr/bin/python3 \
    -DLLVM_ENABLE_LLD=ON \
    -DLLVM_ENABLE_ZLIB=OFF \
    -DLLVM_ENABLE_ZSTD=FORCE_ON \
    -DCMAKE_CXX_VISIBILITY_PRESET=protected

LIT_FILTER_OUT="chlo_legalize_to_mhlo" cmake --build /catalyst/mhlo-build --target check-mlir-hlo
