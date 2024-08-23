#!/usr/bin/env bash

set -e -x
cd /catalyst

# Avoid git safe directory error
git config --global --add safe.directory '*'

# Process args
export GCC_VERSION=$1
export PYTHON_VERSION=$2
export PYTHON_SUBVERSION=$3
export PYTHON_PACKAGE=$4
export PYTHON_ALTERNATIVE_VERSION=$5

# Install system dependencies
dnf update -y 
dnf install -y libzstd-devel gcc-toolset-${GCC_VERSION}
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
export PATH=/catalyst/llvm-build/bin:/opt/_internal/cpython-${PYTHON_VERSION}.${PYTHON_SUBVERSION}/bin:/opt/rh/gcc-toolset-${GCC_VERSION}/root/usr/bin:$PATH

# Install python dependencies
/usr/bin/python3 -m pip install pennylane pybind11 PyYAML cmake ninja delocate 'amazon-braket-pennylane-plugin>1.27.1'

# Build Catalyst runtime
cmake -S runtime -B runtime-build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=runtime-build/lib \
    -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    -DPYTHON_INCLUDE_DIR=/opt/_internal/cpython-${PYTHON_VERSION}.${PYTHON_SUBVERSION}/include/python${PYTHON_VERSION} \
    -DPYTHON_LIBRARY=/opt/_internal/cpython-${PYTHON_VERSION}.${PYTHON_SUBVERSION}/lib \
    -Dpybind11_DIR=/opt/_internal/cpython-${PYTHON_VERSION}.${PYTHON_SUBVERSION}/lib/python${PYTHON_VERSION}/site-packages/pybind11/share/cmake/pybind11 \
    -DLIGHTNING_GIT_TAG=c6b86a5 \
    -DENABLE_LAPACK=OFF \
    -DENABLE_WARNINGS=OFF \
    -DENABLE_OPENQASM=ON \
    -DENABLE_OPENMP=OFF \
    -DLQ_ENABLE_KERNEL_OMP=OFF
cmake --build runtime-build --target rt_capi rtd_lightning rtd_openqasm rtd_dummy

# Build OQC
export OQC_BUILD_DIR="/catalyst/oqc-build"
export RT_BUILD_DIR="/catalyst/runtime-build"
export USE_ALTERNATIVE_CATALYST_PYTHON_INTERPRETER=ON
make oqc

# Build Catalyst dialects
cmake -S mlir -B quantum-build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DQUANTUM_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=/usr/bin/python3 \
    -DPython3_NumPy_INCLUDE_DIRS=/opt/_internal/cpython-${PYTHON_VERSION}.${PYTHON_SUBVERSION}/lib/python${PYTHON_VERSION}/site-packages/numpy/core/include \
    -DMLIR_DIR=/catalyst/llvm-build/lib/cmake/mlir \
    -DMHLO_DIR=/catalyst/mhlo-build/lib/cmake/mlir-hlo \
    -DMHLO_BINARY_DIR=/catalyst/mhlo-build/bin \
    -DEnzyme_DIR=/catalyst/enzyme-build \
    -DENZYME_SRC_DIR=/catalyst/mlir/Enzyme \
    -DLLVM_ENABLE_ZLIB=OFF \
    -DLLVM_ENABLE_ZSTD=FORCE_ON \
    -DLLVM_ENABLE_LLD=ON \
    -DLLVM_DIR=/catalyst/llvm-build/lib/cmake/llvm
cmake --build quantum-build --target check-dialects compiler_driver

# Copy files needed for the wheel where they are expected
cp /catalyst/runtime-build/lib/*/*/*/*/librtd* /catalyst/runtime-build/lib
cp /catalyst/runtime-build/lib/registry/runtime-build/lib/catalyst_callback_registry.cpython-${PYTHON_ALTERNATIVE_VERSION}-aarch64-linux-gnu.so /catalyst/runtime-build/lib
cp /catalyst/runtime-build/lib/capi/runtime-build/lib/librt_capi.so /catalyst/runtime-build/lib/

# Build wheels
export LLVM_BUILD_DIR=/catalyst/llvm-build
export MHLO_BUILD_DIR=/catalyst/mhlo-build
export DIALECTS_BUILD_DIR=/catalyst/quantum-build
export RT_BUILD_DIR=/catalyst/runtime-build
export OQC_BUILD_DIR=/catalyst/oqc-build
export ENZYME_BUILD_DIR=/catalyst/enzyme-build
make wheel

# Exclude libopenblas as we rely on the openblas/lapack library shipped by scipy
auditwheel repair dist/*.whl -w ./wheel --no-update-tags --exclude libopenblasp-r0-23e5df77.3.21.dev.so
