# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Define global build defaults
ARG PENNYLANE_VERSION=main
ARG LIGHTNING_VERSION=main
ARG CATALYST_VERSION=main
ARG GCC_VERSION=13
ARG CUDA_INSTALLER=https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run
ARG ROCM_INSTALLER=https://repo.radeon.com/amdgpu-install/7.0.3/ubuntu/noble/amdgpu-install_7.0.3.70003-1_all.deb
ARG AMD_ARCH=AMD_GFX942
ARG CUDA_ARCH=AMPERE80

# Create basic runtime environment base on Ubuntu 24.04 (noble)
# Create and activate runtime virtual environment
FROM ubuntu:noble AS base-runtime
ARG DEBIAN_FRONTEND=noninteractive
ARG LIGHTNING_VERSION
ARG PENNYLANE_VERSION
ARG CATALYST_VERSION
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    apt-utils \
    ca-certificates \
    git \
    libgomp1 \
    python3 \
    python3-pip \
    python3-venv \
    tzdata \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Create basic build environment with build tools and compilers
FROM base-runtime AS base-build
ARG GCC_VERSION
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    build-essential \
    ccache \
    cmake \
    curl \
    ninja-build \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN update-alternatives \
    --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VERSION} 100 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-${GCC_VERSION} \
    --slave /usr/bin/gcov gcov /usr/bin/gcov-${GCC_VERSION}
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache
RUN ccache --set-config=cache_dir=/opt/ccache

# Create and activate build virtual environment
# Install Lightning dev requirements

# FROM base-build AS base-build-python
# ARG LIGHTNING_VERSION
# WORKDIR /opt/pennylane-lightning
# ENV VIRTUAL_ENV=/opt/venv-build
# RUN python3 -m venv $VIRTUAL_ENV
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# RUN rm -rf tmp && git clone --depth 1 --branch ${LIGHTNING_VERSION} https://github.com/PennyLaneAI/pennylane-lightning.git tmp\
#     && mv tmp/* /opt/pennylane-lightning && rm -rf tmp
# RUN pip install --no-cache-dir build cmake ninja toml wheel setuptools>=75.8.1

# # Download Lightning release and build lightning-qubit backend
# FROM base-build-python AS build-wheel-lightning-qubit
# WORKDIR /opt/pennylane-lightning
# RUN pip uninstall -y pennylane-lightning
# RUN python scripts/configure_pyproject_toml.py || true
# RUN python -m build --wheel

# # Install lightning-qubit backend
# FROM base-runtime AS wheel-lightning-qubit
# ARG PENNYLANE_VERSION
# ARG CATALYST_VERSION
# COPY --from=build-wheel-lightning-qubit /opt/pennylane-lightning/dist/ /
# RUN pip install --force-reinstall --no-cache-dir pennylane_lightning*.whl && rm pennylane_lightning*.whl
# RUN git clone --depth 1 --branch ${CATALYST_VERSION} --no-recurse-submodules \
#     https://github.com/PennyLaneAI/catalyst.git /tmp/catalyst
# RUN pip install --no-cache-dir \
#     git+https://github.com/PennyLaneAI/pennylane.git@${PENNYLANE_VERSION} \
#     && pip install --no-cache-dir --no-deps /tmp/catalyst && rm -rf /tmp/catalyst

# # Download Lightning release and build lightning-kokkos backend with Kokkos-OpenMP
# FROM base-build-python AS build-wheel-lightning-kokkos-openmp
# WORKDIR /opt/pennylane-lightning
# ENV PL_BACKEND=lightning_kokkos
# RUN pip uninstall -y pennylane-lightning
# RUN python scripts/configure_pyproject_toml.py || true
# RUN CMAKE_ARGS="-DKokkos_ENABLE_SERIAL:BOOL=ON -DKokkos_ENABLE_OPENMP:BOOL=ON" python -m build --wheel

# # Install lightning-kokkos OpenMP backend
# FROM base-runtime AS wheel-lightning-kokkos-openmp
# ARG PENNYLANE_VERSION
# ARG CATALYST_VERSION
# COPY --from=build-wheel-lightning-kokkos-openmp /opt/pennylane-lightning/dist/ /
# COPY --from=build-wheel-lightning-qubit /opt/pennylane-lightning/dist/ /
# RUN pip install --force-reinstall --no-cache-dir pennylane_lightning*.whl && rm pennylane_lightning*.whl
# RUN git clone --depth 1 --branch ${CATALYST_VERSION} --no-recurse-submodules \
#     https://github.com/PennyLaneAI/catalyst.git /tmp/catalyst
# RUN pip install --no-cache-dir \
#     git+https://github.com/PennyLaneAI/pennylane.git@${PENNYLANE_VERSION} \
#     && pip install --no-cache-dir --no-deps /tmp/catalyst && rm -rf /tmp/catalyst

# # Install CUDA-12 in build venv image
# FROM base-build-python AS base-build-cuda
# WORKDIR /opt/cuda-build
# ARG CUDA_INSTALLER
# RUN curl -o cuda-install.run ${CUDA_INSTALLER}
# RUN chmod a+x cuda-install.run
# RUN ./cuda-install.run --silent --toolkit --toolkitpath=/usr/local/cuda-$(echo ${CUDA_INSTALLER} | grep -o -P '/cuda/.{0,4}' | cut -d / -f 3)
# ENV PATH=/usr/local/cuda/bin:${PATH}
# ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# # Download Lightning release and build lightning-kokkos backend with Kokkos-CUDA
# FROM base-build-cuda AS build-wheel-lightning-kokkos-cuda
# ARG CUDA_ARCH
# WORKDIR /opt/pennylane-lightning
# ENV PL_BACKEND=lightning_kokkos
# RUN pip uninstall -y pennylane-lightning
# RUN echo >> cmake/support_kokkos.cmake && echo "find_package(CUDAToolkit REQUIRED)" >> cmake/support_kokkos.cmake
# RUN python scripts/configure_pyproject_toml.py || true
# RUN CMAKE_ARGS="-DKokkos_ENABLE_SERIAL:BOOL=ON \
#     -DKokkos_ENABLE_OPENMP:BOOL=ON \
#     -DKokkos_ENABLE_CUDA:BOOL=ON \
#     -DKokkos_ARCH_${CUDA_ARCH}=ON" \
#     python -m build --wheel

# # Install python3 and setup runtime virtual env in CUDA-12-runtime image (includes CUDA runtime and math libraries)
# # Install lightning-kokkos CUDA backend
# FROM nvidia/cuda:12.9.1-base-ubuntu24.04 AS wheel-lightning-kokkos-cuda
# ARG PENNYLANE_VERSION
# ARG CATALYST_VERSION
# ARG GCC_VERSION
# ENV DEBIAN_FRONTEND=noninteractive
# RUN apt-get update \
#     && apt-get update \
#     && apt-get install --no-install-recommends -y \
#     gcc-${GCC_VERSION} g++-${GCC_VERSION} cpp-${GCC_VERSION} \
#     libgomp1 \
#     git \
#     python3 \
#     python3-pip \
#     python3-venv \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*
# ENV VIRTUAL_ENV=/opt/venv
# RUN python3 -m venv $VIRTUAL_ENV
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# COPY --from=build-wheel-lightning-kokkos-cuda /opt/pennylane-lightning/dist/ /
# COPY --from=build-wheel-lightning-qubit /opt/pennylane-lightning/dist/ /
# RUN pip install --force-reinstall --no-cache-dir pennylane_lightning*.whl && rm pennylane_lightning*.whl
# RUN git clone --depth 1 --branch ${CATALYST_VERSION} --no-recurse-submodules \
#     https://github.com/PennyLaneAI/catalyst.git /tmp/catalyst
# RUN pip install --no-cache-dir \
#     git+https://github.com/PennyLaneAI/pennylane.git@${PENNYLANE_VERSION} \
#     && pip install --no-cache-dir --no-deps /tmp/catalyst && rm -rf /tmp/catalyst

# # Download and build Lightning-GPU release
# FROM base-build-cuda AS build-wheel-lightning-gpu
# WORKDIR /opt/pennylane-lightning
# ENV PL_BACKEND=lightning_gpu
# RUN pip install --no-cache-dir wheel custatevec-cu12
# RUN pip uninstall -y pennylane-lightning
# RUN python scripts/configure_pyproject_toml.py || true
# RUN CUQUANTUM_SDK=$(python -c "import site; print( f'{site.getsitepackages()[0]}/cuquantum/lib')") python -m build --wheel


# # Install python3 and setup runtime virtual env in CUDA-12-runtime image (includes CUDA runtime and math libraries)
# # Install lightning-kokkos CUDA backend
# FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04 AS wheel-lightning-gpu
# ARG PENNYLANE_VERSION
# ARG CATALYST_VERSION
# ENV DEBIAN_FRONTEND=noninteractive
# RUN apt-get update \
#     && apt-get install --no-install-recommends -y \
#     git \
#     libgomp1 \
#     python3 \
#     python3-pip \
#     python3-venv \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*
# ENV VIRTUAL_ENV=/opt/venv
# RUN python3 -m venv $VIRTUAL_ENV
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# RUN pip install --no-cache-dir custatevec-cu12
# ENV LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/cuquantum/lib:$LD_LIBRARY_PATH"
# COPY --from=build-wheel-lightning-gpu /opt/pennylane-lightning/dist/ /
# COPY --from=build-wheel-lightning-qubit /opt/pennylane-lightning/dist/ /
# RUN pip install --no-cache-dir --force-reinstall pennylane_lightning*.whl && rm pennylane_lightning*.whl
# RUN git clone --depth 1 --branch ${CATALYST_VERSION} --no-recurse-submodules \
#     https://github.com/PennyLaneAI/catalyst.git /tmp/catalyst
# RUN pip install --no-cache-dir \
#     git+https://github.com/PennyLaneAI/pennylane.git@${PENNYLANE_VERSION} \
#     && pip install --no-cache-dir --no-deps /tmp/catalyst && rm -rf /tmp/catalyst

# # Install ROCm in build venv image
# FROM base-build-python AS base-build-rocm
# ARG ROCM_INSTALLER
# RUN wget --progress=dot:giga ${ROCM_INSTALLER}
# RUN apt-get update \
#     && apt-get upgrade -y \
#     && apt-get install --no-install-recommends -y \
#        gnupg \
#        gcc-11 g++-11 \
#        ./$(echo ${ROCM_INSTALLER} | xargs -I {} basename {}) \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*
# RUN wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor | tee /etc/apt/trusted.gpg.d/rocm.gpg > /dev/null
# RUN amdgpu-install -y --accept-eula --usecase=rocm,hiplibsdk --no-dkms

# # Download Lightning release and build lightning-kokkos backend with Kokkos-ROCm
# FROM base-build-rocm AS build-wheel-lightning-kokkos-rocm
# ARG AMD_ARCH
# WORKDIR /opt/pennylane-lightning
# ENV CMAKE_PREFIX_PATH=/opt/rocm:$CMAKE_PREFIX_PATH
# ENV CXX=hipcc
# ENV PL_BACKEND=lightning_amdgpu
# RUN pip uninstall -y pennylane-lightning
# RUN python scripts/configure_pyproject_toml.py || true
# RUN CMAKE_ARGS="-DKokkos_ENABLE_SERIAL:BOOL=ON \
#     -DKokkos_ENABLE_HIP:BOOL=ON \
#     -DKokkos_ARCH_${AMD_ARCH}=ON" \
#     python -m build --wheel

# # Install lightning-amdgpu HIP backend
# FROM rocm/dev-ubuntu-24.04:7.0.2 AS wheel-lightning-kokkos-rocm
# ARG PENNYLANE_VERSION
# ARG CATALYST_VERSION
# ENV DEBIAN_FRONTEND=noninteractive
# RUN apt-get update \
#     && apt-get install --no-install-recommends -y \
#     git \
#     libgomp1 \
#     libomp-dev \
#     python3 \
#     python3-pip \
#     python3-venv \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*
# ENV VIRTUAL_ENV=/opt/venv
# RUN python3 -m venv $VIRTUAL_ENV
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# ENV LD_LIBRARY_PATH="/usr/lib/llvm-14/lib:$LD_LIBRARY_PATH"
# COPY --from=build-wheel-lightning-kokkos-rocm /opt/pennylane-lightning/dist/ /
# COPY --from=build-wheel-lightning-qubit /opt/pennylane-lightning/dist/ /
# RUN pip install --force-reinstall --no-cache-dir pennylane_lightning*.whl && rm pennylane_lightning*.whl
# RUN git clone --depth 1 --branch ${CATALYST_VERSION} --no-recurse-submodules \
#     https://github.com/PennyLaneAI/catalyst.git /tmp/catalyst
# RUN pip install --no-cache-dir \
#     git+https://github.com/PennyLaneAI/pennylane.git@${PENNYLANE_VERSION} \
#     && pip install --no-cache-dir --no-deps /tmp/catalyst && rm -rf /tmp/catalyst


# Download and build Catalyst
FROM quay.io/pypa/manylinux_2_28_x86_64 AS wheel-catalyst
ARG PENNYLANE_VERSION
ARG CATALYST_VERSION
ARG GCC_VERSION
RUN cat /etc/dnf.conf | sed "s/\[main\]/\[main\]\ntimeout=5/g" > /etc/dnf.conf
RUN dnf update -y && dnf install -y libzstd-devel gcc-toolset-13

WORKDIR /opt/catalyst
ENV PYTHON=/opt/python/cp313-cp313/bin/python
ENV PATH="/opt/python/cp313-cp313/bin:${PATH}"
RUN python -m pip install numpy "nanobind<2.13" pybind11 PyYAML cmake ninja

ENV C_COMPILER=/usr/bin/gcc
ENV CXX_COMPILER=/usr/bin/g++
ENV LLVM_BUILD_DIR=/opt/catalyst/llvm-build

RUN git clone --depth 1 --branch ${CATALYST_VERSION} --recurse-submodules --shallow-submodules \
    https://github.com/PennyLaneAI/catalyst.git /tmp/catalyst-src \
    && cp -a /tmp/catalyst-src/. /opt/catalyst/ \
    && rm -rf /tmp/catalyst-src

# ENV LLVM_TARGETS=check-mlir

RUN PYTHON=$PYTHON \
    C_COMPILER=$C_COMPILER \
    CXX_COMPILER=$CXX_COMPILER \
    LLVM_BUILD_DIR="$/opt/catalyst/llvm-build" \
    LLVM_PROJECTS="lld;mlir" \
    LLVM_TARGETS="lld check-mlir" \
    ENABLE_ZLIB=FORCE_ON \
    ENABLE_LLD=OFF \
    make llvm

# Build stablehlo dialect
ENV COMPILER_LAUNCHER=""
RUN C_COMPILER=$(which gcc) \
    CXX_COMPILER=$(which g++) \
    LLVM_BUILD_DIR="$(pwd)/llvm-build" \
    STABLEHLO_BUILD_DIR="/opt/catalyst/stablehlo-build" \
    COMPILER_LAUNCHER="" \
    ENABLE_LLD=OFF \
    make stablehlo

# Build enzyme
RUN cmake -S mlir/Enzyme/enzyme -B /opt/catalyst/enzyme-build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_DIR="/opt/catalyst/lib/cmake/llvm" \
    -DENZYME_STATIC_LIB=ON \
    -DCMAKE_CXX_VISIBILITY_PRESET=default

RUN cmake --build /opt/catalyst/enzyme-build --target EnzymeStatic-22
# install dependencies
RUN dnf update -y && dnf install -y openmpi-devel libzstd-devel gcc-toolset-13
# Build catalyst runtime
RUN cmake -S runtime -B /opt/catalyst/runtime-build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="/opt/catalyst/runtime-build/lib" \
    -DPython_EXECUTABLE=$PYTHON \
    -DENABLE_OPENQASM=ON \
    -DENABLE_OQD=OFF \
    -DMLIR_INCLUDE_DIRS="/opt/catalyst/mlir/llvm-project/mlir/include"

RUN cmake --build /opt/catalyst/runtime-build --target rt_capi rt_rsdecomp rt_decoder rtd_openqasm rtd_null_qubit
# Build OQC-Runtime
RUN OQC_BUILD_DIR="/opt/catalyst/oqc-build" \
    RT_BUILD_DIR="/opt/catalyst/runtime-build" \
    make oqc
# Build Quantum and Gradient Dialects
RUN  cmake -S mlir -B /opt/catalyst/quantum-build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_PREFIX_PATH=/opt/catalyst \
    -DQUANTUM_ENABLE_BINDINGS_PYTHON=ON \
    -DPython_EXECUTABLE=$PYTHON \
    -DPython3_EXECUTABLE=$PYTHON \
    -DPython3_NumPy_INCLUDE_DIRS=$($PYTHON -c "import numpy as np; print(np.get_include())") \
    -DMLIR_DIR="/opt/catalyst/llvm-build/lib/cmake/mlir" \
    -DSTABLEHLO_DIR="/opt/catalyst/mlir/stablehlo" \
    -DSTABLEHLO_BUILD_DIR="/opt/catalyst/stablehlo-build" \
    -DEnzyme_DIR="/opt/catalyst/enzyme-build" \
    -DENZYME_SRC_DIR="/opt/catalyst/mlir/Enzyme" \
    -DLLVM_ENABLE_ZLIB=FORCE_ON \
    -DLLVM_ENABLE_ZSTD=OFF \
    -DLLVM_ENABLE_LLD=ON
RUN cmake --build /opt/catalyst/quantum-build --target check-dialects catalyst-cli
RUN cd /opt/catalyst/quantum-build && cpack
# Build plugin wheel
RUN MLIR_DIR="/opt/catalyst/llvm-build/lib/cmake/mlir" \
    LLVM_BUILD_DIR="/opt/catalyst/llvm-build" \
    make plugin-wheel
RUN PYTHON=$PYTHON \
    LLVM_BUILD_DIR="/opt/catalyst/llvm-build" \
    STABLEHLO_BUILD_DIR="/opt/catalyst/stablehlo-build" \
    DIALECTS_BUILD_DIR="/opt/catalyst/quantum-build" \
    RT_BUILD_DIR="/opt/catalyst/runtime-build" \
    OQC_BUILD_DIR="/opt/catalyst/oqc-build" \
    ENZYME_BUILD_DIR="/opt/catalyst/enzyme-build" \
    make wheel

RUN auditwheel repair dist/*.whl -w ./wheel --no-update-tags --exclude libopenblasp-r0-23e5df77.3.21.dev.so
