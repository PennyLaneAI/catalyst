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

# Download and build Catalyst
FROM quay.io/pypa/manylinux_2_28_x86_64 AS base-catalyst
ARG PENNYLANE_VERSION
ARG CATALYST_VERSION
ARG LIGHTNING_VERSION
ARG GCC_VERSION
ARG LLVM_CACHE
ARG STABLEHLO_CACHE
RUN cat /etc/dnf.conf | sed "s/\[main\]/\[main\]\ntimeout=5/g" > /etc/dnf.conf
RUN dnf update -y && dnf install -y libzstd-devel gcc-toolset-13 openmpi-devel
ENV C_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/gcc
ENV CXX_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/g++
ENV PATH="/opt/rh/gcc-toolset-13/root/usr/bin:${PATH}"
WORKDIR /opt/catalyst
ENV PYTHON=/opt/python/cp313-cp313/bin/python
ENV PATH="/opt/python/cp313-cp313/bin:${PATH}"
RUN python -m pip install numpy "nanobind<2.13" pybind11 PyYAML cmake ninja

RUN git clone --depth 1 --branch ${CATALYST_VERSION} \
    --recurse-submodules --shallow-submodules \
    https://github.com/PennyLaneAI/catalyst.git /tmp/catalyst-src \
    && cp -a /tmp/catalyst-src/. /opt/catalyst/ \
    && rm -rf /tmp/catalyst-src


FROM base-catalyst AS build-llvm
ENV LLVM_BUILD_DIR=/opt/catalyst/llvm-build
ENV PATH="${LLVM_BUILD_DIR}/bin:${PATH}"

# ENV LLVM_TARGETS=check-mlir
RUN cd /opt/catalyst/mlir/llvm-project \
    && git apply /opt/catalyst/mlir/patches/llvm-bufferization-segfault.patch \
    && git apply /opt/catalyst/mlir/patches/llvm-python-bindinggen-annotations.patch
RUN cd /opt/catalyst/mlir/Enzyme \
    && git apply /opt/catalyst/mlir/patches/enzyme-nvvm-fabs-intrinsics.patch

RUN PYTHON=$PYTHON \
    C_COMPILER=$(which gcc)  \
    CXX_COMPILER=$(which g++)  \
    LLVM_BUILD_DIR="/opt/catalyst/llvm-build" \
    LLVM_PROJECTS="lld;mlir" \
    LLVM_TARGETS="lld check-mlir" \
    ENABLE_ZLIB=FORCE_ON \
    ENABLE_LLD=OFF \
    make llvm

# Build stablehlo dialect
ENV COMPILER_LAUNCHER=""
RUN mkdir /opt/catalyst/stablehlo-build
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
    -DLLVM_DIR="/opt/catalyst/llvm-build/lib/cmake/llvm" \
    -DENZYME_STATIC_LIB=ON \
    -DCMAKE_CXX_VISIBILITY_PRESET=default

RUN cmake --build /opt/catalyst/enzyme-build --target EnzymeStatic-22


FROM base-catalyst AS build-runtime
RUN dnf update -y && dnf install -y openmpi-devel libzstd-devel gcc-toolset-13
COPY --from=build-llvm /opt/catalyst/llvm-build /opt/catalyst/llvm-build
COPY --from=build-llvm /opt/catalyst/stablehlo-build /opt/catalyst/stablehlo-build
COPY --from=build-llvm /opt/catalyst/mlir/stablehlo /opt/catalyst/mlir/stablehlo
COPY --from=build-llvm /opt/catalyst/enzyme-build /opt/catalyst/enzyme-build
# Build catalyst runtime
ENV PATH="/opt/catalyst/llvm-build/bin:${PATH}"
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


FROM base-catalyst AS build-wheel-catalyst
COPY --from=build-runtime /opt/catalyst/llvm-build /opt/catalyst/llvm-build
COPY --from=build-runtime /opt/catalyst/stablehlo-build /opt/catalyst/stablehlo-build
COPY --from=build-runtime /opt/catalyst/enzyme-build /opt/catalyst/enzyme-build
COPY --from=build-runtime /opt/catalyst/runtime-build /opt/catalyst/runtime-build
COPY --from=build-runtime /opt/catalyst/oqc-build /opt/catalyst/oqc-build
COPY --from=build-runtime /opt/catalyst/quantum-build /opt/catalyst/quantum-build
ENV PATH="/opt/catalyst/llvm-build/bin:${PATH}"
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

# Build Pennylane Lightning Catalyst 
FROM pennylane-lightning AS pennylane-catalyst
ARG PENNYLANE_VERSION
ARG LIGHTNING_VERSION
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    apt-utils \
    ca-certificates \
    g++ \
    gcc \
    git \
    libgomp1 
COPY --from=build-wheel-catalyst /opt/catalyst/wheel /wheels    
RUN pip install --no-cache-dir --extra-index-url https://test.pypi.org/simple \
        /wheels/pennylane_catalyst*.whl 
RUN pip install --no-cache-dir \
    git+https://github.com/PennyLaneAI/pennylane.git@${PENNYLANE_VERSION}
RUN pip install --no-cache-dir \
    git+https://github.com/PennyLaneAI/pennylane-lightning.git@${LIGHTNING_VERSION} \
    && rm -rf /wheels