# syntax=docker/dockerfile:1

FROM ubuntu:noble AS base
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential ca-certificates git lld python3-dev python3-venv zlib1g-dev libzstd-dev \
    && rm -rf /var/lib/apt/lists/*
ENV C_COMPILER=/usr/bin/gcc
ENV CXX_COMPILER=/usr/bin/g++
ENV COMPILER_LAUNCHER=""
# Noble's python is externally managed (PEP 668), so everything goes in one venv that the
# final image keeps too.
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:${PATH}"
ENV PYTHON=$VIRTUAL_ENV/bin/python
RUN python3 -m venv $VIRTUAL_ENV \
    && pip install --no-cache-dir numpy "nanobind<2.13" pybind11 PyYAML cmake ninja \
        auditwheel patchelf

# ---------------------------------------------------------------------------
FROM base AS deps
WORKDIR /opt/catalyst
# Only the build machinery and the submodules (git submodule update --init --depth 1), so
# Catalyst source edits don't invalidate the LLVM build. The make targets apply the patches.
COPY mlir/Makefile mlir/
COPY mlir/patches mlir/patches
COPY mlir/llvm-project mlir/llvm-project
COPY mlir/stablehlo mlir/stablehlo
COPY mlir/Enzyme mlir/Enzyme

ENV LLVM_BUILD_DIR=/opt/catalyst/llvm-build
ENV STABLEHLO_BUILD_DIR=/opt/catalyst/stablehlo-build
ENV ENZYME_BUILD_DIR=/opt/catalyst/enzyme-build

# Makefile defaults, as in check-catalyst.yaml. The default LLVM_TARGETS includes
# check-mlir, which looks like a test target to drop, but it is what pulls in the MLIR
# libraries, tools and Python bindings.
RUN make -C mlir llvm
RUN make -C mlir stablehlo
RUN make -C mlir enzyme

# ---------------------------------------------------------------------------
FROM deps AS build
WORKDIR /opt/catalyst
# The submodules are already in place from deps.
COPY --exclude=mlir/llvm-project --exclude=mlir/stablehlo --exclude=mlir/Enzyme . .

ENV RT_BUILD_DIR=/opt/catalyst/runtime-build
ENV OQC_BUILD_DIR=/opt/catalyst/oqc-build
ENV DIALECTS_BUILD_DIR=/opt/catalyst/quantum-build

# BUILD_TYPE is explicit because runtime/Makefile defaults to RelWithDebInfo, whose debug
# info would ship in the image.
RUN BUILD_TYPE=Release make runtime
RUN make oqc

# Not `make dialects`: that also builds check-dialects and check-unit-tests and runs the
# Catch2 suite. Targets are named rather than building `all`, because `all` also builds
# stim's CLI executable -- Catalyst only links libstim, and only libstim gets the `-w`
# that keeps CMAKE_COMPILE_WARNING_AS_ERROR from failing on stim's own sources.
# QuantumPythonModules is named separately because catalyst-cli does not depend on it,
# but `make wheel` copies its output out of python_packages/. default_pipelines and
# QuantumPythonDecompositions, which the wheel also needs, are already
# add_dependencies() of CatalystCompilerDriver.
RUN cmake -S mlir -B ${DIALECTS_BUILD_DIR} -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DCMAKE_PREFIX_PATH=/opt/catalyst \
        -DQUANTUM_ENABLE_BINDINGS_PYTHON=ON \
        -DPython_EXECUTABLE=$PYTHON \
        -DPython3_EXECUTABLE=$PYTHON \
        -DPython3_NumPy_INCLUDE_DIRS=$($PYTHON -c "import numpy as np; print(np.get_include())") \
        -DMLIR_DIR="${LLVM_BUILD_DIR}/lib/cmake/mlir" \
        -DSTABLEHLO_DIR=/opt/catalyst/mlir/stablehlo \
        -DSTABLEHLO_BUILD_DIR="${STABLEHLO_BUILD_DIR}" \
        -DEnzyme_DIR="${ENZYME_BUILD_DIR}" \
        -DENZYME_SRC_DIR=/opt/catalyst/mlir/Enzyme \
        -DLLVM_ENABLE_ZLIB=FORCE_ON \
        -DLLVM_ENABLE_ZSTD=OFF \
        -DLLVM_ENABLE_LLD=ON \
    && cmake --build ${DIALECTS_BUILD_DIR} --target catalyst-cli QuantumPythonModules

# The exclude is CI's: Catalyst relies on the openblas shipped by scipy.
RUN make wheel \
    && auditwheel repair dist/*.whl -w /opt/wheel --no-update-tags \
        --exclude libopenblasp-r0-23e5df77.3.21.dev.so

# ---------------------------------------------------------------------------
FROM base AS catalyst
# Empty means "whatever Catalyst pins in .dep-versions" — the combination CI tests.
ARG PENNYLANE_VERSION=
ARG LIGHTNING_VERSION=

COPY --from=build /opt/wheel /opt/wheel

# Catalyst first, so its pinned PennyLane/Lightning/Kokkos resolve as a consistent set;
# installing it last would silently downgrade anything pinned below. .dep-versions pins
# PennyLane and Lightning to dev builds, which are published to TestPyPI only -- hence
# the extra index, exactly as build-wheel-linux-x86_64.yaml does when testing the wheel.
RUN pip install --no-cache-dir --extra-index-url https://test.pypi.org/simple \
        /opt/wheel/*.whl
RUN if [ -n "${PENNYLANE_VERSION}" ]; then \
        pip install --no-cache-dir --no-deps --force-reinstall \
            "git+https://github.com/PennyLaneAI/pennylane.git@${PENNYLANE_VERSION}"; \
    fi
# Lightning's pyproject is templated per backend; configure_pyproject_toml.py materialises
# it. The default PL_BACKEND is lightning_qubit.
RUN if [ -n "${LIGHTNING_VERSION}" ]; then \
        git clone -q --depth 1 --branch "${LIGHTNING_VERSION}" \
            https://github.com/PennyLaneAI/pennylane-lightning.git /tmp/lightning \
        && cd /tmp/lightning && python scripts/configure_pyproject_toml.py \
        && pip install --no-cache-dir --no-deps --force-reinstall . \
        && git clean -fdxq \
        && PL_BACKEND=lightning_kokkos python scripts/configure_pyproject_toml.py \
        && CMAKE_ARGS="-DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=ON" \
            pip install --no-cache-dir --no-deps --force-reinstall . \
        && rm -rf /tmp/lightning; \
    fi
RUN python -c "import pennylane as qp; qp.device('lightning.qubit', wires=1); qp.device('lightning.kokkos', wires=1)"

WORKDIR /work
CMD ["python", "-c", "import catalyst, pennylane as qp; qp.about()"]
