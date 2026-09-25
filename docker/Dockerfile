# syntax=docker/dockerfile:1

FROM ubuntu:noble AS base
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential ca-certificates git lld python3-dev python3-venv zlib1g-dev libzstd-dev \
    && rm -rf /var/lib/apt/lists/*
ENV C_COMPILER=/usr/bin/gcc
ENV CXX_COMPILER=/usr/bin/g++
ENV COMPILER_LAUNCHER=""
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:${PATH}"
ENV PYTHON=$VIRTUAL_ENV/bin/python
RUN python3 -m venv $VIRTUAL_ENV \
    && pip install --no-cache-dir numpy "nanobind<2.13" pybind11 PyYAML cmake ninja \
        auditwheel patchelf

# ---------------------------------------------------------------------------
FROM base AS deps
WORKDIR /opt/catalyst
# Only the build machinery and the submodules, so
# Catalyst source edits don't invalidate the LLVM build.
COPY mlir/Makefile mlir/
COPY mlir/patches mlir/patches
COPY mlir/llvm-project mlir/llvm-project
COPY mlir/stablehlo mlir/stablehlo
COPY mlir/Enzyme mlir/Enzyme

ENV LLVM_BUILD_DIR=/opt/catalyst/llvm-build
ENV STABLEHLO_BUILD_DIR=/opt/catalyst/stablehlo-build
ENV ENZYME_BUILD_DIR=/opt/catalyst/enzyme-build

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

RUN BUILD_TYPE=Release make runtime
RUN make oqc
RUN make dialects

RUN make wheel \
    && auditwheel repair dist/*.whl -w /opt/wheel --no-update-tags \
        --exclude libopenblasp-r0-23e5df77.3.21.dev.so

# ---------------------------------------------------------------------------
FROM base AS catalyst
# Empty means use Catalyst's pinned versions in .dep-versions
ARG PENNYLANE_VERSION=
ARG LIGHTNING_VERSION=

COPY --from=build /opt/wheel /opt/wheel

RUN pip install --no-cache-dir --extra-index-url https://test.pypi.org/simple \
        /opt/wheel/*.whl
RUN if [ -n "${PENNYLANE_VERSION}" ]; then \
        pip install --no-cache-dir --no-deps --force-reinstall \
            "git+https://github.com/PennyLaneAI/pennylane.git@${PENNYLANE_VERSION}"; \
    fi


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

WORKDIR /work
