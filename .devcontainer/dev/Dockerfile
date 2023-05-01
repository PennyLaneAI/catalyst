FROM python:3-slim

# Add non-root user to the image.

ARG USERNAME="catalyst"
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME
RUN useradd --uid $USER_UID --gid $USER_GID -m $USERNAME
RUN apt-get update && apt-get install -y --no-install-recommends sudo \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME
RUN chmod 0440 /etc/sudoers.d/$USERNAME

## Image specific instructions: Ensure developer requirements are met.       ##
## Catalyst will be installed from source from within the running container. ##

# Install the required C++ build toolchain and git.
RUN apt-get update && apt-get install -y --no-install-recommends git curl patch \
      make ninja-build clang lld ccache libomp-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER $USERNAME

# Install a recent version of CMake not yet available via apt.
RUN pip install --no-cache-dir "cmake>=3.20"
# Install Python kernel for use with Jupyter Notebooks.
RUN pip install --no-cache-dir ipykernel

# Install the Rust toolchain for use with LLVM.
ENV PATH="/home/${USERNAME}/.cargo/bin:${PATH}"
RUN curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN rustup component add llvm-tools-preview

ENV SHELL /bin/bash
