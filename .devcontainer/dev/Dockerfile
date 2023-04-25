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

# Image specific instructions: Ensure developer requirements are met.
# Catalyst will be installed from source from within the running container.

RUN apt-get update && apt-get install -y --no-install-recommends git curl patch \
      make ninja-build clang lld ccache libomp-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER $USERNAME

RUN pip install --no-cache-dir cmake
RUN curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN "${HOME}/.cargo/bin/rustup" component add llvm-tools-preview
RUN pip install --no-cache-dir ipykernel

ENV SHELL /bin/bash
ENV PATH="${HOME}/.cargo/bin:${PATH}"
