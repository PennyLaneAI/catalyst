FROM --platform=linux/amd64 python:3-slim

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

## Image specific instructions: Install Catalyst straight from PyPI. ##

# Install git.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER $USERNAME

# Install Python kernel for use with Jupyter Notebooks.
RUN pip install --no-cache-dir ipykernel

RUN pip install --no-cache-dir PennyLane-Catalyst

ENV SHELL /bin/bash
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"
