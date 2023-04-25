FROM python:3-slim

# Add non-root user to the image.

ARG USERNAME="catalyst"
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME
RUN useradd --uid $USER_UID --gid $USER_GID -m $USERNAME
RUN apt-get update && apt-get install -y sudo
RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME
RUN chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

# Image specific instructions: Install Catalyst straight from PyPI.

RUN sudo apt-get install -y git
RUN pip install ipykernel

RUN pip install pennylane-catalyst

ENV SHELL /bin/bash
