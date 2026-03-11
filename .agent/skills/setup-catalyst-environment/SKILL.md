---
name: setup-catalyst-environment
description: Use when setting up a fresh machine or fixing environment issues for building Catalyst.
---

# Setup Catalyst Environment

## Overview
This skill ensures the development environment has all necessary dependencies installed and configured for building the Catalyst quantum compiler from source.

## When to Use
- Setting up a new development machine for Catalyst work.
- Building Catalyst fails due to missing tools (cmake, ninja, clang).
- Encountering "command not found" errors during build.
- Python environment issues (missing packages).

## Prerequisites
- macOS (arm64) or Linux (x86_64, aarch64)
- Git
- Python 3.11+

## Steps

### 1. Verification
Check if dependencies are already installed:

```bash
cmake --version
ninja --version
clang --version
ccache --version
python3 --version
```

### 2. Installation (macOS)
If dependencies are missing on macOS:

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew dependencies
brew install cmake ninja ccache gfortran

# Add ccache to PATH (optional but recommended)
export PATH="/usr/local/opt/ccache/libexec:$PATH"
```

### 3. Installation (Linux - Debian/Ubuntu)
If dependencies are missing on Linux:

```bash
sudo apt update
sudo apt install -y clang lld ccache make cmake ninja-build python3-dev
```

### 4. Repository Setup
Ensure submodules are initialized:

```bash
git submodule update --init --recursive
```

### 5. Python Dependencies
Install Python requirements:

```bash
pip install -r requirements.txt
pip install -r doc/requirements.txt # Optional, for docs
```

## Common Issues
- **Xcode license not accepted**: Run `sudo xcodebuild -license`.
- **Python version mismatch**: Ensure `python3` points to version 3.11 or higher.
- **Submodule errors**: If `git submodule update` fails, try `git submodule sync` first.
