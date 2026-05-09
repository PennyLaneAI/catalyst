---
name: build-catalyst
description: Use when building the Catalyst quantum compiler from source or recompiling components.
---

# Build Catalyst

## Overview
This skill provides instructions for building the Catalyst project, including the runtime, MLIR dialects, and Python frontend.

## When to Use
- Compiling the project after cloning.
- Recompiling after making changes to C++ or TableGen files.
- Cleaning build artifacts to fix build errors.
- Building specific components (runtime, mlir, frontend).

## Prerequisites
- `setup-catalyst-environment` skill must be completed.

## Commands

### Full Build
To build everything (Runtime, MLIR, Frontend, OQC):

```bash
make all
```

### Component Builds
To build specific components:

- **Runtime**: `make runtime`
- **MLIR Dialects**: `make mlir`
- **Frontend (Python)**: `make frontend`
- **OQC Runtime**: `make oqc`

### Cleaning Builds
If typical build commands fail with inexplicable errors, try cleaning:

- **Clean All**: `make clean-all` (Removes everything, including LLVM build)
- **Clean Catalyst**: `make clean-catalyst` (Removes Catalyst artifacts, keeps LLVM)
- **Clean Specific**: `make clean-runtime`, `make clean-mlir`, `make clean-oqc`

## Troubleshooting

### "ENOTEMPTY" or Directory Issues
If you see errors related to non-empty directories during CMake configuration:
1. Run `make clean-catalyst`
2. If that fails, manually remove the build directory: `rm -rf mlir/build runtime/build`

### "Ninja: build stopped: subcommand failed"
Check the error output above the stop message. Common causes:
- **Missing dependencies**: Run `setup-catalyst-environment`.
- **Compiler errors**: Check for syntax errors in your changes.
- **Linker errors**: Ensure all libraries are present.

### Incremental Builds
`make` will attempt incremental builds. If you change a header file widely used, it may trigger a large rebuild.

### Ccache
Ensure `ccache` is in your PATH to speed up recompilation.
