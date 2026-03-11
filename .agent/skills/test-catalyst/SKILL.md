---
name: test-catalyst
description: Use when verifying Catalyst build, running test suites, or checking for regressions.
---

# Test Catalyst

## Overview
This skill guides you through running the various test suites available in Catalyst to ensure the build is correct and functional.

## When to Use
- After a successful build to verify functionality.
- Before submitting changes to ensure no regressions.
- When debugging specific components.

## Prerequisites
- `build-catalyst` skill must be completed (project must be built).

## Test Commands

### Run Everything
To run all test suites (Runtime, Frontend, Demos):

```bash
make test
```
*Note: This can take a significant amount of time.*

### Component Tests

#### Runtime (C++)
To test the runtime library:

```bash
make test-runtime
```

#### Frontend (Python & Lit)
To test the Python frontend and MLIR dialects:

```bash
make test-frontend
```

This runs both `pytest` (Python logic) and `lit` (MLIR/LLVM tests).

#### Python Only
To run only the Python test suite (faster iteration for frontend changes):

```bash
make pytest
```
You can also run specific test files using `pytest` directly:
```bash
python3 -m pytest frontend/test/pytest/test_name.py
```

#### Lit Only (MLIR/LLVM)
To run only the LLVM Integrated Tester (lit) suite:

```bash
make lit
```

#### OQC Runtime
To test the OQC runtime integration:

```bash
make test-oqc
```

## Troubleshooting
- **Missing shared libraries**: Ensure `LD_LIBRARY_PATH` (Linux) or `DYLD_LIBRARY_PATH` (macOS) is set correctly if running binaries manually. `make test` usually handles this.
- **Python import errors**: Ensure `PYTHONPATH` includes the build artifacts if running `pytest` manually.
- **Flaky tests**: Some tests involving quantum simulation might be flaky. Re-run to confirm.
