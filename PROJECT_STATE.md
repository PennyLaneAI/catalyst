# Project State

## Current Version
- **Branch**: `feature/translation_qasm`
- **Latest Commit**: `e9e8f6b9` (Fix quantum-opt segfault and ensure QASM 3 pipeline runs after Catalyst canonicalization)

## Active Work
- Implemented and verified OpenQASM 3 generation from Qiskit circuits using the `quantum-translate` MLIR pass.
- Enhanced the Qiskit-to-Catalyst MLIR importer to support conditional logic (`if_else`), mid-circuit measurements, and parameterized gates.
- Fixed a translation segfault in `ExtractOp` and added `InsertOp` handling to allow proper execution of OpenQASM translation *after* running backend decomposition and canonicalization passes (`quantum-opt`).

## Component Status
- **`qiskit_importer_standalone.py`**: Supports basic gates, parameterized gates (`rx`, `ry`, `rz`), mid-circuit measurements with classical bit tracking, and `IfElseOp` translation.
- **`mlir/lib/Target/OpenQASM3/TranslateToQASM3.cpp`**: Implements translation of MLIR to OpenQASM 3.0, properly handling variables canonicalized into attributes.
- **Test Suite (`test_translation_qasm3/`)**: 7 tests currently passing natively with the full canonicalization pipeline.

## Environment Info
- **Environment**: Conda environment `catalyst`
- **Build Status**: MLIR tools (`quantum-translate`, `quantum-opt`) are built.
- **Execution**: Tests run via `python test_translation_qasm3/test_translation.py` (with PYTHONPATH correctly mapping `mlir_core`).
