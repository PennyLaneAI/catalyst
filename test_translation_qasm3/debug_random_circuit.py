#!/usr/bin/env python3
"""Debug script to investigate random circuit translation failures."""

import sys
import tempfile
import subprocess
from pathlib import Path

# Setup paths
sys.path.append(str(Path.cwd().parent))
sys.path.append(str(Path.cwd().parent / 'mlir/llvm-project/build/tools/mlir/python_packages/mlir_core'))

from qiskit.circuit.random import random_circuit
from qiskit import transpile
from qiskit_importer_standalone import QiskitToCatalystImporter

def decompose_to_standard_gates(circuit):
    """Decompose circuit to standard gates."""
    basis_gates = [
        'h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg', 'sx', 'sxdg',
        'rx', 'ry', 'rz', 'p', 'u', 'u1', 'u2', 'u3',
        'cx', 'cy', 'cz', 'ch', 'cp', 'crx', 'cry', 'crz', 'cu',
        'swap',
        'id', 'measure', 'barrier'
    ]
    return transpile(circuit, basis_gates=basis_gates, optimization_level=1)

# Generate the failing circuit
qc = random_circuit(4, 10, seed=42)
print("Original circuit:")
print(qc)
print()

# Decompose
decomposed = decompose_to_standard_gates(qc)
print("Decomposed circuit gates:")
print([op.operation.name for op in decomposed.data])
print()

# Convert to MLIR
importer = QiskitToCatalystImporter(decomposed)
module = importer.convert()

# Save original MLIR
with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
    f.write(str(module))
    mlir_file = f.name
    print(f"Original MLIR saved to: {mlir_file}")

# Run quantum-opt with canonicalization
opt_cmd = ['../mlir/build/bin/quantum-opt', '--pass-pipeline=builtin.module(canonicalize)', mlir_file]
result = subprocess.run(opt_cmd, capture_output=True, text=True)

if result.returncode != 0:
    print("quantum-opt FAILED:")
    print(result.stderr)
    sys.exit(1)

# Save optimized MLIR
with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
    f.write(result.stdout)
    opt_file = f.name
    print(f"Optimized MLIR saved to: {opt_file}")

# Try to translate
trans_cmd = ['../mlir/build/bin/quantum-translate', '--mlir-to-qasm3', opt_file]
result2 = subprocess.run(trans_cmd, capture_output=True, text=True)

if result2.returncode != 0:
    print("\nquantum-translate FAILED:")
    print(result2.stderr)
    print("\nShowing optimized MLIR:")
    with open(opt_file) as f:
        lines = f.readlines()
        for i, line in enumerate(lines, start=1):
            print(f"{i:3}: {line}", end='')
    sys.exit(1)

print("\nSUCCESS! QASM3 output:")
print(result2.stdout)
