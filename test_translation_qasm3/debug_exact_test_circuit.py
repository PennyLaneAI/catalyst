#!/usr/bin/env python3
"""Debug script that runs EXACTLY what the test runs."""

import sys
import tempfile
import subprocess
from pathlib import Path

# Setup paths - exactly as in conftest.py
sys.path.append(str(Path.cwd().parent))
sys.path.append(str(Path.cwd().parent / 'mlir/llvm-project/build/tools/mlir/python_packages/mlir_core'))

from test_translation_qasm3.random_circuit_generator import RandomCircuitGenerator, RandomCircuitConfig
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

# Exactly as the test does it
generator = RandomCircuitGenerator(seed=42)

config = RandomCircuitConfig(
    num_qubits=4,
    depth=10,
    measure=True,
    seed=42
)

circuit = generator.generate_standard_random(config)
print("Generated circuit gates:", set([op.operation.name for op in circuit.data]))
print()

# Decompose
circuit = decompose_to_standard_gates(circuit)
print("Decomposed circuit gates:", set([op.operation.name for op in circuit.data]))
print()

# Convert to MLIR
importer = QiskitToCatalystImporter(circuit)
module = importer.convert()

# Save original MLIR
with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
    f.write(str(module))
    mlir_file = f.name
    print(f"Original MLIR saved to: {mlir_file}")

# Run quantum-opt with canonicalization (as the test does now)
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
    print("\nShowing problematic MLIR lines:")
    with open(opt_file) as f:
        lines = f.readlines()
        # Find the error line (around line 82 based on earlier error)
        start = max(0, 72)
        end = min(len(lines), 92)
        for i in range(start, end):
            print(f"{i+1:3}: {lines[i]}", end='')
    sys.exit(1)

print("\nSUCCESS! QASM3 output:")
print(result2.stdout[:500])
