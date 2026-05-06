#!/usr/bin/env python3
import sys
from pathlib import Path
import tempfile
import subprocess

sys.path.append(str(Path.cwd().parent))
sys.path.append(
    str(Path.cwd().parent / "mlir/llvm-project/build/tools/mlir/python_packages/mlir_core")
)

from random_circuit_generator import RandomCircuitGenerator, RandomCircuitConfig
from qiskit import transpile
from qiskit_importer_standalone import QiskitToCatalystImporter

# Generate EXACT same circuit as test
generator = RandomCircuitGenerator(seed=42)
config = RandomCircuitConfig(num_qubits=4, depth=10, measure=True, seed=42)
circuit = generator.generate_standard_random(config)

# Decompose
basis_gates = [
    "h",
    "x",
    "y",
    "z",
    "s",
    "sdg",
    "t",
    "tdg",
    "sx",
    "sxdg",
    "rx",
    "ry",
    "rz",
    "p",
    "u",
    "u1",
    "u2",
    "u3",
    "cx",
    "cy",
    "cz",
    "ch",
    "cp",
    "crx",
    "cry",
    "crz",
    "cu",
    "swap",
    "id",
    "measure",
    "barrier",
]
decomposed = transpile(circuit, basis_gates=basis_gates, optimization_level=1)

# Convert to MLIR
importer = QiskitToCatalystImporter(decomposed)
module = importer.convert()

# Save
with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
    f.write(str(module))
    mlir_file = f.name

# Run quantum-opt
opt_result = subprocess.run(
    [
        "../mlir/build/bin/quantum-opt",
        "--pass-pipeline=builtin.module(canonicalize)",
        mlir_file,
        "-o",
        "/tmp/test_opt.mlir",
    ],
    capture_output=True,
    text=True,
)

if opt_result.returncode != 0:
    print("quantum-opt FAILED")
else:
    # Try translate
    trans_result = subprocess.run(
        ["../mlir/build/bin/quantum-translate", "--mlir-to-qasm3", "/tmp/test_opt.mlir"],
        capture_output=True,
        text=True,
    )
    if trans_result.returncode != 0:
        print("FAILED")
    else:
        print("SUCCESS!")
