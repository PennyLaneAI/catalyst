# QASM 3 Translation Examples

This document provides practical examples of using the QASM 3 translation pipeline.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Creating Test Circuits](#creating-test-circuits)
3. [Using the Enhanced Importer](#using-the-enhanced-importer)
4. [Common Algorithms](#common-algorithms)
5. [Debugging and Troubleshooting](#debugging-and-troubleshooting)

---

## Basic Usage

### Example 1: Simple Bell State

```python
from qiskit import QuantumCircuit
from qiskit_importer_standalone import QiskitToCatalystImporter
import subprocess
import tempfile

# Create a Bell state
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

# Convert to MLIR
importer = QiskitToCatalystImporter(qc)
module = importer.convert()

print("Generated MLIR:")
print(module)

# Save to file
with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
    f.write(str(module))
    mlir_path = f.name

# Apply quantum-opt
subprocess.run([
    "./mlir/build/bin/quantum-opt",
    "--pass-pipeline=builtin.module(apply-transform-sequence, canonicalize, merge-rotations)",
    mlir_path,
    "-o", mlir_path
], check=True)

# Translate to QASM3
result = subprocess.run([
    "./mlir/build/bin/quantum-translate",
    "--mlir-to-qasm3",
    mlir_path
], capture_output=True, text=True, check=True)

print("\nGenerated QASM3:")
print(result.stdout)
```

**Output:**
```qasm
OPENQASM 3.0;
include "stdgates.inc";

def main() {
    qubit[2] q;
    bit[2] c;

    h q[0];
    cx q[0], q[1];
    c[0] = measure q[0];
    c[1] = measure q[1];
}
```

### Example 2: Parameterized Circuit

```python
from qiskit import QuantumCircuit, Parameter
from qiskit_importer_standalone import QiskitToCatalystImporter

# Create parameterized circuit
theta = Parameter('θ')
qc = QuantumCircuit(1, 1)
qc.rx(theta, 0)
qc.measure(0, 0)

# Bind parameter
bound_circuit = qc.bind_parameters({theta: 0.5})

# Convert
importer = QiskitToCatalystImporter(bound_circuit)
module = importer.convert()

print(module)
```

---

## Creating Test Circuits

### Example 3: GHZ State Generator

```python
from qiskit import QuantumCircuit

def create_ghz_circuit(n_qubits):
    """Create an n-qubit GHZ state."""
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Prepare superposition on first qubit
    qc.h(0)

    # Entangle all qubits
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

    # Measure all
    qc.measure(range(n_qubits), range(n_qubits))

    return qc

# Create 5-qubit GHZ state
ghz5 = create_ghz_circuit(5)

# Save to QASM file
with open('test_translation_qasm3/qasm3_circuits/ghz_5qubit_generated.qasm', 'w') as f:
    f.write(ghz5.qasm())

print("Circuit saved!")
```

### Example 4: Quantum Fourier Transform

```python
from qiskit import QuantumCircuit
import numpy as np

def qft(n):
    """Create n-qubit QFT circuit."""
    qc = QuantumCircuit(n, n)

    # QFT on n qubits
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            qc.cp(np.pi / 2**(k - j), k, j)

    # Swap qubits
    for i in range(n // 2):
        qc.swap(i, n - i - 1)

    qc.measure(range(n), range(n))
    return qc

# Create 4-qubit QFT
qft4 = qft(4)

# Test translation
from test_translation import run_pipeline
from pathlib import Path

# Save circuit
qft4.qasm(filename='test_translation_qasm3/qasm3_circuits/qft_4qubit.qasm')

# Run pipeline
circuit_path = Path('test_translation_qasm3/qasm3_circuits/qft_4qubit.qasm')
success = run_pipeline(circuit_path)

print(f"QFT translation: {'Success' if success else 'Failed'}")
```

---

## Using the Enhanced Importer

### Example 5: With Logging

```python
from qiskit_importer_enhanced import QiskitToCatalystImporter
from qiskit import QuantumCircuit
import logging

# Enable logging
logging.basicConfig(level=logging.DEBUG)

# Create circuit
qc = QuantumCircuit(3, 3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.measure([0, 1, 2], [0, 1, 2])

# Convert with logging
importer = QiskitToCatalystImporter(qc, enable_logging=True)
module = importer.convert()

# Get statistics
stats = importer.get_stats()
print(f"\nCompilation Statistics:")
print(f"  Gates processed: {stats['gates_processed']}")
print(f"  Measurements: {stats['measurements']}")
print(f"  Control flow operations: {stats['control_flow']}")
```

**Output:**
```
2026-03-11 10:15:23 - __main__ - INFO - Initialized importer for circuit with 3 qubits, 3 classical bits, 4 operations
2026-03-11 10:15:23 - __main__ - DEBUG - Allocated register with 3 qubits
2026-03-11 10:15:23 - __main__ - DEBUG - Extracted 3 qubits
2026-03-11 10:15:23 - __main__ - DEBUG - Processing instruction 0: h
2026-03-11 10:15:23 - __main__ - DEBUG - Emitted gate h on 1 qubit(s)
2026-03-11 10:15:23 - __main__ - DEBUG - Processing instruction 1: cx
2026-03-11 10:15:23 - __main__ - DEBUG - Emitted gate cnot on 2 qubit(s)
2026-03-11 10:15:23 - __main__ - DEBUG - Processing instruction 2: cx
2026-03-11 10:15:23 - __main__ - DEBUG - Emitted gate cnot on 2 qubit(s)
2026-03-11 10:15:23 - __main__ - DEBUG - Processing instruction 3: measure
2026-03-11 10:15:23 - __main__ - DEBUG - Measured qubit Qubit(QuantumRegister(3, 'q'), 0) into classical bit Clbit(ClassicalRegister(3, 'c'), 0)
2026-03-11 10:15:23 - __main__ - DEBUG - Measured qubit Qubit(QuantumRegister(3, 'q'), 1) into classical bit Clbit(ClassicalRegister(3, 'c'), 1)
2026-03-11 10:15:23 - __main__ - DEBUG - Measured qubit Qubit(QuantumRegister(3, 'q'), 2) into classical bit Clbit(ClassicalRegister(3, 'c'), 2)
2026-03-11 10:15:23 - __main__ - INFO - Conversion complete. Stats: {'gates_processed': 3, 'measurements': 1, 'control_flow': 0}

Compilation Statistics:
  Gates processed: 3
  Measurements: 1
  Control flow operations: 0
```

### Example 6: Error Handling

```python
from qiskit_importer_enhanced import QiskitToCatalystImporter, CompileError
from qiskit import QuantumCircuit

# Create circuit with conditional (without measurement)
qc = QuantumCircuit(2, 2)
qc.h(0)
# Missing measurement here!
qc.x(1).c_if(0, 1)  # This will fail

try:
    importer = QiskitToCatalystImporter(qc)
    module = importer.convert()
except CompileError as e:
    print(f"Compilation failed: {e}")
```

**Output:**
```
Compilation failed: Classical bit c[0] used in if_else but not measured previously.
Ensure measurement occurs before conditional operation.
```

---

## Common Algorithms

### Example 7: Grover's Algorithm

```python
from qiskit import QuantumCircuit
from math import pi

def grover_2qubit():
    """Grover's algorithm searching for |11⟩."""
    qc = QuantumCircuit(2, 2)

    # Initialize to equal superposition
    qc.h([0, 1])

    # Oracle: mark |11⟩
    qc.cz(0, 1)

    # Diffusion operator
    qc.h([0, 1])
    qc.x([0, 1])
    qc.cz(0, 1)
    qc.x([0, 1])
    qc.h([0, 1])

    qc.measure([0, 1], [0, 1])
    return qc

# Test
grover_circ = grover_2qubit()
grover_circ.qasm(filename='test_translation_qasm3/qasm3_circuits/grover_2qubit_example.qasm')

from qiskit_importer_standalone import QiskitToCatalystImporter
importer = QiskitToCatalystImporter(grover_circ)
module = importer.convert()

print("Grover circuit converted successfully!")
print(f"Circuit depth: {grover_circ.depth()}")
print(f"Gate count: {len(grover_circ.data)}")
```

### Example 8: Quantum Teleportation

```python
from qiskit import QuantumCircuit

def teleportation():
    """Quantum teleportation protocol."""
    qc = QuantumCircuit(3, 3)

    # Prepare state to teleport (|+⟩ in this example)
    qc.h(0)

    # Create Bell pair between qubits 1 and 2
    qc.h(1)
    qc.cx(1, 2)

    # Bell measurement on qubits 0 and 1
    qc.cx(0, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.measure(1, 1)

    # Conditional corrections on qubit 2
    qc.x(2).c_if(1, 1)
    qc.z(2).c_if(0, 1)

    # Final measurement
    qc.measure(2, 2)

    return qc

# Test
teleport = teleportation()

from qiskit_importer_standalone import QiskitToCatalystImporter
importer = QiskitToCatalystImporter(teleport)
module = importer.convert()

print("Teleportation circuit converted!")
```

### Example 9: Variational Quantum Eigensolver (VQE) Ansatz

```python
from qiskit import QuantumCircuit
import numpy as np

def vqe_ansatz(n_qubits, depth, parameters):
    """
    Hardware-efficient VQE ansatz.

    Args:
        n_qubits: Number of qubits
        depth: Circuit depth (number of layers)
        parameters: List of rotation angles
    """
    qc = QuantumCircuit(n_qubits, n_qubits)

    param_idx = 0
    for d in range(depth):
        # Rotation layer
        for q in range(n_qubits):
            qc.ry(parameters[param_idx], q)
            param_idx += 1

        # Entangling layer
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc

# Example with 3 qubits, depth 2
n_params = 3 * 2  # n_qubits * depth
params = np.random.uniform(0, 2 * np.pi, n_params)

vqe_circ = vqe_ansatz(3, 2, params)

from qiskit_importer_standalone import QiskitToCatalystImporter
importer = QiskitToCatalystImporter(vqe_circ)
module = importer.convert()

print(f"VQE ansatz converted: {n_params} parameters")
```

---

## Debugging and Troubleshooting

### Example 10: Inspecting MLIR Output

```python
from qiskit import QuantumCircuit
from qiskit_importer_standalone import QiskitToCatalystImporter

# Create circuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

# Convert
importer = QiskitToCatalystImporter(qc)
module = importer.convert()

# Save MLIR for inspection
with open('debug_output.mlir', 'w') as f:
    f.write(str(module))

print("MLIR saved to debug_output.mlir")
print("\nYou can now:")
print("  1. Inspect: cat debug_output.mlir")
print("  2. Validate: mlir/build/bin/quantum-opt debug_output.mlir")
print("  3. Translate: mlir/build/bin/quantum-translate --mlir-to-qasm3 debug_output.mlir")
```

### Example 11: Comparing Simulation Results

```python
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import qiskit.qasm3
from qiskit_importer_standalone import QiskitToCatalystImporter
import subprocess
import tempfile

def compare_circuits(original_circuit, qasm3_code, shots=1000):
    """Compare original and translated circuits."""
    sim = AerSimulator()

    # Simulate original
    qc_t = transpile(original_circuit, sim)
    result1 = sim.run(qc_t, shots=shots).result()
    counts1 = result1.get_counts()

    # Simulate translated
    qc2 = qiskit.qasm3.loads(qasm3_code)
    qc2_t = transpile(qc2, sim)
    result2 = sim.run(qc2_t, shots=shots).result()
    counts2 = result2.get_counts()

    print("Original circuit counts:")
    print(counts1)
    print("\nTranslated circuit counts:")
    print(counts2)

    # Simple comparison
    common_states = set(counts1.keys()) & set(counts2.keys())
    print(f"\nCommon measurement outcomes: {len(common_states)}")

# Example
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

# Convert to QASM3 (abbreviated)
importer = QiskitToCatalystImporter(qc)
module = importer.convert()

with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
    f.write(str(module))
    mlir_path = f.name

subprocess.run([
    "./mlir/build/bin/quantum-opt",
    "--pass-pipeline=builtin.module(apply-transform-sequence, canonicalize)",
    mlir_path, "-o", mlir_path
], check=True)

result = subprocess.run([
    "./mlir/build/bin/quantum-translate",
    "--mlir-to-qasm3", mlir_path
], capture_output=True, text=True, check=True)

compare_circuits(qc, result.stdout)
```

**Output:**
```
Original circuit counts:
{'00': 512, '11': 488}

Translated circuit counts:
{'00': 498, '11': 502}

Common measurement outcomes: 2
```

### Example 12: Batch Processing

```python
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit_importer_standalone import QiskitToCatalystImporter
import subprocess
import tempfile

def batch_translate(circuit_dir):
    """Translate all QASM files in a directory."""
    circuits_path = Path(circuit_dir)
    results = []

    for qasm_file in circuits_path.glob("*.qasm"):
        print(f"Processing {qasm_file.name}...")

        try:
            # Load circuit
            qc = QuantumCircuit.from_qasm_file(str(qasm_file))

            # Convert
            importer = QiskitToCatalystImporter(qc)
            module = importer.convert()

            # Optimize and translate
            with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
                f.write(str(module))
                mlir_path = f.name

            subprocess.run([
                "./mlir/build/bin/quantum-opt",
                "--pass-pipeline=builtin.module(apply-transform-sequence, canonicalize)",
                mlir_path, "-o", mlir_path
            ], check=True)

            result = subprocess.run([
                "./mlir/build/bin/quantum-translate",
                "--mlir-to-qasm3", mlir_path
            ], capture_output=True, text=True, check=True)

            # Save output
            output_path = qasm_file.parent / f"{qasm_file.stem}_qasm3.qasm"
            with open(output_path, 'w') as f:
                f.write(result.stdout)

            results.append((qasm_file.name, True, None))
            print(f"  ✓ Success")

        except Exception as e:
            results.append((qasm_file.name, False, str(e)))
            print(f"  ✗ Failed: {e}")

    # Summary
    print(f"\nBatch processing complete:")
    print(f"  Total: {len(results)}")
    print(f"  Success: {sum(1 for _, success, _ in results if success)}")
    print(f"  Failed: {sum(1 for _, success, _ in results if not success)}")

    return results

# Run on test circuits
results = batch_translate('test_translation_qasm3/qasm3_circuits')
```

---

## Performance Tips

### Example 13: Measuring Translation Time

```python
import time
from qiskit import QuantumCircuit
from qiskit_importer_standalone import QiskitToCatalystImporter
import subprocess
import tempfile

def measure_translation_time(circuit, num_runs=10):
    """Measure average translation time."""
    times = {
        'mlir_generation': [],
        'optimization': [],
        'qasm3_emission': []
    }

    for _ in range(num_runs):
        # MLIR generation
        start = time.time()
        importer = QiskitToCatalystImporter(circuit)
        module = importer.convert()
        times['mlir_generation'].append(time.time() - start)

        # Save MLIR
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
            f.write(str(module))
            mlir_path = f.name

        # Optimization
        start = time.time()
        subprocess.run([
            "./mlir/build/bin/quantum-opt",
            "--pass-pipeline=builtin.module(apply-transform-sequence, canonicalize)",
            mlir_path, "-o", mlir_path
        ], check=True, capture_output=True)
        times['optimization'].append(time.time() - start)

        # QASM3 emission
        start = time.time()
        subprocess.run([
            "./mlir/build/bin/quantum-translate",
            "--mlir-to-qasm3", mlir_path
        ], check=True, capture_output=True)
        times['qasm3_emission'].append(time.time() - start)

    # Calculate averages
    avg_times = {k: sum(v) / len(v) for k, v in times.items()}

    print(f"Average translation times ({num_runs} runs):")
    print(f"  MLIR generation:   {avg_times['mlir_generation']*1000:.2f} ms")
    print(f"  Optimization:      {avg_times['optimization']*1000:.2f} ms")
    print(f"  QASM3 emission:    {avg_times['qasm3_emission']*1000:.2f} ms")
    print(f"  Total:             {sum(avg_times.values())*1000:.2f} ms")

# Test with different circuit sizes
for n in [2, 5, 10, 20]:
    print(f"\n{n}-qubit GHZ state:")
    qc = QuantumCircuit(n, n)
    qc.h(0)
    for i in range(n-1):
        qc.cx(i, i+1)
    qc.measure(range(n), range(n))

    measure_translation_time(qc, num_runs=5)
```

---

For more examples, see:
- [CLAUDE.md](../CLAUDE.md) - Full testing guide
- [README.md](README.md) - Test suite documentation
- [test_translation.py](test_translation.py) - Complete test runner

**Last Updated**: 2026-03-11
