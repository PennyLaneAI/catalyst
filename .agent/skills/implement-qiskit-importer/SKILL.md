---
name: implement-qiskit-importer
description: Use when implementing the Python frontend to convert Qiskit circuits to Catalyst MLIR.
---

# Implement Qiskit Importer

## Overview
This skill guides the implementation of `QiskitToCatalystImporter`, a Python class that converts `qiskit.QuantumCircuit` objects into Catalyst's MLIR representation using the `catalyst.ir` module.

## When to Use
- Creating or modifying `qiskit_importer.py`.
- Adding support for new Qiskit gates (e.g., H, CX, RZ).
- Implementing control flow conversion (e.g., Qiskit `for_loop` to MLIR `scf.for`).
- Managing Qubit to SSA value mapping.

## Key Requirements
1.  **Class Structure**: Create `QiskitToCatalystImporter` taking a `qiskit.QuantumCircuit` in `__init__` or a `convert` method.
2.  **Context**: Use `catalyst.ir.Context` and `catalyst.ir.Location`.
3.  **Module Construction**: Build an MLIR module using `catalyst.ir.Module.create`.
4.  **Qubit Mapping**: Maintain a dictionary `self.qubit_map = {}` mapping `qiskit.circuit.Qubit` objects to their current MLIR SSA value (`ir.Value`).
5.  **Gate Mapping**:
    *   `HGate` -> `quantum.custom "h" ...`
    *   `CXGate` -> `quantum.custom "cnot" ...`
6.  **Control Flow**:
    *   Convert `circuit.for_loop` to `scf.for`.
    *   Ensure loop induction variables and region arguments are handled correctly.

## Implementation Pattern

```python
import qiskit
from catalyst.ir import Context, Module, Location, InsertionPoint
from catalyst.dialects import quantum, scf, func, arith

class QiskitToCatalystImporter:
    def __init__(self, circuit: qiskit.QuantumCircuit):
        self.circuit = circuit
        self.ctx = Context()
        self.ctx.load_all_dialects()
        self.module = Module.create(Location.unknown(self.ctx))
        self.qubit_map = {} # Maps qiskit.Qubit -> ir.Value

    def convert(self):
        with InsertionPoint(self.module.body):
            # Define main function
            # ... implementation details ...
            pass
    
    def _visit_instruction(self, instruction):
        # Dispatch based on instruction type
        if instruction.operation.name == "h":
            self._emit_h(instruction)
        elif instruction.operation.name == "cx":
            self._emit_cx(instruction)
        # ...
```

## Common Pitfalls
- **SSA Discipline**: MLIR is SSA-based. Transforming a qubit returns a *new* SSA value representing that qubit's new state. You must update `self.qubit_map` after every gate operation.
- **Context Management**: Ensure all IR operations are created within the correct `Context` and `InsertionPoint`.
- **Dialect Loading**: Forgot to load `quantum` or `scf` dialects in the context.
