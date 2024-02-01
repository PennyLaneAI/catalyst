from pathlib import Path

import pennylane as qml

from catalyst import pennylane_extensions
from catalyst.utils.contexts import EvaluationContext


def qjit(fn=None):
    from catalyst.compilation_pipelines import qjit_cuda

    return qjit_cuda(fn)


class CudaQDevice(qml.QubitDevice):

    name = "CudaQ Device"
    short_name = "cudaq"
    pennylane_requires = "0.35.0-dev"
    version = "1.0"
    author = "Catalyst authors wrote this wrapper for CUDA Quantum"

    operations = [
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "S",
        "T",
        "RX",
        "RY",
        "RZ",
    ]
    observables = []
    config = Path(__file__).parent / "cuda.toml"

    def __init__(self, shots=None, wires=None):
        super().__init__(wires=wires, shots=shots)

    def apply(self, operations, **kwargs):
        """Unused"""
        raise RuntimeError("We are not applying operation by operation.")


__all__ = (
    "qjit",
    "CudaQDevice",
)
