import pennylane as qml
from pathlib import Path


class CudaQDevice(qml.QubitDevice):

    name = "CudaQ Device"
    short_name = "cudaq"
    pennylane_requires = "0.34.0"
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
