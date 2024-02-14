# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains a CudaQDevice and the qjit
entry point.
"""

from pathlib import Path

import cudaq
import pennylane as qml

from catalyst.cuda.catalyst_to_cuda_interpreter import interpret


def qjit(fn=None, **kwargs):
    """Wrapper around QJIT for CUDA-quantum."""

    if fn is not None:
        return interpret(fn)

    def wrap_fn(fn):
        return interpret(fn)

    return wrap_fn


class BaseCudaInstructionSet(qml.QubitDevice):
    """Base instruction set for CUDA-Quantum devices"""

    # TODO: Once 0.35 is released, remove -dev suffix.
    pennylane_requires = "0.35.0-dev"
    version = "0.1.0"
    author = "Xanadu, Inc."

    # pylint: disable=duplicate-code
    operations = [
        "CNOT",
        "CY",
        "CZ",
        "CRX",
        "CRY",
        "CRZ",
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "S",
        "T",
        "RX",
        "RY",
        "RZ",
        "SWAP",
        # "CSWAP", This is a bug in cuda-quantum. CSWAP is not exposed.
    ]
    observables = []
    config = Path(__file__).parent / "cuda_quantum.toml"

    def __init__(self, shots=None, wires=None, mps=False, multi_gpu=False):
        self.mps = mps
        self.multi_gpu = multi_gpu
        super().__init__(wires=wires, shots=shots)

    def apply(self, operations, **kwargs):
        """Unused"""
        raise NotImplementedError(
            "CudaQDevice must be used with `catalyst.qjit`"
        )  # pragma: no cover


class CudaQDevice(BaseCudaInstructionSet):
    """Concrete device class for qpp-cpu"""

    name = "CudaQ Device"
    short_name = "softwareq.qpp"


class NvidiaCuStateVec(BaseCudaInstructionSet):
    """Concrete device class for CuStateVec"""

    name = "CuStateVec"
    short_name = "nvidia.custatevec"

    def __init__(self, shots=None, wires=None, multi_gpu=False):  # pragma: no cover
        super().__init__(wires=wires, shots=shots, multi_gpu=multi_gpu)


class NvidiaCuTensorNet(BaseCudaInstructionSet):
    """Concrete device class for CuTensorNet"""

    name = "CuTensorNet"
    short_name = "nvidia.cutensornet"

    def __init__(self, shots=None, wires=None, mps=False):  # pragma: no cover
        super().__init__(wires=wires, shots=shots, mps=mps)
