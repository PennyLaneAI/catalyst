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
        return interpret(fn, **kwargs)

    def wrap_fn(fn):
        return interpret(fn, **kwargs)

    return wrap_fn


# Do we need to reimplement apply for every child?
# pylint: disable=abstract-method
class BaseCudaInstructionSet(qml.QubitDevice):
    """Base instruction set for CUDA-Quantum devices"""

    # TODO: Once 0.35 is released, remove -dev suffix.
    pennylane_requires = "0.34"
    version = "0.1.0"
    author = "Xanadu, Inc."

    # There are similar lines of code in possibly
    # all other list of operations supported by devices.
    # At the time of writing, this warning is raised
    # due to similar lines of code in the QJITDevice
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
        "CSWAP",
    ]
    observables = []
    config = Path(__file__).parent / "cuda_quantum.toml"

    def __init__(self, shots=None, wires=None):
        super().__init__(wires=wires, shots=shots)

    def apply(self, operations, **kwargs):
        """Unused"""
        raise NotImplementedError(
            "This device is only supported with `qml.qjit`."
        )  # pragma: no cover


class SoftwareQQPP(BaseCudaInstructionSet):
    """Concrete device class for qpp-cpu"""

    formal_name = "SoftwareQ q++ simulator"
    short_name = "softwareq.qpp"

    @property
    def name(self):
        """Target name"""
        return "qpp-cpu"


class NvidiaCuStateVec(BaseCudaInstructionSet):
    """Concrete device class for CuStateVec"""

    formal_name = "CuStateVec"
    short_name = "nvidia.custatevec"

    def __init__(self, shots=None, wires=None, multi_gpu=False):  # pragma: no cover
        self.multi_gpu = multi_gpu
        super().__init__(wires=wires, shots=shots)

    @property
    def name(self):  # pragma: no cover
        """Target name"""
        option = "-mgpu" if self.multi_gpu else ""
        return f"nvidia{option}"


class NvidiaCuTensorNet(BaseCudaInstructionSet):
    """Concrete device class for CuTensorNet"""

    formal_name = "CuTensorNet"
    short_name = "nvidia.cutensornet"

    def __init__(self, shots=None, wires=None, mps=False):  # pragma: no cover
        self.mps = mps
        super().__init__(wires=wires, shots=shots)

    @property
    def name(self):  # pragma: no cover
        """Target name"""
        option = "-mps" if self.mps else ""
        return f"tensornet{option}"
