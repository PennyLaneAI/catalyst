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

import pennylane as qml

from catalyst import pennylane_extensions
from catalyst.cuda.catalyst_to_cuda_interpreter import interpret
from catalyst.utils.contexts import EvaluationContext


def qjit(fn=None, **kwargs):
    """Entry point to qjit_cuda."""
    return qjit_cuda(fn, **kwargs)  # pragma: no cover


def qjit_cuda(fn=None, **kwargs):
    """Wrapper around QJIT for CUDA-quantum."""

    if kwargs.get("target", "binary") == "binary":
        # Catalyst uses binary as a default.
        # If use are using catalyst's qjit
        # then, let's override it with cuda_quantum's default.
        kwargs["target"] = "qpp-cpu"

    target = kwargs.get("target", "qpp-cpu")
    if target not in {
        "qpp-cpu",
        "nvidia",
        "tensornet",
        "nvidia-mgpu",
        "quantinuum",
        "ionq",
        "IQM",
        "OQC",
    }:
        msg = f"Unsupported target {target}."
        raise ValueError(msg)

    if target != "qpp-cpu":
        msg = f"Unimplemented target {target}."
        raise NotImplementedError(msg)

    if fn is not None:
        return interpret(fn)

    def wrap_fn(fn):
        return interpret(fn)

    return wrap_fn


class CudaQDevice(qml.QubitDevice):
    """CudaQ Device."""

    name = "CudaQ Device"
    short_name = "cudaq"
    # TODO: Once 0.35 is released, remove -dev suffix.
    pennylane_requires = "0.35.0-dev"
    version = "1.0"
    author = "Catalyst authors wrote this wrapper for CUDA Quantum"

    # pylint: disable=duplicate-code
    operations = [
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
    ]
    observables = []
    config = Path(__file__).parent / "cuda.toml"

    def __init__(self, shots=None, wires=None):
        super().__init__(wires=wires, shots=shots)

    def apply(self, operations, **kwargs):
        """Unused"""
        raise RuntimeError("We are not applying operation by operation.")  # pragma: no cover
