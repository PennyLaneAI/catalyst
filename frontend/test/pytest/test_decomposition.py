# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import tempfile

import pennylane as qml
import pytest
from jax import numpy as jnp

from catalyst import CompileError, ctrl, measure, qjit


class CustomDevice(qml.QubitDevice):
    """Custom Gate Set Device"""

    name = "Custom Device"
    short_name = "lightning.qubit"
    pennylane_requires = "0.35.0"
    version = "0.0.2"
    author = "Tester"

    lightning_device = qml.device("lightning.qubit", wires=0)
    operations = lightning_device.operations.copy() - {
        "MultiControlledX",
        "Rot",
        "S",
        "C(Rot)",
        "C(S)",
    }
    observables = lightning_device.observables.copy()

    config = None
    backend_name = "default"
    backend_lib = "default"
    backend_kwargs = {}

    def __init__(self, shots=None, wires=None):
        super().__init__(wires=wires, shots=shots)
        self.toml_file = None

    def apply(self, operations, **kwargs):
        """Unused"""
        raise RuntimeError("Only C/C++ interface is defined")

    def __enter__(self, *args, **kwargs):
        lightning_toml = self.lightning_device.config
        with open(lightning_toml, mode="r", encoding="UTF-8") as f:
            toml_contents = f.readlines()

        # TODO: update once schema 2 is merged
        updated_toml_contents = []
        for line in toml_contents:
            if re.match(r"^MultiControlledX\s", line):
                continue
            if re.match(r"^Rot\s", line):
                continue
            if re.match(r"^S\s", line):
                continue

            updated_toml_contents.append(line)

        self.toml_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        self.toml_file.writelines(updated_toml_contents)
        self.toml_file.close()  # close for now without deleting

        self.config = self.toml_file.name
        return self

    def __exit__(self, *args, **kwargs):
        os.unlink(self.toml_file.name)
        self.config = None


@pytest.mark.parametrize("param,expected", [(0.0, True), (jnp.pi, False)])
def test_decomposition(param, expected):
    with CustomDevice(wires=2) as dev:

        @qjit
        @qml.qnode(dev)
        def mid_circuit(x: float):
            qml.Hadamard(wires=0)
            qml.Rot(0, 0, x, wires=0)
            qml.Hadamard(wires=0)
            m = measure(wires=0)
            b = m ^ 0x1
            qml.Hadamard(wires=1)
            qml.Rot(0, 0, b * jnp.pi, wires=1)
            qml.Hadamard(wires=1)
            return measure(wires=1)

        assert mid_circuit(param) == expected


class TestControlledDecomposition:
    """Test behaviour around the decomposition of the `Controlled` class."""

    def test_no_matrix(self, backend):
        """Test that controlling an operation without a matrix method raises an error."""
        dev = qml.device(backend, wires=4)

        class OpWithNoMatrix(qml.operation.Operation):
            num_wires = qml.operation.AnyWires

            def matrix(self):
                raise NotImplementedError()

        @qml.qnode(dev)
        def f():
            ctrl(OpWithNoMatrix(wires=[0, 1]), control=[2, 3])
            return qml.probs()

        with pytest.raises(CompileError, match="could not be decomposed, it might be unsupported."):
            qjit(f, target="jaxpr")


if __name__ == "__main__":
    pytest.main(["-x", __file__])
