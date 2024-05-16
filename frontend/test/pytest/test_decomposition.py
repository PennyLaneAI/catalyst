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

from copy import deepcopy

import pennylane as qml
import pytest
from jax import numpy as jnp

from catalyst import CompileError, ctrl, measure, qjit
from catalyst.utils.toml import pennylane_operation_set
from catalyst.utils.toml import ProgramFeatures, get_device_capabilities


class CustomDevice(qml.QubitDevice):
    """Custom Gate Set Device"""

    name = "Custom Device"
    short_name = "lightning.qubit"
    pennylane_requires = "0.35.0"
    version = "0.0.2"
    author = "Tester"

    lightning_device = qml.device("lightning.qubit", wires=0)

    backend_name = "default"
    backend_lib = "default"
    backend_kwargs = {}

    def __init__(self, shots=None, wires=None):
        super().__init__(wires=wires, shots=shots)
        program_features = ProgramFeatures(shots_present=self.shots is not None)
        lightning_capabilities = get_device_capabilities(self.lightning_device, program_features)
        custom_capabilities = deepcopy(lightning_capabilities)
        custom_capabilities.native_ops.pop("Rot")
        custom_capabilities.native_ops.pop("S")
        custom_capabilities.to_decomp_ops.pop("MultiControlledX")
        self.qjit_capabilities = custom_capabilities

    def apply(self, operations, **kwargs):
        """Unused"""
        raise RuntimeError("Only C/C++ interface is defined")

    @property
    def operations(self):
        """Get PennyLane operations."""
        return (
            pennylane_operation_set(self.qjit_capabilities.native_ops)
            | pennylane_operation_set(self.qjit_capabilities.to_decomp_ops)
            | pennylane_operation_set(self.qjit_capabilities.to_matrix_ops)
        )

    @property
    def observables(self):
        """Get PennyLane observables."""
        return pennylane_operation_set(self.qjit_capabilities.native_obs)


@pytest.mark.parametrize("param,expected", [(0.0, True), (jnp.pi, False)])
def test_decomposition(param, expected):
    dev = CustomDevice(wires=2)

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
