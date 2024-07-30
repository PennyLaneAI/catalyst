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

"""Unit test module for catalyst/device/decomposition.py"""

from copy import deepcopy

import pennylane as qml
import pytest

from catalyst import CompileError, ctrl, qjit
from catalyst.device import get_device_capabilities
from catalyst.device.decomposition import catalyst_decomposer
from catalyst.utils.toml import (
    DeviceCapabilities,
    OperationProperties,
    ProgramFeatures,
    pennylane_operation_set,
)


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
        program_features = ProgramFeatures(shots_present=bool(self.shots))
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


class TestGateAliases:
    """Test the decomposition of gates wich are in fact supported via aliased or equivalent
    op definitions."""

    special_control_ops = (
        qml.CNOT([0, 1]),
        qml.Toffoli([0, 1, 2]),
        qml.MultiControlledX([1, 2], 0, [True, False]),
        qml.CZ([0, 1]),
        qml.CCZ([0, 1, 2]),
        qml.CY([0, 1]),
        qml.CSWAP([0, 1, 2]),
        qml.CH([0, 1]),
        qml.CRX(0.1, [0, 1]),
        qml.CRY(0.1, [0, 1]),
        qml.CRZ(0.1, [0, 1]),
        qml.CRot(0.1, 0.2, 0.3, [0, 1]),
        qml.ControlledPhaseShift(0.1, [0, 1]),
        qml.ControlledQubitUnitary([[1, 0], [0, 1j]], 1, 0),
    )
    control_base_ops = (
        qml.PauliX,
        qml.PauliX,
        qml.PauliX,
        qml.PauliZ,
        qml.PauliZ,
        qml.PauliY,
        qml.SWAP,
        qml.Hadamard,
        qml.RX,
        qml.RY,
        qml.RZ,
        qml.Rot,
        qml.PhaseShift,
        qml.QubitUnitary,
    )
    assert len(special_control_ops) == len(control_base_ops)

    @pytest.mark.parametrize("gate, base", zip(special_control_ops, control_base_ops))
    def test_control_aliases(self, gate, base):
        """Test the decomposition of specialized control operations."""

        capabilities = DeviceCapabilities(
            native_ops={base.__name__: OperationProperties(controllable=True)}
        )
        decomp = catalyst_decomposer(gate, capabilities)

        assert len(decomp) == 1
        assert type(decomp[0]) is qml.ops.ControlledOp
        assert type(decomp[0].base) is base


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
