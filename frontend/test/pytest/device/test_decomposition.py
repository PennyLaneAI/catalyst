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

import platform
from copy import deepcopy

import numpy as np
import pennylane as qml
import pytest
from pennylane.devices.capabilities import DeviceCapabilities, OperatorProperties

from catalyst import CompileError, ctrl, qjit
from catalyst.compiler import get_lib_path
from catalyst.device import get_device_capabilities
from catalyst.device.decomposition import catalyst_decomposer


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
            operations={base.__name__: OperatorProperties(controllable=True)}
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

    def test_no_unitary_support(self):
        """Test that unknown controlled operations without QubitUnitary support raise an error."""

        class UnknownOp(qml.operation.Operation):
            num_wires = qml.operation.AnyWires

            def matrix(self):
                return np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.complex128,
                )

        dev = get_custom_device_without(4, {"QubitUnitary"})

        @qml.qnode(dev)
        def f():
            ctrl(UnknownOp(wires=[0, 1]), control=[2, 3])
            return qml.probs()

        with pytest.raises(CompileError, match="not supported with catalyst on this device"):
            qjit(f, target="jaxpr")


def get_custom_device_without(num_wires, discards=frozenset(), force_matrix=frozenset()):
    """Generate a custom device without gates in discards."""

    class CustomDevice(qml.devices.Device):
        """Custom Gate Set Device"""

        name = "Custom Device"
        pennylane_requires = "0.35.0"
        version = "0.0.2"
        author = "Tester"

        lightning_device = qml.device("lightning.qubit", wires=0)

        config = None
        backend_name = "default"
        backend_lib = "default"
        backend_kwargs = {}

        def __init__(self, shots=None, wires=None):
            super().__init__(wires=wires, shots=shots)
            lightning_capabilities = get_device_capabilities(self.lightning_device)
            custom_capabilities = deepcopy(lightning_capabilities)
            for gate in discards:
                custom_capabilities.native_ops.pop(gate, None)
                custom_capabilities.to_decomp_ops.pop(gate, None)
                custom_capabilities.to_matrix_ops.pop(gate, None)
            for gate in force_matrix:
                custom_capabilities.native_ops.pop(gate, None)
                custom_capabilities.to_decomp_ops.pop(gate, None)
                custom_capabilities.to_matrix_ops[gate] = OperationProperties(False, False, False)
            self.qjit_capabilities = custom_capabilities

        def apply(self, operations, **kwargs):
            """Unused"""
            raise RuntimeError("Only C/C++ interface is defined")

        @staticmethod
        def get_c_interface():
            """Returns a tuple consisting of the device name, and
            the location to the shared object with the C/C++ device implementation.
            """
            system_extension = ".dylib" if platform.system() == "Darwin" else ".so"
            lib_path = (
                get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/librtd_null_qubit" + system_extension
            )
            return "NullQubit", lib_path

        def execute(self, circuits, execution_config):
            """Execution."""
            return circuits, execution_config

    return CustomDevice(wires=num_wires)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
