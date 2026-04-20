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

import numpy as np
import pennylane as qp
import pytest
from pennylane.devices.capabilities import DeviceCapabilities, OperatorProperties
from utils import CONFIG_CUSTOM_DEVICE

from catalyst import CompileError, ctrl, qjit
from catalyst.compiler import get_lib_path
from catalyst.device.decomposition import catalyst_decomposer


class TestGateAliases:
    """Test the decomposition of gates wich are in fact supported via aliased or equivalent
    op definitions."""

    special_control_ops = (
        qp.CNOT(wires=[0, 1]),
        qp.Toffoli(wires=[0, 1, 2]),
        qp.MultiControlledX(wires=[1, 2, 0], control_values=[True, False]),
        qp.CZ(wires=[0, 1]),
        qp.CCZ(wires=[0, 1, 2]),
        qp.CY(wires=[0, 1]),
        qp.CSWAP(wires=[0, 1, 2]),
        qp.CH(wires=[0, 1]),
        qp.CRX(0.1, wires=[0, 1]),
        qp.CRY(0.1, wires=[0, 1]),
        qp.CRZ(0.1, wires=[0, 1]),
        qp.CRot(0.1, 0.2, 0.3, wires=[0, 1]),
        qp.ControlledPhaseShift(0.1, wires=[0, 1]),
        qp.ControlledQubitUnitary([[1, 0], [0, 1j]], wires=[1, 0]),
    )
    control_base_ops = (
        qp.PauliX,
        qp.PauliX,
        qp.PauliX,
        qp.PauliZ,
        qp.PauliZ,
        qp.PauliY,
        qp.SWAP,
        qp.Hadamard,
        qp.RX,
        qp.RY,
        qp.RZ,
        qp.Rot,
        qp.PhaseShift,
        qp.QubitUnitary,
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
        assert type(decomp[0]) is qp.ops.ControlledOp
        assert type(decomp[0].base) is base


class NoUnitaryDevice(qp.devices.Device):
    """Custom device used for testing purposes."""

    config_filepath = CONFIG_CUSTOM_DEVICE

    def __init__(self, shots=None, wires=None):
        super().__init__(wires=wires, shots=shots)
        self.qjit_capabilities = self.capabilities

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


NoUnitaryDevice.capabilities.operations.pop("QubitUnitary")


class TestControlledDecomposition:
    """Test behaviour around the decomposition of the `Controlled` class."""

    def test_no_matrix(self, backend):
        """Test that controlling an operation without a matrix method raises an error."""

        dev = qp.device(backend, wires=4)

        class OpWithNoMatrix(qp.operation.Operation):
            """Op without a matrix or decomp"""

        @qp.qnode(dev)
        def f():
            ctrl(OpWithNoMatrix(wires=[0, 1]), control=[2, 3])
            return qp.probs()

        with pytest.raises(CompileError, match="not supported with catalyst on this device"):
            qjit(f, target="jaxpr")

    def test_no_unitary_support(self):
        """Test that unknown controlled operations without QubitUnitary support raise an error."""

        class UnknownOp(qp.operation.Operation):
            """An unknown operation"""

            def matrix(self):
                """The matrix"""
                return np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.complex128,
                )

        dev = NoUnitaryDevice(wires=4)

        @qp.set_shots(4)
        @qp.qnode(dev)
        def f():
            ctrl(UnknownOp(wires=[0, 1]), control=[2, 3])
            return qp.probs()

        with pytest.raises(CompileError, match="not supported with catalyst on this device"):
            qjit(f, target="jaxpr")


if __name__ == "__main__":
    pytest.main(["-x", __file__])
