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

import pennylane as qml
import pytest
from pennylane.ops.op_math.controlled import _get_pauli_x_based_ops, _get_special_ops

from catalyst.device.decomposition import catalyst_decomposer
from catalyst.utils.toml import DeviceCapabilities, OperationProperties


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


if __name__ == "__main__":
    pytest.main(["-x", __file__])
