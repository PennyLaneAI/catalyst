# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test for qml.specs() Catalyst integration"""
import pennylane as qml
import pytest
from jax import numpy as jnp

from catalyst import qjit

# pylint:disable = protected-access,attribute-defined-outside-init


@pytest.mark.parametrize("level", ["device"])
def test_simple(level):
    """Test a simple case of qml.specs() against PennyLane"""

    dev = qml.device("lightning.qubit", wires=1)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        return qml.expval(qml.PauliZ(0))

    pl_specs = qml.specs(circuit, level=level)()
    cat_specs = qml.specs(qjit(circuit), level=level)()

    assert pl_specs["resources"].num_wires == cat_specs["resources"].num_wires
    assert pl_specs["resources"].num_gates == cat_specs["resources"].num_gates
    assert pl_specs["resources"].depth == cat_specs["resources"].depth
    assert pl_specs["device_name"] == cat_specs["device_name"]

    assert len(cat_specs["resources"].gate_types) == len(pl_specs["resources"].gate_types)
    for gate, count in pl_specs["resources"].gate_types.items():
        assert gate in cat_specs["resources"].gate_types
        assert count == cat_specs["resources"].gate_types[gate]

    assert len(cat_specs["resources"].gate_sizes) == len(pl_specs["resources"].gate_sizes)
    for gate, count in pl_specs["resources"].gate_sizes.items():
        assert gate in cat_specs["resources"].gate_sizes
        assert count == cat_specs["resources"].gate_sizes[gate]


@pytest.mark.parametrize("level", ["device"])
def test_complex(level):
    """Test a complex case of qml.specs() against PennyLane"""

    dev = qml.device("lightning.qubit", wires=4)
    U = 1 / jnp.sqrt(2) * jnp.array([[1, 1], [1, -1]], dtype=jnp.complex128)

    @qml.qnode(dev)
    def circuit():
        qml.PauliX(0)
        qml.adjoint(qml.T)(0)
        qml.ctrl(op=qml.S, control=[1], control_values=[1])(0)
        qml.ctrl(op=qml.S, control=[1, 2], control_values=[1, 0])(0)
        qml.ctrl(op=qml.adjoint(qml.Y), control=[2], control_values=[1])(0)
        qml.CNOT([0, 1])

        qml.QubitUnitary(U, wires=0)
        qml.ControlledQubitUnitary(U, control_values=[1], wires=[1, 0])
        qml.adjoint(qml.QubitUnitary(U, wires=0))
        qml.adjoint(qml.ControlledQubitUnitary(U, control_values=[1, 1], wires=[1, 2, 0]))

        return qml.probs()

    pl_specs = qml.specs(circuit, level=level)()
    cat_specs = qml.specs(qjit(circuit), level=level)()

    assert cat_specs["device_name"] == "lightning.qubit"
    assert pl_specs["resources"].num_wires == cat_specs["resources"].num_wires
    assert pl_specs["resources"].num_gates == cat_specs["resources"].num_gates
    assert pl_specs["resources"].depth == cat_specs["resources"].depth

    # Catalyst level specs should report the number of controls for multi-controlled gates
    assert "2C(S)" in cat_specs["resources"].gate_types
    cat_specs["resources"].gate_types["C(S)"] += cat_specs["resources"].gate_types["2C(S)"]
    del cat_specs["resources"].gate_types["2C(S)"]

    # Catalyst will handle Adjoint(PauliY) == PauliY
    assert "CY" in cat_specs["resources"].gate_types
    cat_specs["resources"].gate_types["C(Adjoint(PauliY))"] += cat_specs["resources"].gate_types[
        "CY"
    ]
    del cat_specs["resources"].gate_types["CY"]

    assert len(cat_specs["resources"].gate_types) == len(pl_specs["resources"].gate_types)
    for gate, count in pl_specs["resources"].gate_types.items():
        assert gate in cat_specs["resources"].gate_types
        assert count == cat_specs["resources"].gate_types[gate]

    assert len(cat_specs["resources"].gate_sizes) == len(pl_specs["resources"].gate_sizes)
    for gate, count in pl_specs["resources"].gate_sizes.items():
        assert gate in cat_specs["resources"].gate_sizes
        assert count == cat_specs["resources"].gate_sizes[gate]


if __name__ == "__main__":
    pytest.main(["-x", __file__])
