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
"""Tests for qml.specs() Catalyst integration"""

import pennylane as qml
import pytest
from jax import numpy as jnp

from catalyst import qjit

# pylint:disable = protected-access,attribute-defined-outside-init


# TODO: Remove this method once feature pairty has been reached, and instead use `==` directly
def check_specs_same(specs1, specs2, skip_measurements=False):
    """Check that two specs dictionaries are the same."""
    assert specs1["device_name"] == specs2["device_name"]
    assert specs1["num_device_wires"] == specs2["num_device_wires"]
    assert specs1["shots"] == specs2["shots"]

    assert type(specs1["resources"]) == type(specs2["resources"])

    if not isinstance(specs1["resources"], dict):
        all_res1 = {None: specs1["resources"]}
        all_res2 = {None: specs2["resources"]}

    else:
        all_res1 = specs1["resources"]
        all_res2 = specs2["resources"]

    for res1, res2 in zip(all_res1.values(), all_res2.values()):
        assert res1.gate_types == res2.gate_types
        assert res1.gate_sizes == res2.gate_sizes

        # TODO: Measurements are not yet supported in Catalyst device-level specs
        if not skip_measurements:
            assert res1.measurements == res2.measurements

        assert res1.num_allocs == res2.num_allocs
        assert res1.depth == res2.depth
        assert res1.num_gates == res2.num_gates

class TestDeviceLevelSpecs:
    """Test qml.specs() at device level"""

    def test_simple(self):
        """Test a simple case of qml.specs() against PennyLane"""

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        pl_specs = qml.specs(circuit, level="device")()
        cat_specs = qml.specs(qjit(circuit), level="device")()

        assert cat_specs["device_name"] == "lightning.qubit"
        check_specs_same(pl_specs, cat_specs, skip_measurements=True)


    def test_complex(self):
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

        pl_specs = qml.specs(circuit, level="device")()
        cat_specs = qml.specs(qjit(circuit), level="device")()

        assert cat_specs["device_name"] == "lightning.qubit"

        # Catalyst will handle Adjoint(PauliY) == PauliY
        assert "CY" in cat_specs["resources"].gate_types
        cat_specs["resources"].gate_types["C(Adjoint(PauliY))"] = cat_specs["resources"].gate_types[
            "CY"
        ]
        del cat_specs["resources"].gate_types["CY"]

        check_specs_same(pl_specs, cat_specs, skip_measurements=True)

class TestPassByPassSpecs:
    """Test qml.specs() pass-by-pass specs"""

    pass

if __name__ == "__main__":
    pytest.main(["-x", __file__])
