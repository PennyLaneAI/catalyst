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

import jax.numpy as jnp
import pennylane as qml
import pytest

from catalyst import qjit

ops = [
    qml.Identity(wires=0),
    qml.PauliX(wires=1),
    qml.PauliY(wires=2),
    qml.PauliZ(wires=0),
    qml.Hadamard(wires=0),
    qml.S(wires=0),
    qml.T(wires=0),
    qml.CNOT(wires=[0, 1]),
    qml.CY(wires=[0, 1]),
    qml.CZ(wires=[0, 1]),
    qml.SWAP(wires=[0, 1]),
    qml.CSWAP(wires=[0, 1, 2]),
    qml.QubitUnitary(
        1 / jnp.sqrt(2) * jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex), wires=0
    ),
    qml.QubitUnitary(
        jnp.array(
            [
                [0.99500417 - 0.09983342j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.99500417 + 0.09983342j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.99500417 + 0.09983342j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.99500417 - 0.09983342j],
            ]
        ),
        wires=[1, 2],
    ),
    qml.Rot(0.6, 0.3, 0.9, wires=0),
    qml.RX(0.6, wires=0),
    qml.RY(0.6, wires=1),
    qml.RZ(0.6, wires=2),
    qml.RZ(0.6, wires=0),
    qml.RY(0.6, wires=1),
    qml.RX(0.6, wires=2),
    qml.IsingXX(0.6, wires=[0, 1]),
    qml.IsingXX(0.6, wires=[1, 2]),
    qml.IsingYY(0.6, wires=[0, 1]),
    qml.IsingYY(0.6, wires=[1, 2]),
    qml.IsingZZ(0.6, wires=[0, 1]),
    qml.IsingZZ(0.6, wires=[1, 2]),
    qml.CRX(0.6, wires=[0, 1]),
    qml.CRY(0.6, wires=[0, 1]),
    qml.CRZ(0.6, wires=[0, 1]),
    qml.CRX(0.6, wires=[1, 2]),
    qml.CRY(0.6, wires=[1, 2]),
    qml.CRZ(0.6, wires=[1, 2]),
    qml.MultiRZ(0.6, wires=[0, 1, 2, 3]),
    qml.MultiControlledX(wires=[1, 2, 3]),
]


@pytest.mark.parametrize("g", ops)
def test_adjoint(g):
    def circuit():
        qml.Rot(0.3, 0.4, 0.5, wires=0)
        qml.adjoint(g)
        return qml.state()

    result = qjit(qml.qnode(qml.device("lightning.qubit", wires=4))(circuit))()
    expected = qml.qnode(qml.device("default.qubit", 4), interface="jax")(circuit)()

    assert jnp.allclose(result, expected)


@pytest.mark.parametrize("g", ops)
@pytest.mark.parametrize("ctrls", [[4], [4, 5, 6]])
def test_control(g, ctrls):
    def circuit():
        qml.Rot(0.3, 0.4, 0.5, wires=0)
        qml.ctrl(g, control=ctrls)
        return qml.state()

    result = qjit(qml.qnode(qml.device("lightning.qubit", wires=7))(circuit))()
    expected = qml.qnode(qml.device("default.qubit", 7), interface="jax")(circuit)()

    assert jnp.allclose(result, expected)


@pytest.mark.parametrize("g", ops)
@pytest.mark.parametrize("ctrls", [[4], [4, 5, 6]])
def test_control_variable_wires(g, ctrls):
    def circuit(ctrls):
        qml.Rot(0.3, 0.4, 0.5, wires=0)
        qml.ctrl(g, control=ctrls)
        return qml.state()

    result = qjit(qml.qnode(qml.device("lightning.qubit", wires=7))(circuit))(jnp.array(ctrls))
    expected = qml.qnode(qml.device("default.qubit", 7), interface="jax")(circuit)(ctrls)

    assert jnp.allclose(result, expected)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
