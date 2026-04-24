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
import pennylane as qp
import pytest

from catalyst import qjit

ops = [
    qp.Identity(wires=0),
    qp.PauliX(wires=1),
    qp.PauliY(wires=2),
    qp.PauliZ(wires=0),
    qp.Hadamard(wires=0),
    qp.S(wires=0),
    qp.T(wires=0),
    qp.CNOT(wires=[0, 1]),
    qp.CY(wires=[0, 1]),
    qp.CZ(wires=[0, 1]),
    qp.SWAP(wires=[0, 1]),
    qp.CSWAP(wires=[0, 1, 2]),
    qp.QubitUnitary(1 / jnp.sqrt(2) * jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex), wires=0),
    qp.QubitUnitary(
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
    qp.Rot(0.6, 0.3, 0.9, wires=0),
    qp.RX(0.6, wires=0),
    qp.RY(0.6, wires=1),
    qp.RZ(0.6, wires=2),
    qp.RZ(0.6, wires=0),
    qp.RY(0.6, wires=1),
    qp.RX(0.6, wires=2),
    qp.IsingXX(0.6, wires=[0, 1]),
    qp.IsingXX(0.6, wires=[1, 2]),
    qp.IsingYY(0.6, wires=[0, 1]),
    qp.IsingYY(0.6, wires=[1, 2]),
    qp.IsingZZ(0.6, wires=[0, 1]),
    qp.IsingZZ(0.6, wires=[1, 2]),
    qp.CRX(0.6, wires=[0, 1]),
    qp.CRY(0.6, wires=[0, 1]),
    qp.CRZ(0.6, wires=[0, 1]),
    qp.CRX(0.6, wires=[1, 2]),
    qp.CRY(0.6, wires=[1, 2]),
    qp.CRZ(0.6, wires=[1, 2]),
    qp.MultiRZ(0.6, wires=[0, 1, 2, 3]),
    qp.MultiControlledX(wires=[1, 2, 3]),
    qp.PCPhase(0.6, dim=0, wires=[0, 1, 2, 3]),
    qp.PCPhase(0.6, dim=2, wires=[0, 1, 2, 3]),
]


@pytest.mark.parametrize("g", ops)
def test_adjoint(g):
    def circuit():
        qp.Rot(0.3, 0.4, 0.5, wires=0)
        qp.adjoint(g)
        return qp.state()

    result = qjit(qp.qnode(qp.device("lightning.qubit", wires=4))(circuit))()
    expected = qp.qnode(qp.device("default.qubit", 4), interface="jax")(circuit)()

    assert jnp.allclose(result, expected)


@pytest.mark.parametrize("g", ops)
@pytest.mark.parametrize("ctrls", [[4], [4, 5, 6]])
def test_control(g, ctrls):
    def circuit():
        qp.Rot(0.3, 0.4, 0.5, wires=0)
        qp.ctrl(g, control=ctrls)
        return qp.state()

    result = qjit(qp.qnode(qp.device("lightning.qubit", wires=7))(circuit))()
    expected = qp.qnode(qp.device("default.qubit", 7), interface="jax")(circuit)()

    assert jnp.allclose(result, expected)


@pytest.mark.parametrize("g", ops)
@pytest.mark.parametrize("ctrls", [[4], [4, 5, 6]])
def test_control_variable_wires(g, ctrls):
    def circuit(ctrls):
        qp.Rot(0.3, 0.4, 0.5, wires=0)
        qp.ctrl(g, control=ctrls)
        return qp.state()

    result = qjit(qp.qnode(qp.device("lightning.qubit", wires=7))(circuit))(jnp.array(ctrls))
    expected = qp.qnode(qp.device("default.qubit", 7), interface="jax")(circuit)(ctrls)

    assert jnp.allclose(result, expected)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
