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

"""Test that numerical jax functions produce correct results when compiled with qml.qjit"""

import numpy as np
import pennylane as qml
import pytest
import scipy as sp
from jax import numpy as jnp
from jax import scipy as jsp

from catalyst import qjit


def test_expm_numerical():
    """Test jax.scipy.linalg.expm is numerically correct"""

    """Test floating point numerics with jax.scipy.linalg.expm"""

    @qjit
    def f1(x):
        return jsp.linalg.expm(-2.0 * x)

    y1 = jnp.array([[0.1, 0.2], [5.3, 1.2]])
    res1 = f1(y1)
    # expected1 = jnp.array([[2.0767685, -0.23879551], [-6.32808103, 0.76339319]])
    expected1 = sp.linalg.expm(-2.0 * y1)

    """Test integer numerics with jax.scipy.linalg.expm"""

    @qjit
    def f2(x):
        return jsp.linalg.expm(x)

    y2 = jnp.array([[1, 0], [0, 1]])
    res2 = f2(y2)
    # expected2 = jnp.array([[2.71828183, 0.0], [0.0, 2.71828183]])
    expected2 = sp.linalg.expm(y2)

    """Test complex numerics with jax.scipy.linalg.expm"""
    """
	Note: a common usage pattern in Hamiltonian simulation is 
	   exp(-iHt)
	where H is a (Hermitian) matrix.
	"""

    @qjit
    def f3(x):
        return jsp.linalg.expm(-2j * x)

    y3 = jnp.array([[1, -1j], [1j, -1]])  # This is PauliY + PauliZ
    res3 = f3(y3)
    # expected3 = jnp.array([[-0.95136313-0.21783962j, -0.21783962+0.j],
    #                       [ 0.21783962+0.j, -0.95136313+0.21783962j]])
    expected3 = sp.linalg.expm(-2j * y3)

    """Test an entire quantum workflow with jax.scipy.linag.expm"""
    """Rotate |0> about Bloch x axis for 180 degrees to get |1>"""

    @qjit
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def circuit_expm():
        generator = -1j * jnp.pi * jnp.array([[0, 1], [1, 0]]) / 2
        unitary = jsp.linalg.expm(generator)
        qml.QubitUnitary(unitary, wires=[0])
        return qml.probs()

    res4 = circuit_expm()

    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def circuit_rot():
        qml.RX(np.pi, wires=[0])
        return qml.probs()

    # expected4 = [0,1]
    expected4 = circuit_rot()

    assert np.allclose(res1, expected1)
    assert np.allclose(res2, expected2)
    assert np.allclose(res3, expected3)
    assert np.allclose(res4, expected4)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
