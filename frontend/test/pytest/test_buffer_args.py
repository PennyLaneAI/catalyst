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

import pytest

from catalyst import qjit, grad
import pennylane as qml
import jax.numpy as jnp


def f(arg0: int, arg1: int):
    qml.RX(arg0 * jnp.pi, wires=[0])
    return qml.state()


def g(arg0: int, arg1: int, arg2: int):
    qml.RX(arg0 * jnp.pi, wires=[0])
    return qml.state()


@pytest.mark.parametrize(
    "f,params",
    [
        (f, [0, 0]),
        (f, [0, 1]),
        (f, [1, 0]),
        (f, [1, 1]),
        (g, [0, 0, 0]),
        (g, [0, 0, 1]),
        (g, [0, 1, 0]),
        (g, [0, 1, 1]),
        (g, [1, 0, 0]),
        (g, [1, 0, 1]),
        (g, [1, 1, 0]),
        (g, [1, 1, 1]),
    ],
)
def test_buffer_args(f, params):
    device = qml.device("lightning.qubit", wires=1)
    interpreted_fn = qml.QNode(f, device)
    jitted_fn = qjit(interpreted_fn)
    assert jnp.allclose(interpreted_fn(*params), jitted_fn(*params))


class TestReturnValues:
    def test_return_values(self):
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circuit(params):
            qml.SingleExcitation(params[0], wires=[0, 1])
            qml.SingleExcitation(params[1], wires=[0, 2])
            return qml.expval(qml.PauliZ(2))

        @qjit()
        def order1(params):
            diff = grad(circuit, argnum=0)
            h = diff(params)
            return h, params

        @qjit()
        def order2(params):
            diff = grad(circuit, argnum=0)
            h = diff(params)
            return params, h

        data_in = jnp.array([1.0, 4.0])
        result1 = order1(data_in)
        result2 = order2(data_in)
        assert jnp.allclose(result1[0], result2[1]) and jnp.allclose(result1[1], result2[0])

    @pytest.mark.parametrize("dtype", [(jnp.complex128), (jnp.complex64)])
    def test_complex(self, dtype):
        @qjit
        def f():
            return jnp.array(0, dtype=dtype)

        assert jnp.allclose(f(), complex(0, 0))


if __name__ == "__main__":
    pytest.main(["-x", __file__])
