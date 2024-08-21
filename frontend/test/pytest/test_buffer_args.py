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

import jax
import jax.numpy as jnp
import pennylane as qml
import pytest

from catalyst import grad, qjit

# pylint: disable=missing-function-docstring


def f(arg0: int, _arg1: int):
    qml.RX(arg0 * jnp.pi, wires=[0])
    return qml.state()


def g(arg0: int, _arg1: int, _arg2: int):
    qml.RX(arg0 * jnp.pi, wires=[0])
    return qml.state()


@pytest.mark.parametrize(
    "fn,params",
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
def test_buffer_args(fn, params):
    """Test multiple arguments passed to compiled function."""

    device = qml.device("lightning.qubit", wires=1)
    interpreted_fn = qml.QNode(fn, device)
    jitted_fn = qjit(interpreted_fn)
    assert jnp.allclose(interpreted_fn(*params), jitted_fn(*params))


class TestReturnValues:
    """Test return value buffers."""

    def test_return_values(self):
        """Test return values are correctly stored in return buffers."""

        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circuit(params):
            qml.SingleExcitation(params[0], wires=[0, 1])
            qml.SingleExcitation(params[1], wires=[0, 2])
            return qml.expval(qml.PauliZ(2))

        @qjit()
        def order1(params):
            diff = grad(circuit, argnums=0)
            h = diff(params)
            return h[0], params

        @qjit()
        def order2(params):
            diff = grad(circuit, argnums=0)
            h = diff(params)
            return params, h[0]

        data_in = jnp.array([1.0, 4.0])
        result1 = order1(data_in)
        result2 = order2(data_in)

        assert jnp.allclose(result1[0], result2[1]) and jnp.allclose(result1[1], result2[0])

    @pytest.mark.parametrize(
        "dtype",
        [(jnp.complex128), (jnp.complex64), (jnp.float32), (jnp.int8), (jnp.int16), (jnp.int32)],
    )
    def test_return_complex_scalar(self, dtype):
        """Complex scalars take a different path when being returned from the
        compiled function. See `ranked_memref_to_numpy` and `to_numpy` in
        llvm-project/mlir/python/mlir/runtime/np_to_memref.py.

        Also test that we can return all these types if specifically requested by the user in the
        compiled function itself.
        """

        @qjit
        def return_scalar():
            return jnp.array(0, dtype=dtype)

        assert jnp.allclose(return_scalar(), complex(0, 0))

    def test_returns_jax_array(self):
        """Tests that the return value is a jax array"""

        @qjit
        def identity(x):
            return x

        assert isinstance(identity(1.0), jax.Array)

    @pytest.mark.parametrize("dtype", [(jnp.float16)])
    def test_types_which_are_unhandled(self, dtype):
        """Test that there's a nice error message when a function returns an f16."""

        def return_scalar():
            return jnp.array(0, dtype=dtype)

        with pytest.raises(TypeError, match="Requested return type is unavailable."):
            qjit(return_scalar)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
