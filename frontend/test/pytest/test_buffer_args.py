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
def test_buffer_args(f, params, backend):
    device = qml.device(backend, wires=1)
    interpreted_fn = qml.QNode(f, device)
    jitted_fn = qjit(interpreted_fn)
    assert jnp.allclose(interpreted_fn(*params), jitted_fn(*params))


class TestReturnValues:
    def test_return_values(self, backend):
        @qml.qnode(qml.device(backend, wires=3))
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
        # pylint: disable=missing-function-docstring
        def return_scalar():
            return jnp.array(0, dtype=dtype)

        assert jnp.allclose(return_scalar(), complex(0, 0))

    @pytest.mark.parametrize("dtype", [(jnp.float16)])
    def test_types_which_are_unhandled(self, dtype):
        """Test that there's a nice error message when a function returns an f16."""
        with pytest.raises(TypeError, match="Requested return type is unavailable."):

            @qjit
            # pylint: disable=missing-function-docstring
            def return_scalar():
                return jnp.array(0, dtype=dtype)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
