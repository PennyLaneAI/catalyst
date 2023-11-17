# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This test fixes our expectations regarding the JAX dynamic API."""

import numpy as np
import pennylane as qml
import pytest
from jax import numpy as jnp
from numpy import array_equal
from numpy.testing import assert_allclose

from catalyst import qjit, while_loop

DTYPES = [float, int, jnp.float32, jnp.float64, jnp.int8, jnp.int16, "float32", np.float64]
SHAPES = [3, (2, 3, 1), (), jnp.array([2, 1, 3], dtype=int)]


def _assert_equal(a, b):
    """Check that two arrays have exactly the same values and types"""

    assert array_equal(a, b)
    assert a.dtype == b.dtype


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_interpretation(shape, dtype):
    """Test that tensor primitive work in the interpretation mode"""
    # pylint: disable=unnecessary-direct-lambda-call

    _assert_equal((lambda: jnp.zeros(shape, dtype))(), jnp.zeros(shape, dtype=dtype))
    _assert_equal((lambda: jnp.ones(shape, dtype))(), jnp.ones(shape, dtype=dtype))
    _assert_equal((lambda s: jnp.ones(s, dtype))(shape), jnp.ones(shape, dtype=dtype))
    _assert_equal((lambda s: jnp.zeros(s, dtype))(shape), jnp.zeros(shape, dtype=dtype))

    def f(s):
        return jnp.empty(shape=s, dtype=dtype)

    res = f(shape)
    assert_allclose(res.shape, shape)
    assert res.dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_classical_tracing(shape, dtype):
    """Test that tensor primitive work in the classical tracing mode"""

    _assert_equal(qjit(lambda: jnp.zeros(shape, dtype))(), jnp.zeros(shape, dtype=dtype))
    _assert_equal(qjit(lambda: jnp.ones(shape, dtype))(), jnp.ones(shape, dtype=dtype))
    _assert_equal(qjit(lambda s: jnp.ones(s, dtype))(shape), jnp.ones(shape, dtype=dtype))
    _assert_equal(qjit(lambda s: jnp.zeros(s, dtype))(shape), jnp.zeros(shape, dtype=dtype))

    @qjit
    def f(s):
        res = jnp.empty(shape=s, dtype=dtype)
        return res

    res = f(shape)
    assert_allclose(res.shape, shape)
    assert res.dtype == dtype


def test_classical_tracing_2():
    """Test that tensor primitive work in the classical tracing mode, the traced dimention case"""

    @qjit
    def f(x):
        return jnp.ones(shape=[1, x], dtype=int)

    _assert_equal(f(3), jnp.ones((1, 3), dtype=int))


def test_quantum_tracing1():
    """Test that catalyst tensor primitive is compatible with quantum tracing mode"""

    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(shape):
        i = 0
        a = jnp.ones(shape, dtype=float)

        @while_loop(lambda _, i: i < 3)
        def loop(a, i):
            qml.PauliX(wires=0)
            a = a + a
            # b = jnp.ones(shape, dtype=float)
            i += 1
            return (b, i)

        a2, _ = loop(a, i)
        return a2

    result = f([2, 3])
    expected = jnp.ones([2, 3]) * 8
    print("MLIR")
    print(f.mlir)
    print("MLIR")
    assert array_equal(result, expected)


def test_quantum_tracing2():
    """Test that catalyst tensor primitive is compatible with quantum tracing mode"""

    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(x,y):
        i = 0
        a = jnp.ones((x,y+1), dtype=float)

        @while_loop(lambda _, i: i < 3)
        def loop(a, i):
            qml.PauliX(wires=0)
            # a = a + a
            b = jnp.ones((x,y+1), dtype=float)
            i += 1
            return (b, i)

        a2, _ = loop(a, i)
        return a2

    result = f(2, 3)
    print(result)
    expected = jnp.ones((2, 4))
    print("MLIR")
    print(f.mlir)
    print("MLIR")
    assert array_equal(result, expected)


@pytest.mark.parametrize(
    "bad_shape",
    [
        [[2, 3]],
        [2, 3.0],
        [1, jnp.array(2, dtype=float)],
    ],
)
def test_invalid_shapes(bad_shape):
    """Test the unsupported shape formats"""

    def f():
        return jnp.empty(shape=bad_shape, dtype=int)

    with pytest.raises(
        TypeError, match="Shapes must be 1D sequences of concrete values of integer type"
    ):
        qjit(f)


@pytest.mark.skip(f"Jax does not detect error in this use-case")
def test_invalid_shapes_2():
    """Test the unsupported shape formats"""
    bad_shape = jnp.array([[3, 2]], dtype=int)

    def f():
        return jnp.empty(shape=bad_shape, dtype=int)

    with pytest.raises(TypeError):
        qjit(f)


def test_shapes_type_conversion():
    """Test fixes jax behavior regarding the shape conversions"""

    def f(x):
        return jnp.empty(shape=[2, x], dtype=int)

    assert qjit(f)(3.1).shape == (2, 3)
    assert qjit(f)(4.9).shape == (2, 4)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
