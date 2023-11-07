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

"""Test integration for the lowering of catalyst.tensor_init."""

import numpy as np
import pennylane as qml
import pytest
from jax import numpy as jnp
from numpy import array_equal
from numpy.testing import assert_allclose

from catalyst import empty, ones, qjit, zeros

DTYPES = [float, int, jnp.float32, jnp.float64, jnp.int8, jnp.int16, "float32", np.float64]
SHAPES = [3, (2, 3, 1), (), jnp.array([2, 1], dtype=int)]


def _assert_equal(a, b):
    """Check that two arrays have exactly the same values and types"""

    assert array_equal(a, b)
    assert a.dtype == b.dtype


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_interpretation(shape, dtype):
    """Test that tensor primitive work in the interpretation mode"""
    # pylint: disable=unnecessary-direct-lambda-call

    _assert_equal((lambda: zeros(shape, dtype))(), jnp.zeros(shape, dtype=dtype))
    _assert_equal((lambda: ones(shape, dtype))(), jnp.ones(shape, dtype=dtype))
    _assert_equal((lambda s: ones(s, dtype))(shape), jnp.ones(shape, dtype=dtype))
    _assert_equal((lambda s: zeros(s, dtype))(shape), jnp.zeros(shape, dtype=dtype))

    def f(s):
        return empty(shape=s, dtype=dtype)

    res = f(shape)
    assert_allclose(res.shape, shape)
    assert res.dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_classical_tracing(shape, dtype):
    """Test that tensor primitive work in the classical tracing mode"""

    _assert_equal(qjit(lambda: zeros(shape, dtype))(), jnp.zeros(shape, dtype=dtype))
    _assert_equal(qjit(lambda: ones(shape, dtype))(), jnp.ones(shape, dtype=dtype))
    _assert_equal(qjit(lambda s: ones(s, dtype))(shape), jnp.ones(shape, dtype=dtype))
    _assert_equal(qjit(lambda s: zeros(s, dtype))(shape), jnp.zeros(shape, dtype=dtype))

    @qjit
    def f(s):
        return empty(shape=s, dtype=dtype)

    res = f(shape)
    assert_allclose(res.shape, shape)
    assert res.dtype == dtype


def test_quantum_tracing():
    """Test that catalyst tensor primitive is compatible with quantum tracing mode"""

    @qjit(autograph=True)
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(shape):
        i = 0
        a = ones(shape, dtype=float)
        while i < 3:
            a = a + a
            qml.PauliX(wires=0)
            i += 1
        return a

    result = f([2, 3])
    expected = jnp.ones([2, 3]) * 8
    assert array_equal(result, expected)


def test_unsupported():
    """Test the unsupported initializer error raising on invalid dtypes"""

    def f():
        return ones(shape=[2, 3], dtype=bool)

    with pytest.raises(
        ValueError,
        match="Unsupported initializer",
    ):
        qjit(f)


def test_invalid_shape_of_shape():
    """Test the unsupported shape format"""

    def f():
        return empty(shape=[[2, 3]], dtype=int)

    with pytest.raises(
        ValueError,
        match="The shape is expected to have rank one and contain integers",
    ):
        qjit(f)


def test_invalid_dtype_of_shape():
    """Test the unsupported shape format"""

    def f():
        return empty(shape=[[2.0, 3.0]], dtype=float)

    with pytest.raises(
        ValueError,
        match="The shape is expected to have rank one and contain integers",
    ):
        qjit(f)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
