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
from jax import scipy as jsp
from numpy import array_equal
from numpy.testing import assert_allclose

from catalyst import qjit, while_loop

DTYPES = [float, int, jnp.float32, jnp.float64, jnp.int8, jnp.int16, "float32", np.float64]
SHAPES = [3, (2, 3, 1), (), jnp.array([2, 1, 3], dtype=int)]


def assert_array_and_dtype_equal(a, b):
    """Check that two arrays have exactly the same values and types"""

    assert array_equal(a, b)
    assert a.dtype == b.dtype


def test_qjit_abstracted_axes():
    """Test that qjit accepts dynamical arguments."""

    @qjit(abstracted_axes={0: "n"})
    def identity(a):
        return a

    param = jnp.array([1, 2, 3])
    result = identity(param)
    assert_array_and_dtype_equal(param, result)
    assert "tensor<?xi64>" in identity.mlir, identity.mlir


def test_qnode_abstracted_axis():
    """Test that qnode accepts dynamical arguments."""

    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def circuit(a):
        return a

    @qjit(abstracted_axes={0: "n"})
    def identity(a):
        return circuit(a)

    param = jnp.array([1, 2, 3])
    result = identity(param)

    assert_array_and_dtype_equal(param, result)
    assert "tensor<?xi64>" in identity.mlir, identity.mlir


def test_qnode_dynamic_structured_args():
    """Test that qnode accepts dynamically-shaped structured args"""

    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def circuit(a_b):
        return a_b[0] + a_b[1]

    @qjit(abstracted_axes={0: "n"})
    def func(a_b):
        return circuit(a_b)

    param = jnp.array([1, 2, 3])
    c = func((param, param))
    assert_array_and_dtype_equal(c, param + param)
    assert "tensor<?xi64>" in func.mlir, func.mlir


def test_qnode_dynamic_structured_results():
    """Test that qnode returns dynamically-shaped results"""

    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def circuit(a):
        return (
            jnp.ones((a + 1,)),
            jnp.ones(
                (a + 2),
            ),
        )

    @qjit
    def func(a):
        return circuit(a)

    a, b = func(3)
    assert_allclose(a, jnp.ones((4,)))
    assert_allclose(b, jnp.ones((5,)))
    assert "tensor<?xf64>" in func.mlir, func.mlir


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_classical_tracing_init(shape, dtype):
    """Test that tensor primitive work in the classical tracing mode"""

    assert_array_and_dtype_equal(
        qjit(lambda: jnp.zeros(shape, dtype))(), jnp.zeros(shape, dtype=dtype)
    )
    assert_array_and_dtype_equal(
        qjit(lambda: jnp.ones(shape, dtype))(), jnp.ones(shape, dtype=dtype)
    )
    assert_array_and_dtype_equal(
        qjit(lambda s: jnp.ones(s, dtype))(shape), jnp.ones(shape, dtype=dtype)
    )
    assert_array_and_dtype_equal(
        qjit(lambda s: jnp.zeros(s, dtype))(shape), jnp.zeros(shape, dtype=dtype)
    )

    @qjit
    def f(s):
        res = jnp.empty(shape=s, dtype=dtype)
        return res

    res = f(shape)
    assert_allclose(res.shape, shape)
    assert res.dtype == dtype


@pytest.mark.parametrize(
    "op",
    [
        jnp.sin,
        jnp.abs,
    ],
)
def test_classical_tracing_unary_ops(op):
    """Test that tensor primitives work with basic unary operations"""

    shape = (3, 4)
    dtype = complex

    @qjit
    def f(s):
        return op(jnp.ones(s, dtype))

    assert_array_and_dtype_equal(f(shape), op(jnp.ones(shape, dtype)))


@pytest.mark.parametrize(
    "op",
    [
        (lambda x, y: x + y),
        (lambda x, y: x - y),
        (lambda x, y: x * y),
        (lambda x, y: x / y),
    ],
)
def test_classical_tracing_binary_ops(op):
    """Test that tensor primitives work with basic binary operations"""

    shape = (3, 4)
    dtype = complex

    @qjit
    def f(s):
        return op(jnp.ones(s, dtype), jnp.ones(s, dtype))

    assert_array_and_dtype_equal(f(shape), op(jnp.ones(shape, dtype), jnp.ones(shape, dtype)))


@pytest.mark.xfail(reason="A bug in Jax/dynamic API")
def test_classical_tracing_binary_ops_3D():
    """Test that tensor primitives work with basic binary operations on 3D arrays"""
    # TODO: Merge with the binary operations test after fixing
    # pylint: disable=unnecessary-lambda-assignment

    shape = (1, 2, 3)
    dtype = complex
    op = lambda a, b: a + b

    @qjit
    def f(s):
        return op(jnp.ones(s, dtype), jnp.ones(s, dtype))

    assert_array_and_dtype_equal(f(shape), op(jnp.ones(shape, dtype), jnp.ones(shape, dtype)))


@pytest.mark.xfail(reason="A Jax check at _src/lax/slicing.py:1520")
@pytest.mark.parametrize("shape,idx", [((1, 2, 3), (0, 1, 2)), ((3,), (2,))])
def test_access_dynamic_array_static_index(shape, idx):
    """Test accessing dynamic array elements using static indices"""

    dtype = complex

    @qjit
    def f(s):
        return jnp.ones(s, dtype)[idx]

    assert f(shape) == jnp.ones(shape, dtype)[idx]
    assert f"tensor<{'x'.join(['?']*len(shape))}xcomplex<f64>>" in f.mlir
    assert "gather" in f.mlir


@pytest.mark.xfail(reason="A Jax check at _src/lax/slicing.py:1520")
@pytest.mark.parametrize("shape,idx", [((1, 2, 3), (0, 1, -2)), ((3,), (2,))])
def test_access_dynamic_array_dynamic_index(shape, idx):
    """Test accessing dynamic array elements using dynamic indices"""

    dtype = complex

    @qjit
    def f(s, i):
        return jnp.ones(s, dtype)[i]

    assert f(shape, idx) == jnp.ones(shape, dtype)[idx]
    assert f"tensor<{'x'.join(['?']*len(shape))}xcomplex<f64>>" in f.mlir
    assert "gather" in f.mlir


@pytest.mark.xfail(reason="MLIR is incompatible with our pipeline")
@pytest.mark.parametrize("shape,idx,val", [((1, 2, 3), (0, 1, 2), 1j), ((3,), (2,), 0)])
def test_modify_dynamic_array_dynamic_index(shape, idx, val):
    """Test dynamic array modification using dynamic indices"""

    dtype = complex

    @qjit
    def f(s, i):
        return jnp.ones(s, dtype).at[i].set(val)

    assert_array_and_dtype_equal(f(shape, idx), jnp.ones(shape, dtype).at[idx].set(val))
    assert f"tensor<{'x'.join(['?']*len(shape))}xcomplex<f64>>" in f.mlir
    assert "gather" in f.mlir


@pytest.mark.xfail(reason="Slicing is not supported by JAX?")
def test_slice_dynamic_array_dynamic_index():
    """Test dynamic array modification using dynamic indices"""

    shape = (1, 2, 3)
    dtype = complex

    @qjit
    def f(s):
        return jnp.ones(s, dtype)[0, 1, 0:1]

    assert f(shape) == jnp.ones(shape, dtype)[0, 1, 0:1]
    assert f"tensor<{'x'.join(['?']*len(shape))}xcomplex<f64>>" in f.mlir


def test_classical_tracing_2():
    """Test that tensor primitive work in the classical tracing mode, the traced dimension case"""

    @qjit
    def f(x):
        return jnp.ones(shape=[1, x], dtype=int)

    assert_array_and_dtype_equal(f(3), jnp.ones((1, 3), dtype=int))


@pytest.mark.skip("Dynamic arrays support in quantum control flow is not implemented")
def test_quantum_tracing_1():
    """Test that catalyst tensor primitive is compatible with quantum tracing mode"""

    @qjit
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(shape):
        i = 0
        a = jnp.ones(shape, dtype=float)

        @while_loop(lambda _, i: i < 3)
        def loop(_, i):
            qml.PauliX(wires=0)
            b = jnp.ones(shape, dtype=float)
            i += 1
            return (b, i)

        a2, _ = loop(a, i)
        return a2

    result = f([2, 3])
    expected = jnp.ones([2, 3]) * 8
    assert_array_and_dtype_equal(result, expected)


@pytest.mark.skip("Dynamic arrays support in quantum control flow is not implemented")
def test_quantum_tracing_2():
    """Test that catalyst tensor primitive is compatible with quantum tracing mode"""

    @qjit
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(x, y):
        i = 0
        a = jnp.ones((x, y + 1), dtype=float)

        @while_loop(lambda _, i: i < 3)
        def loop(_, i):
            qml.PauliX(wires=0)
            b = jnp.ones((x, y + 1), dtype=float)
            i += 1
            return (b, i)

        a2, _ = loop(a, i)
        return a2

    result = f(2, 3)
    print(result)
    expected = jnp.ones((2, 4))
    assert_array_and_dtype_equal(result, expected)


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

    with pytest.raises(TypeError, match="Shapes must be 1D sequences of integer scalars"):
        qjit(f)


@pytest.mark.skip("Jax does not detect error in this use-case")
def test_invalid_shapes_2():
    """Test the unsupported shape formats"""
    bad_shape = jnp.array([[3, 2]], dtype=int)

    def f():
        return jnp.empty(shape=bad_shape, dtype=int)

    with pytest.raises(TypeError):
        qjit(f)


def test_accessing_shapes():
    """Test that dynamic tensor shapes are available for calculations"""

    @qjit
    def f(sz):
        a = jnp.ones((sz, sz))
        sa = jnp.array(a.shape)
        return jnp.sum(sa)

    assert f(3) == 6


def test_no_recompilation():
    """Test that the function is not recompiled when changing the argument shape across
    invocations."""

    @qjit(abstracted_axes={0: "n"})
    def i(x):
        return x

    i(jnp.array([1]))
    _id0 = id(i.compiled_function)
    i(jnp.array([1, 1]))
    _id1 = id(i.compiled_function)
    assert _id0 == _id1


def test_array_indexing():
    """Test the support of indexing of dynamically-shaped arrays"""

    @qjit
    def fun(sz, idx):
        r = jnp.ones((sz, 3, sz + 1), dtype=int)
        return r[idx, 2, idx]

    res = fun(5, 2)
    assert res == 1


def test_array_assignment():
    """Test the support of assigning a value to a dynamically-shaped array"""

    @qjit
    def fun(sz, idx, val):
        r = jnp.ones((sz, 3, sz), dtype=int)
        r = r.at[idx, 0, idx].set(val)
        return r

    result = fun(5, 2, 33)
    expected = jnp.ones((5, 3, 5), dtype=int).at[2, 0, 2].set(33)
    assert_array_and_dtype_equal(result, expected)


@pytest.mark.xfail(
    raises=OSError,
    reason="""JAX requires BLAS to be linked with,
    but we don't have it linked.""",
)
def test_expm():
    """Test jax.scipy.linalg.expm"""

    @qjit
    def f1(x):
        return jsp.linalg.expm(-2.0 * x)

    y1 = jnp.array([[0.1, 0.2], [5.3, 1.2]])
    res1 = f1(y1)
    expected1 = jnp.array([[2.0767685, -0.23879551], [-6.32808103, 0.76339319]])

    @qjit
    def f2(x):
        return jsp.linalg.expm(x)

    y2 = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    res2 = f2(y2)
    expected2 = jnp.array([[2.71828183, 0.0], [0.0, 2.71828183]])

    assert_allclose(res1, expected1)
    assert_allclose(res2, expected2)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
