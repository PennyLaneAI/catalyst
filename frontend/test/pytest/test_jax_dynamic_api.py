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

# pylint: disable=too-many-lines

import numpy as np
import pennylane as qml
import pytest
from jax import numpy as jnp
from numpy import array_equal
from numpy.testing import assert_allclose

from catalyst import cond, for_loop, qjit, while_loop
from catalyst.jax_extras import DShapedArray, ShapedArray
from catalyst.jax_extras.tracing import trace_to_jaxpr
from catalyst.tracing.contexts import EvaluationContext

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
        jnp.cos,
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
    expected = jnp.ones([2, 3])
    assert_array_and_dtype_equal(result, expected)


def test_quantum_tracing_2():
    """Test that catalyst tensor primitive is compatible with quantum tracing mode"""

    @qjit
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(x, y):
        i = 0
        a = jnp.ones((x, y + 1), dtype=float)

        @while_loop(lambda _, i: i < 3, allow_array_resizing=True)
        def loop(_, i):
            qml.PauliX(wires=0)
            b = jnp.ones((x, y + 1), dtype=float)
            i += 1
            return (b, i)

        a2, _ = loop(a, i)
        return a2

    result = f(2, 3)
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


def test_qjit_forloop_identity():
    """Test simple for-loop primitive vs dynamic dimensions"""

    @qjit()
    def f(sz):
        a = jnp.ones([sz], dtype=float)

        @for_loop(0, 10, 2)
        def loop(_, a):
            return a

        a2 = loop(a)
        return a2

    result = f(3)
    expected = jnp.ones(3)
    assert_array_and_dtype_equal(result, expected)


def test_qjit_forloop_capture():
    """Test simple for-loop primitive vs dynamic dimensions"""

    @qjit()
    def f(sz):
        x = jnp.ones([sz], dtype=float)

        @for_loop(0, 3, 1)
        def loop(_, a):
            return a + x

        a2 = loop(x)
        return a2

    result = f(3)
    expected = 4 * jnp.ones(3)
    assert_array_and_dtype_equal(result, expected)


def test_qjit_forloop_shared_indbidx():
    """Test for-loops with shared dynamic input dimensions in classical tracing mode"""

    @qjit()
    def f(sz):
        a = jnp.ones([sz], dtype=float)
        b = jnp.ones([sz], dtype=float)

        @for_loop(0, 10, 2)
        def loop(_, a, b):
            return (a, b)

        a2, b2 = loop(a, b)
        return a2 + b2

    result = f(3)
    expected = 2 * jnp.ones(3)
    assert_array_and_dtype_equal(result, expected)


def test_qjit_forloop_indbidx_outdbidx():
    """Test for-loops with shared dynamic output dimensions in classical tracing mode"""

    @qjit()
    def f(sz):
        a = jnp.ones([sz, 3], dtype=float)
        b = jnp.ones([sz, 3], dtype=float)

        @for_loop(0, 10, 2, allow_array_resizing=True)
        def loop(_i, a, _b):
            b = jnp.ones([sz + 1, 3], dtype=float)
            return (a, b)

        a2, b2 = loop(a, b)
        # import pdb; pdb.set_trace()
        return a2, b2

    res_a, res_b = f(3)
    assert_array_and_dtype_equal(res_a, jnp.ones([3, 3]))
    assert_array_and_dtype_equal(res_b, jnp.ones([4, 3]))


def test_qjit_forloop_index_indbidx():
    """Test for-loops referring loop return new dimension variable."""

    @qjit()
    def f(sz):
        a0 = jnp.ones([sz], dtype=float)

        @for_loop(0, 10, 1, allow_array_resizing=True)
        def loop(i, _):
            return jnp.ones([i], dtype=float)

        a2 = loop(a0)
        assert a2.shape[0] is not sz
        return a2

    res_a = f(3)
    assert_array_and_dtype_equal(res_a, jnp.ones(9))


def test_qjit_forloop_indbidx_const():
    """Test for-loops preserve type information in the presence of a constant."""

    @qjit()
    def f(sz):
        a0 = jnp.ones([sz], dtype=float)

        @for_loop(0, 3, 1)
        def loop(_i, a):
            return a * sz

        a2 = loop(a0)
        assert a2.shape[0] is sz
        return a2

    res_a = f(3)
    assert_array_and_dtype_equal(res_a, jnp.ones(3) * (3**3))


def test_qjit_forloop_shared_dimensions():
    """Test catalyst for-loop primitive's experimental_preserve_dimensions option"""

    @qjit
    def f(sz: int):
        input_a = jnp.ones([sz + 1], dtype=float)
        input_b = jnp.ones([sz + 2], dtype=float)

        @for_loop(0, 10, 1, allow_array_resizing=True)
        def loop(_i, _a, _b):
            return (input_a, input_a)

        outputs = loop(input_b, input_b)
        assert outputs[0].shape[0] is outputs[1].shape[0]
        return outputs

    result = f(3)
    expected = (jnp.ones(4, dtype=float), jnp.ones(4, dtype=float))
    assert_array_and_dtype_equal(result[0], expected[0])
    assert_array_and_dtype_equal(result[1], expected[1])


def test_qnode_forloop_identity():
    """Test simple for-loops with dynamic dimensions while doing quantum tracing."""

    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(sz):
        a = jnp.ones([sz], dtype=float)

        @for_loop(0, 10, 2)
        def loop(_, a):
            return a

        a2 = loop(a)
        return a2

    result = f(3)
    expected = jnp.ones(3)
    assert_array_and_dtype_equal(result, expected)


def test_qnode_forloop_capture():
    """Test simple for-loops with dynamic dimensions while doing quantum tracing."""

    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(sz):
        x = jnp.ones([sz], dtype=float)

        @for_loop(0, 3, 1)
        def loop(_, a):
            return a + x

        a2 = loop(x)
        return a2

    result = f(3)
    expected = 4 * jnp.ones(3)
    assert_array_and_dtype_equal(result, expected)


def test_qnode_forloop_shared_indbidx():
    """Tests that for-loops preserve equality of output dynamic dimensions."""

    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(sz):
        a = jnp.ones([sz], dtype=float)
        b = jnp.ones([sz], dtype=float)

        @for_loop(0, 10, 2)
        def loop(_, a, b):
            return (a, b)

        a2, b2 = loop(a, b)
        return a2 + b2

    result = f(3)
    expected = 2 * jnp.ones(3)
    assert_array_and_dtype_equal(result, expected)


def test_qnode_forloop_indbidx_outdbidx():
    """Test for-loops with mixed input and output dimension variables during the quantum tracing."""

    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(sz):
        a = jnp.ones([sz], dtype=float)
        b = jnp.ones([sz], dtype=float)

        @for_loop(0, 10, 2, allow_array_resizing=True)
        def loop(_i, a, _b):
            b = jnp.ones([sz + 1], dtype=float)
            return (a, b)

        a2, b2 = loop(a, b)
        return a2, b2

    res_a, res_b = f(3)
    assert_array_and_dtype_equal(res_a, jnp.ones(3))
    assert_array_and_dtype_equal(res_b, jnp.ones(4))


def test_qnode_forloop_abstracted_axes():
    """Test for-loops with mixed input and output dimension variables during the quantum tracing.
    Use abstracted_axes as the source of dynamism."""

    @qjit(abstracted_axes={0: "n"})
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(a, b):
        @for_loop(0, 10, 2, allow_array_resizing=True)
        def loop(_i, a, _b):
            b = jnp.ones([a.shape[0] + 1], dtype=float)
            return (a, b)

        a2, b2 = loop(a, b)
        return a2, b2

    a = jnp.ones([3], dtype=float)
    b = jnp.ones([3], dtype=float)
    res_a, res_b = f(a, b)
    assert_array_and_dtype_equal(res_a, jnp.ones(3))
    assert_array_and_dtype_equal(res_b, jnp.ones(4))


def test_qnode_forloop_index_indbidx():
    """Test for-loops referring loop index as a dimension during the quantum tracing."""

    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(sz):
        a = jnp.ones([sz, 3], dtype=float)

        @for_loop(0, 10, 1, allow_array_resizing=True)
        def loop(i, _):
            b = jnp.ones([i, 3], dtype=float)
            return b

        a2 = loop(a)
        return a2

    res_a = f(3)
    assert_array_and_dtype_equal(res_a, jnp.ones([9, 3]))


def test_qnode_whileloop_1():
    """Test that catalyst tensor primitive is compatible with quantum while"""

    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(sz):
        a0 = jnp.ones([sz + 1], dtype=float)

        @while_loop(lambda _, i: i < 3)
        def loop(a, i):
            i += 1
            return (a, i)

        a2, _ = loop(a0, 0)
        return a2

    result = f(3)
    expected = jnp.ones(4)
    assert_array_and_dtype_equal(result, expected)


def test_qnode_whileloop_2():
    """Test that catalyst tensor primitive is compatible with quantum while"""

    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(sz):
        a = jnp.ones([sz + 1], dtype=float)

        @while_loop(lambda _, i: i < 3, allow_array_resizing=True)
        def loop(_, i):
            b = jnp.ones([sz + 1], dtype=float)
            i += 1
            return (b, i)

        a2, _ = loop(a, 0)
        return a2

    result = f(3)
    expected = jnp.ones(4)
    assert_array_and_dtype_equal(result, expected)


def test_qnode_whileloop_capture():
    """Tests that while-loop primitive can capture variables from the outer scope"""

    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(sz):
        x = jnp.ones([sz], dtype=float)

        @while_loop(lambda i, _: i < 3)
        def loop(i, a):
            return i + 1, a + x

        _, a2 = loop(1, x)
        return a2

    result = f(3)
    expected = 3 * jnp.ones(3)
    assert_array_and_dtype_equal(result, expected)


def test_qnode_whileloop_abstracted_axes():
    """Test that catalyst tensor primitive is compatible with quantum while. Use abstracted_axes as
    the source of dynamism."""

    @qjit(abstracted_axes={0: "n"})
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(a, b):
        @while_loop(lambda _a, _b, i: i < 3)
        def loop(a, b, i):
            i += 1
            return (a, b, i)

        a2, b2, _ = loop(a, b, 0)
        return a2 + b2

    a = jnp.ones([3], dtype=float)
    b = jnp.ones([3], dtype=float)
    result = f(a, b)
    expected = 2 * jnp.ones(3)
    assert_array_and_dtype_equal(result, expected)


def test_qnode_whileloop_shared_indbidx():
    """Test that catalyst tensor primitive is compatible with quantum while"""

    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(sz):
        a = jnp.ones([sz], dtype=float)
        b = jnp.ones([sz], dtype=float)

        @while_loop(lambda _a, _b, i: i < 3)
        def loop(a, b, i):
            i += 1
            return (a, b, i)

        a2, b2, _ = loop(a, b, 0)
        return a2 + b2

    result = f(3)
    expected = 2 * jnp.ones(3)
    assert_array_and_dtype_equal(result, expected)


def test_qnode_whileloop_indbidx_outdbidx():
    """Test that catalyst tensor primitive is compatible with quantum while"""

    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(sz):
        a = jnp.ones([sz], dtype=float)
        b = jnp.ones([sz], dtype=float)

        @while_loop(lambda _a, _b, i: i < 3, allow_array_resizing=True)
        def loop(a, _, i):
            b = jnp.ones([sz + 1], dtype=float)
            i += 1
            return (a, b, i)

        a2, b2, _ = loop(a, b, 0)
        return a + a2, b2

    res_a, res_b = f(3)
    assert_array_and_dtype_equal(res_a, 2 * jnp.ones(3))
    assert_array_and_dtype_equal(res_b, jnp.ones(4))


def test_qnode_whileloop_outer():
    """Test that catalyst tensor primitive is compatible with quantum while"""

    @qjit
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(sz):
        a0 = jnp.ones([sz], dtype=float)

        @while_loop(lambda _a, i: i < 3)
        def loop(_a, i):
            i += 1
            return (a0, i)

        a2, _ = loop(a0, 0)
        return a0 + a2

    res_a = f(3)
    assert_array_and_dtype_equal(res_a, 2 * jnp.ones(3))


def test_qjit_whileloop_1():
    """Test that catalyst tensor primitive is compatible with quantum while"""

    @qjit
    def f(sz):
        a = jnp.ones([sz + 1], dtype=float)

        @while_loop(lambda _, i: i < 3, allow_array_resizing=True)
        def loop(_, i):
            b = jnp.ones([sz + 1], dtype=float)
            i += 1
            return (b, i)

        a2, _ = loop(a, 0)
        return a2

    result = f(3)
    expected = jnp.ones(4)
    assert_array_and_dtype_equal(result, expected)


def test_qjit_whileloop_2():
    """Test that catalyst tensor primitive is compatible with quantum while"""

    @qjit()
    def f(sz):
        a = jnp.ones([sz + 1], dtype=float)

        @while_loop(lambda _, i: i < 3, allow_array_resizing=True)
        def loop(_, i):
            b = jnp.ones([sz + 1], dtype=float)
            i += 1
            return (b, i)

        a2, _ = loop(a, 0)
        return a2

    result = f(3)
    expected = jnp.ones(4)
    assert_array_and_dtype_equal(result, expected)


def test_qjit_whileloop_shared_dimensions():
    """Test catalyst while loop primitive's preserve dimensions option"""

    @qjit
    def f(sz: int):
        input_a = jnp.ones([sz + 1], dtype=float)
        input_b = jnp.ones([sz + 2], dtype=float)

        @while_loop(lambda _a, _b, c: c, allow_array_resizing=False)
        def loop(_a, _b, _c):
            return (input_a, input_a, False)

        outputs = loop(input_b, input_b, True)
        assert outputs[0].shape[0] is outputs[1].shape[0]
        return outputs

    result = f(3)
    expected = (jnp.ones(4, dtype=float), jnp.ones(4, dtype=float))
    assert_array_and_dtype_equal(result[0], expected[0])
    assert_array_and_dtype_equal(result[1], expected[1])


def test_qjit_whileloop_shared_indbidx():
    """Test that catalyst tensor primitive is compatible with quantum while"""

    @qjit()
    def f(sz):
        a = jnp.ones([sz], dtype=float)
        b = jnp.ones([sz], dtype=float)

        @while_loop(lambda _a, _b, i: i < 3)
        def loop(a, b, i):
            i += 1
            return (a, b, i)

        a2, b2, _ = loop(a, b, 0)
        return a2 + b2

    result = f(3)
    expected = 2 * jnp.ones(3)
    assert_array_and_dtype_equal(result, expected)


def test_qjit_whileloop_indbidx_outdbidx():
    """Test that catalyst tensor primitive is compatible with quantum while"""

    @qjit()
    def f(sz):
        a0 = jnp.ones([sz], dtype=float)
        b0 = jnp.ones([sz], dtype=float)

        @while_loop(lambda _a, _b, i: i < 3, allow_array_resizing=True)
        def loop(a, _, i):
            b = jnp.ones([sz + 1], dtype=float)
            i += 1
            return (a, b, i)

        a2, b2, _ = loop(a0, b0, 0)
        return a0 + a2, b2

    res_a, res_b = f(3)
    assert_array_and_dtype_equal(res_a, 2 * jnp.ones(3))
    assert_array_and_dtype_equal(res_b, jnp.ones(4))


def test_qjit_whileloop_outer():
    """Test that catalyst tensor primitive is compatible with quantum while"""

    @qjit
    def f(sz):
        a0 = jnp.ones([sz], dtype=float)

        @while_loop(lambda _a, i: i < 3)
        def loop(_a, i):
            i += 1
            return (a0, i)

        a2, _ = loop(a0, 0)
        assert a2.shape[0] is a0.shape[0]
        return a2

    res_a = f(3)
    assert_array_and_dtype_equal(res_a, jnp.ones(3))


def test_qjit_whileloop_capture():
    """Tests that while-loop primitive can capture variables from the outer scope"""

    @qjit()
    def f(sz):
        x = jnp.ones([sz], dtype=float)

        @while_loop(lambda i, _: i < 3)
        def loop(i, a):
            return i + 1, a + x

        _, a2 = loop(1, x)
        return a2

    result = f(3)
    expected = 3 * jnp.ones(3)
    assert_array_and_dtype_equal(result, expected)


def test_qnode_cond_identity():
    """Test that catalyst tensor primitive is compatible with quantum conditional"""

    @qjit
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(flag, sz):
        a = jnp.ones([sz], dtype=float)
        b = jnp.zeros([sz], dtype=float)

        @cond(flag)
        def case():
            return a

        @case.otherwise
        def case():
            return b

        c = case()
        assert c.shape[0] is a.shape[0]
        assert c.shape[0] is b.shape[0]
        return c

    assert_array_and_dtype_equal(f(True, 3), jnp.ones(3))
    assert_array_and_dtype_equal(f(False, 3), jnp.zeros(3))


def test_qnode_cond_abstracted_axes():
    """Test that catalyst tensor primitive is compatible with quantum conditional. Use
    abstracted_axes as the source of dynamism."""

    def f(flag, a, b):
        @qjit(abstracted_axes={0: "n"})
        @qml.qnode(qml.device("lightning.qubit", wires=4))
        def _f(a, b):
            @cond(flag)
            def case():
                return a

            @case.otherwise
            def case():
                return b

            c = case()
            assert c.shape[0] is a.shape[0]
            assert c.shape[0] is b.shape[0]
            return c

        return _f(a, b)

    a = jnp.ones([3], dtype=float)
    b = jnp.zeros([3], dtype=float)
    assert_array_and_dtype_equal(f(True, a, b), jnp.ones(3))
    assert_array_and_dtype_equal(f(False, a, b), jnp.zeros(3))


def test_qnode_cond_capture():
    """Test that catalyst tensor primitive is compatible with quantum conditional"""

    @qjit
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(flag, sz):
        a = jnp.ones([sz, 3], dtype=float)

        @cond(flag)
        def case():
            b = jnp.ones([sz, 3], dtype=float)
            return a + b

        @case.otherwise
        def case():
            b = jnp.zeros([sz, 3], dtype=float)
            return a + b

        c = case()
        return c + a

    assert_array_and_dtype_equal(f(True, 3), 3 * jnp.ones([3, 3]))
    assert_array_and_dtype_equal(f(False, 3), 2 * jnp.ones([3, 3]))


def test_qjit_cond_identity():
    """Test that catalyst tensor primitive is compatible with quantum conditional"""

    @qjit
    def f(flag, sz):
        a = jnp.ones([sz, 3], dtype=float)
        b = jnp.zeros([sz, 3], dtype=float)

        @cond(flag)
        def case():
            return a

        @case.otherwise
        def case():
            return b

        c = case()
        assert c.shape[0] is a.shape[0]
        assert c.shape[0] is b.shape[0]
        return c

    assert_array_and_dtype_equal(f(True, 3), jnp.ones([3, 3]))
    assert_array_and_dtype_equal(f(False, 3), jnp.zeros([3, 3]))


def test_qjit_cond_outdbidx():
    """Test that catalyst tensor primitive is compatible with quantum conditional"""

    @qjit
    def f(flag, sz):
        @cond(flag)
        def case():
            return jnp.ones([sz + 1, 3], dtype=float)

        @case.otherwise
        def case():
            return jnp.zeros([sz + 1, 3], dtype=float)

        return case()

    assert_array_and_dtype_equal(f(True, 3), jnp.ones([4, 3]))
    assert_array_and_dtype_equal(f(False, 3), jnp.zeros([4, 3]))


def test_qjit_cond_capture():
    """Test that catalyst tensor primitive is compatible with quantum conditional"""

    @qjit
    def f(flag, sz):
        a = jnp.ones([sz, 3], dtype=float)

        @cond(flag)
        def case():
            b = jnp.ones([sz, 3], dtype=float)
            return a + b

        @case.otherwise
        def case():
            b = jnp.zeros([sz, 3], dtype=float)
            return a + b

        c = case()
        return c + a

    assert_array_and_dtype_equal(f(True, 3), 3 * jnp.ones([3, 3]))
    assert_array_and_dtype_equal(f(False, 3), 2 * jnp.ones([3, 3]))


def test_trace_to_jaxpr():
    """Test our Jax tracing workaround. The idiomatic Jax would do `jaxpr, tracers, consts =
    trace.frame.to_jaxpr2([r])` which fails with `KeyError` for the below case.
    """
    # pylint: disable=protected-access,unused-variable

    @qjit
    def circuit(sz):
        mode, ctx = EvaluationContext.get_evaluation_mode()

        def f(i, _):
            return i < 3

        with EvaluationContext.frame_tracing_context(ctx) as trace:
            sz2 = trace.full_raise(sz)
            i = trace.new_arg(ShapedArray(shape=[], dtype=jnp.dtype("int64")))
            a = trace.new_arg(DShapedArray(shape=[sz2], dtype=jnp.dtype("float64")))
            r = f(i, a)

            jaxpr, _, _ = trace_to_jaxpr(trace, [i, a], [r])
            assert len(jaxpr._invars) == 2
            assert len(jaxpr._outvars) == 1

        return sz

    r = circuit(3)
    assert r == 3


def test_abstracted_axis_no_recompilation():
    """Test that a function that does not need recompilation can be executed a second time"""

    @qml.qjit(abstracted_axes=(("n",), ()))
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def circuit(x1, x2):

        @qml.for_loop(0, jnp.shape(x1)[0], 1)
        def loop_block(i):
            qml.Hadamard(0)
            qml.RX(x1[i], 0)
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(1)

        loop_block()
        qml.RY(x2, 1)
        return qml.expval(qml.Z(1))

    x1 = jnp.array([0.1, 0.2, 0.3])
    x2 = 0.1967

    res_0 = circuit(x1, x2)
    _id0 = id(circuit.compiled_function)

    res_1 = circuit(x1, x2)
    _id1 = id(circuit.compiled_function)

    assert _id0 == _id1
    assert np.allclose(res_0, res_1)

    x1 = jnp.array([0.1, 0.2, 0.3, 0.4])

    res_2 = circuit(x1, x2)
    _id2 = id(circuit.compiled_function)
    assert _id0 == _id2

    res_3 = circuit(x1, x2)
    assert np.allclose(res_2, res_3)

    _id3 = id(circuit.compiled_function)
    assert _id0 == _id3


if __name__ == "__main__":
    pytest.main(["-x", __file__])
