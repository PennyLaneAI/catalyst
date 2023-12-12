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

from jax._src.core import dim_value_aval
from catalyst import qjit, while_loop, for_loop, cond
from catalyst.utils.contexts import EvaluationContext, EvaluationMode
from catalyst.utils.jax_extras import (
    deduce_avals3, input_type_to_tracers, collapse, expand_args, infer_lambda_input_type,
    ShapedArray, DShapedArray, DBIdx
)

DTYPES = [float, int, jnp.float32, jnp.float64, jnp.int8, jnp.int16, "float32", np.float64]
SHAPES = [3, (2, 3, 1), (), jnp.array([2, 1, 3], dtype=int)]


def assert_array_and_dtype_equal(a, b):
    """Check that two arrays have exactly the same values and types"""

    assert array_equal(a, b)
    assert a.dtype == b.dtype



def test_jax_typing():

    # def fun(a,b):
    #     b2 = jnp.zeros(a.shape[0]+1)
    #     return a,b2
    # a = jnp.zeros([1,1])
    # b = jnp.zeros([1,1])
    # wfun, in_sig, out_sig = deduce_avals3(fun, (a,b), {},
    #                                       abstracted_axes=({0:'0'},{1:'1'}))
    # in_type = in_sig.in_type

    with EvaluationContext(EvaluationMode.CLASSICAL_COMPILATION) as ctx, \
         EvaluationContext.frame_tracing_context(ctx) as trace:

        sz = trace.new_arg(dim_value_aval())
        args = jnp.zeros([0,sz]), jnp.zeros([sz,1])

        _, in_type = expand_args(args, force_implicit_indbidx=False)
        assert [(t.shape,k) for t,k in in_type] == [
            ((), False), ((0,DBIdx(val=0)), True), ((DBIdx(val=0),1), True)
        ]

        _, in_type = expand_args(args, force_implicit_indbidx=True)
        assert [(t.shape,k) for t,k in in_type] == [
            ((), False), ((), False), ((0,DBIdx(val=0)), True), ((DBIdx(val=1),1), True)
        ]


        # argkcollapse(in_sig.in_type, arg_tracers)
        # res_expanded_tracers = [ trace.full_raise(t) for t in wfun.call_wrapped(*arg_expanded_tracers) ]

    # assert in_type == in_type2
    # print(in_type)
    # print(in_type2)
    # print(in_type2)
    # print(out_sig.out_type())
    # print(out_sig.out_jaxpr())
    assert False




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
    """Test that tensor primitive work in the classical tracing mode, the traced dimention case"""

    @qjit
    def f(x):
        return jnp.ones(shape=[1, x], dtype=int)

    assert_array_and_dtype_equal(f(3), jnp.ones((1, 3), dtype=int))

def test_qjit_forloop_identity():

    @qjit()
    def f(sz):
        a = jnp.ones([sz], dtype=float)

        @for_loop(0, 10, 2)
        def loop(i, a):
            return a

        a2 = loop(a)
        return a2

    result = f(3)
    expected = jnp.ones(3)
    assert_array_and_dtype_equal(result, expected)


def test_qjit_forloop_shared_indbidx():

    @qjit()
    def f(sz):
        a = jnp.ones([sz], dtype=float)
        b = jnp.ones([sz], dtype=float)

        @for_loop(0, 10, 2)
        def loop(i, a, b):
            return (a, b)

        a2, b2 = loop(a, b)
        return a2 + b2

    result = f(3)
    expected = 2*jnp.ones(3)
    assert_array_and_dtype_equal(result, expected)


def test_qjit_forloop_indbidx_outdbidx():

    @qjit()
    def f(sz):
        a = jnp.ones([sz], dtype=float)
        b = jnp.ones([sz], dtype=float)

        @for_loop(0, 10, 2)
        def loop(i, a, _):
            b = jnp.ones([sz+1], dtype=float)
            return (a, b)

        a2, b2 = loop(a, b)
        return a2, b2

    res_a, res_b = f(3)
    assert_array_and_dtype_equal(res_a, jnp.ones(3))
    assert_array_and_dtype_equal(res_b, jnp.ones(4))


def test_qjit_forloop_index_indbidx():

    @qjit()
    def f(sz):
        a = jnp.ones([sz], dtype=float)

        @for_loop(0, 10, 1)
        def loop(i, _):
            b = jnp.ones([i], dtype=float)
            return b

        a2 = loop(a)
        return a2

    res_a = f(3)
    assert_array_and_dtype_equal(res_a, jnp.ones(9))


def test_qnode_forloop_identity():

    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(sz):
        a = jnp.ones([sz], dtype=float)

        @for_loop(0, 10, 2)
        def loop(i, a):
            return a

        a2 = loop(a)
        return a2

    result = f(3)
    expected = jnp.ones(3)
    assert_array_and_dtype_equal(result, expected)



def test_qnode_forloop_shared_indbidx():

    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(sz):
        a = jnp.ones([sz], dtype=float)
        b = jnp.ones([sz], dtype=float)

        @for_loop(0, 10, 2)
        def loop(i, a, b):
            return (a, b)

        a2, b2 = loop(a, b)
        return a2 + b2

    result = f(3)
    expected = 2*jnp.ones(3)
    assert_array_and_dtype_equal(result, expected)


def test_qnode_forloop_indbidx_outdbidx():

    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(sz):
        a = jnp.ones([sz], dtype=float)
        b = jnp.ones([sz], dtype=float)

        @for_loop(0, 10, 2)
        def loop(i, a, _):
            b = jnp.ones([sz+1], dtype=float)
            return (a, b)

        a2, b2 = loop(a, b)
        return a2, b2

    res_a, res_b = f(3)
    assert_array_and_dtype_equal(res_a, jnp.ones(3))
    assert_array_and_dtype_equal(res_b, jnp.ones(4))


def test_qnode_forloop_index_indbidx():

    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(sz):
        a = jnp.ones([sz], dtype=float)

        @for_loop(0, 10, 1)
        def loop(i, _):
            b = jnp.ones([i], dtype=float)
            return b

        a2 = loop(a)
        return a2

    res_a = f(3)
    assert_array_and_dtype_equal(res_a, jnp.ones(9))



def test_qnode_while_1():
    """Test that catalyst tensor primitive is compatible with quantum while"""

    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(sz):
        a = jnp.ones([sz+1], dtype=float)

        @while_loop(lambda _, i: i < 3)
        def loop(a, i):
            i += 1
            return (a, i)

        a2, _ = loop(a, 0)
        return a2

    result = f(3)
    expected = jnp.ones(4)
    assert_array_and_dtype_equal(result, expected)


def test_qnode_while_2():
    """Test that catalyst tensor primitive is compatible with quantum while"""

    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(sz):
        a = jnp.ones([sz+1], dtype=float)

        @while_loop(lambda _, i: i < 3)
        def loop(_, i):
            b = jnp.ones([sz+1], dtype=float)
            i += 1
            return (b, i)

        a2, _ = loop(a, 0)
        return a2

    result = f(3)
    expected = jnp.ones(4)
    assert_array_and_dtype_equal(result, expected)


def test_qnode_while_shared_indbidx():
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
    expected = 2*jnp.ones(3)
    assert_array_and_dtype_equal(result, expected)


def test_qnode_while_indbidx_outdbidx():
    """Test that catalyst tensor primitive is compatible with quantum while"""

    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(sz):
        a = jnp.ones([sz], dtype=float)
        b = jnp.ones([sz+1], dtype=float)

        @while_loop(lambda _a, _b, i: i < 3)
        def loop(a, _, i):
            b = jnp.ones([sz+1], dtype=float)
            i += 1
            return (a, b, i)

        a2, b2, _ = loop(a, b, 0)
        return a+a2, b2

    res_a, res_b = f(3)
    assert_array_and_dtype_equal(res_a, 2*jnp.ones(3))
    assert_array_and_dtype_equal(res_b, jnp.ones(4))


def test_qnode_while_outer():
    """Test that catalyst tensor primitive is compatible with quantum while"""

    @qjit
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def f(sz):
        a0 = jnp.ones([sz], dtype=float)

        @while_loop(lambda _a, i: i < 3)
        def loop(a, i):
            i += 1
            return (a0, i)

        a2, _ = loop(a0, 0)
        return a0+a2

    res_a = f(3)
    assert_array_and_dtype_equal(res_a, 2*jnp.ones(3))


def test_qjit_while_1():
    """Test that catalyst tensor primitive is compatible with quantum while"""

    @qjit
    def f(sz):
        a = jnp.ones([sz+1], dtype=float)

        @while_loop(lambda _, i: i < 3)
        def loop(_, i):
            b = jnp.ones([sz+1], dtype=float)
            i += 1
            return (b, i)

        a2, _ = loop(a, 0)
        return a2

    result = f(3)
    expected = jnp.ones(4)
    assert_array_and_dtype_equal(result, expected)


def test_qjit_while_2():
    """Test that catalyst tensor primitive is compatible with quantum while"""

    @qjit()
    def f(sz):
        a = jnp.ones([sz+1], dtype=float)

        @while_loop(lambda _, i: i < 3)
        def loop(_, i):
            b = jnp.ones([sz+1], dtype=float)
            i += 1
            return (b, i)

        a2, _ = loop(a, 0)
        return a2

    result = f(3)
    expected = jnp.ones(4)
    assert_array_and_dtype_equal(result, expected)


def test_qjit_while_3():
    """Test that catalyst tensor primitive is compatible with quantum while"""

    @qjit
    def f(sz:int):
        input_a = jnp.ones([sz+1], dtype=float)
        input_b = jnp.ones([sz+2], dtype=float)

        @while_loop(lambda a, b: False, preserve_dimensions=False)
        def loop(a, b):
            return (input_a, input_b)

        outputs = loop(input_a, input_a)
        return outputs

    result = f(3)
    print(f.jaxpr)
    print(result)
    # assert False
    # TODO: fix


def test_qjit_while_shared_indbidx():
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
    expected = 2*jnp.ones(3)
    assert_array_and_dtype_equal(result, expected)


def test_qjit_while_indbidx_outdbidx():
    """Test that catalyst tensor primitive is compatible with quantum while"""

    @qjit()
    def f(sz):
        a0 = jnp.ones([sz], dtype=float)
        b0 = jnp.ones([sz+1], dtype=float)

        @while_loop(lambda _a, _b, i: i < 3)
        def loop(a, _, i):
            b = jnp.ones([sz+1], dtype=float)
            i += 1
            return (a, b, i)

        a2, b2, _ = loop(a0, b0, 0)
        return a0+a2, b2

    res_a, res_b = f(3)
    assert_array_and_dtype_equal(res_a, 2*jnp.ones(3))
    assert_array_and_dtype_equal(res_b, jnp.ones(4))


def test_qjit_while_outer():
    """Test that catalyst tensor primitive is compatible with quantum while"""

    @qjit
    def f(sz):
        a0 = jnp.ones([sz], dtype=float)

        @while_loop(lambda _a, i: i < 3)
        def loop(a, i):
            i += 1
            return (a0, i)

        a2, _ = loop(a0, 0)
        assert a2.shape[0] is a0.shape[0]
        return a2

    res_a = f(3)
    assert_array_and_dtype_equal(res_a, jnp.ones(3))


def test_qjit_cond_identity():
    """Test that catalyst tensor primitive is compatible with quantum conditional"""

    @qjit
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


def test_qjit_cond_outdbidx():
    """Test that catalyst tensor primitive is compatible with quantum conditional"""

    @qjit
    def f(flag, sz):
        @cond(flag)
        def case():
            return jnp.ones([sz+1], dtype=float)

        @case.otherwise
        def case():
            return jnp.zeros([sz+1], dtype=float)

        return case()

    assert_array_and_dtype_equal(f(True, 3), jnp.ones(4))
    assert_array_and_dtype_equal(f(False, 3), jnp.zeros(4))


def test_qjit_cond_const_outdbidx():
    """Test that catalyst tensor primitive is compatible with quantum conditional"""

    @qjit
    def f(flag, sz):

        a = jnp.zeros([sz], dtype=float)

        @cond(flag)
        def case():
            return jnp.ones([sz+1], dtype=float)

        @case.otherwise
        def case():
            return a

        c = case()
        if flag is False:
            assert c.shape[0] is a.shape[0]
        return c

    assert_array_and_dtype_equal(f(True, 3), jnp.ones(4))
    assert_array_and_dtype_equal(f(False, 3), jnp.zeros(3))


@pytest.mark.skip("Dynamic arrays support in quantum control flow is not implemented")
def test_quantum_tracing_2():
    """Test that catalyst tensor primitive is compatible with quantum tracing mode"""

    @qjit()
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

    with pytest.raises(
        TypeError, match="Shapes must be 1D sequences of concrete values of integer type"
    ):
        qjit(f)


@pytest.mark.skip("Jax does not detect error in this use-case")
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


def test_accessing_shapes():
    """Test that dynamic tensor shapes are available for calculations"""

    @qjit
    def f(sz):
        a = jnp.ones((sz, sz))
        sa = jnp.array(a.shape)
        return jnp.sum(sa)

    assert f(3) == 6


if __name__ == "__main__":
    pytest.main(["-x", __file__])
