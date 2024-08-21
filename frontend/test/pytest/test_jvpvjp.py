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
""" Test JVP/VJP operation lowering """

from typing import TypeVar

import jax
import jax.numpy as jnp
import pennylane as qml
import pytest
from jax import jvp as J_jvp
from jax import vjp as J_vjp
from jax.tree_util import tree_flatten, tree_unflatten
from numpy.testing import assert_allclose

from catalyst import jvp as C_jvp
from catalyst import qjit
from catalyst import vjp as C_vjp

X = TypeVar("X")
T = TypeVar("T")


def circuit_rx(x1, x2):
    """A test quantum function"""
    qml.RX(x1, wires=0)
    qml.RX(x2, wires=0)
    return qml.expval(qml.PauliY(0))


def f_R1_to_R2(x):
    """A test function f : R1 -> R2"""
    return 2 * x, x * x


def g_R3_to_R2(_n, x):
    """A test function g : R3 -> R2, where `_n` is a dummy, non-differentiable parameter"""
    return jnp.stack([1 + x[0] + 2 * x[1] + 3 * x[2], 1 + x[0] + 2 * x[1] ** 2 + 3 * x[2] ** 3])


diff_methods = ["auto", "fd"]


def test_vjp_outside_qjit_scalar_scalar():
    """Test that vjp can be used outside of a jitting context on a scalar-scalar function."""

    def f(x):
        return x**2

    x = jnp.array(4.0)
    ct = jnp.array(1.0)
    res, f_vjp = jax.vjp(f, x)
    expected = tuple([res, f_vjp(ct)])
    result = C_vjp(f, x, ct)

    res_jax, tree_jax = jax.tree_util.tree_flatten(expected)
    res_cat, tree_cat = jax.tree_util.tree_flatten(result)
    assert tree_jax == tree_cat

    for r_j, r_c in zip(res_jax, res_cat):
        assert_allclose(r_j, r_c)


def test_vjp_outside_qjit_tuple_scalar():
    """Test that vjp can be used outside of a jitting context on a tuple-scalar function."""

    def f(x, y):
        return x**2 + y**2

    x = (jnp.array(4.0), jnp.array(4.0))
    ct = jnp.array(1.0)

    res, f_vjp = jax.vjp(f, *x)
    expected = tuple([res, f_vjp(ct)])
    result = C_vjp(f, x, ct)

    res_jax, tree_jax = jax.tree_util.tree_flatten(expected)
    res_cat, tree_cat = jax.tree_util.tree_flatten(result)
    assert tree_jax == tree_cat

    for r_j, r_c in zip(res_jax, res_cat):
        assert_allclose(r_j, r_c)


def test_vjp_outside_qjit_tuple_tuple():
    """Test that vjp can be used outside of a jitting context on a tuple-tuple function."""

    def f(x, y):
        return x**2, y**2

    x = (4.0, 4.0)
    ct = (1.0, 1.0)

    res, f_vjp = jax.vjp(f, *x)
    expected = tuple([res, f_vjp(ct)])
    result = C_vjp(f, x, ct)

    res_jax, tree_jax = jax.tree_util.tree_flatten(expected)
    res_cat, tree_cat = jax.tree_util.tree_flatten(result)
    assert tree_jax == tree_cat

    for r_j, r_c in zip(res_jax, res_cat):
        assert_allclose(r_j, r_c)


def test_jvp_outside_qjit_scalar_scalar():
    """Test that jvp can be used outside of a jitting context on a scalar-scalar function."""

    def f(x):
        return x**2

    x = (4.0,)
    t = (1.0,)

    expected = jax.jvp(f, x, t)
    result = C_jvp(f, x, t)

    assert_allclose(expected, result)


def test_jvp_outside_qjit_tuple_scalar():
    """Test that jvp can be used outside of a jitting context on a tuple-scalar function."""

    def f(x, y):
        return x**2 + y**2

    x = (4.0, 4.0)
    t = (1.0, 1.0)

    expected = jax.jvp(f, x, t)
    result = C_jvp(f, x, t)

    assert_allclose(expected, result)


def test_jvp_outside_qjit_tuple_tuple():
    """Test that jvp can be used outside of a jitting context on a tuple-tuple function."""

    def f(x, y):
        return x**2, y**2

    x = (4.0, 4.0)
    t = (1.0, 1.0)

    expected = jax.jvp(f, x, t)
    result = C_jvp(f, x, t)

    assert_allclose(expected, result)


@pytest.mark.parametrize("diff_method", diff_methods)
def test_jvp_against_jax_full_argnum_case_S_SS(diff_method):
    """Numerically tests Catalyst's jvp against the JAX version."""

    x, t = (
        [-0.1, 0.5],
        [0.1, 0.33],
    )

    @qjit
    def C_workflow():
        f = qml.QNode(circuit_rx, device=qml.device("lightning.qubit", wires=1))
        return C_jvp(f, x, t, method=diff_method, argnum=list(range(len(x))))

    @jax.jit
    def J_workflow():
        f = qml.QNode(circuit_rx, device=qml.device("default.qubit.jax", wires=1), interface="jax")
        return J_jvp(f, x, t)

    r1 = C_workflow()
    r2 = J_workflow()
    res_jax, tree_jax = jax.tree_util.tree_flatten(r1)
    res_cat, tree_cat = jax.tree_util.tree_flatten(r2)
    assert tree_jax == tree_cat
    assert_allclose(res_jax, res_cat)


@pytest.mark.parametrize("diff_method", diff_methods)
def test_jvp_against_jax_full_argnum_case_T_T(diff_method):
    """Numerically tests Catalyst's jvp against the JAX version."""

    def f(x):
        return jnp.stack([1 * x, 2 * x, 3 * x])

    x, t = (
        [jnp.zeros([4], dtype=float)],
        [jnp.ones([4], dtype=float)],
    )

    @qjit
    def C_workflow():
        return C_jvp(f, x, t, method=diff_method, argnum=list(range(len(x))))

    @jax.jit
    def J_workflow():
        return J_jvp(f, x, t)

    r1 = C_workflow()
    r2 = J_workflow()
    res_jax, tree_jax = jax.tree_util.tree_flatten(r1)
    res_cat, tree_cat = jax.tree_util.tree_flatten(r2)
    assert tree_jax == tree_cat
    assert_allclose(res_jax, res_cat)


@pytest.mark.parametrize("diff_method", diff_methods)
def test_jvp_against_jax_full_argnum_case_TT_T(diff_method):
    """Numerically tests Catalyst's jvp against the JAX version."""

    def f(x1, x2):
        return jnp.stack(
            [
                3 * jnp.reshape(x1, [6]) + 4 * jnp.reshape(x2, [6]),
                3 * jnp.reshape(x1, [6]) + 4 * jnp.reshape(x2, [6]),
            ]
        )

    x, t = (
        [jnp.zeros([3, 2], dtype=float), jnp.zeros([2, 3], dtype=float)],
        [jnp.ones([3, 2], dtype=float), jnp.ones([2, 3], dtype=float)],
    )

    @qjit
    def C_workflow():
        return C_jvp(f, x, t, method=diff_method, argnum=list(range(len(x))))

    @jax.jit
    def J_workflow():
        return J_jvp(f, x, t)

    r1 = C_workflow()
    r2 = J_workflow()
    res_jax, tree_jax = jax.tree_util.tree_flatten(r1)
    res_cat, tree_cat = jax.tree_util.tree_flatten(r2)
    assert tree_jax == tree_cat
    assert_allclose(res_jax, res_cat)


@pytest.mark.parametrize("diff_method", diff_methods)
def test_jvp_against_jax_full_argnum_case_T_TT(diff_method):
    """Numerically tests Catalyst's jvp against the JAX version."""

    def f(x):
        return (x, jnp.stack([1 * x, 2 * x, 3 * x]))

    x, t = (
        [jnp.zeros([4], dtype=float)],
        [jnp.ones([4], dtype=float)],
    )

    @qjit
    def C_workflow():
        return C_jvp(f, x, t, method=diff_method, argnum=list(range(len(x))))

    @jax.jit
    def J_workflow():
        return J_jvp(f, x, t)

    r1 = C_workflow()
    r2 = J_workflow()
    res_jax, tree_jax = jax.tree_util.tree_flatten(r1)
    res_cat, tree_cat = jax.tree_util.tree_flatten(r2)
    assert tree_jax == tree_cat

    for r_j, r_c in zip(res_jax, res_cat):
        assert_allclose(r_j, r_c)


@pytest.mark.parametrize("diff_method", diff_methods)
def test_jvp_against_jax_full_argnum_case_TT_TT(diff_method):
    """Numerically tests Catalyst's jvp against the JAX version."""

    def f(x1, x2):
        return (
            1 * jnp.reshape(x1, [6]) + 2 * jnp.reshape(x2, [6]),
            jnp.stack(
                [
                    3 * jnp.reshape(x1, [6]) + 4 * jnp.reshape(x2, [6]),
                    3 * jnp.reshape(x1, [6]) + 4 * jnp.reshape(x2, [6]),
                ]
            ),
        )

    x, t = (
        [jnp.zeros([3, 2], dtype=float), jnp.zeros([2, 3], dtype=float)],
        [jnp.ones([3, 2], dtype=float), jnp.ones([2, 3], dtype=float)],
    )

    @qjit
    def C_workflow():
        return C_jvp(f, x, t, method=diff_method, argnum=list(range(len(x))))

    @jax.jit
    def J_workflow():
        return J_jvp(f, x, t)

    r1 = C_workflow()
    r2 = J_workflow()
    res_jax, tree_jax = jax.tree_util.tree_flatten(r1)
    res_cat, tree_cat = jax.tree_util.tree_flatten(r2)
    assert tree_jax == tree_cat

    for r_j, r_c in zip(res_jax, res_cat):
        assert_allclose(r_j, r_c)


@pytest.mark.parametrize("diff_method", diff_methods)
def test_jvp_pytrees_return(diff_method):
    """Test that a JVP with pytrees as return."""

    def f(x, y):
        return [x, {"res": y}, x + y]

    @qjit
    def workflow():
        return C_jvp(f, [0.1, 0.2], [1.0, 1.0], method=diff_method, argnum=[0, 1])

    catalyst_res = workflow()
    jax_res = J_jvp(f, [0.1, 0.2], [1.0, 1.0])

    catalyst_res_flatten, tree_cat = jax.tree_util.tree_flatten(catalyst_res)
    jax_res_flatten, tree_jax = jax.tree_util.tree_flatten(jax_res)
    assert tree_cat == tree_jax
    assert_allclose(catalyst_res_flatten, jax_res_flatten)


@pytest.mark.parametrize("diff_method", diff_methods)
def test_jvp_pytrees_args(diff_method):
    """Test that a JVP with pytrees as args."""

    def f(x, y):
        return x["res1"] + y * x["res2"]

    @qjit
    def workflow():
        return C_jvp(
            f,
            [{"res1": 0.1, "res2": 0.2}, 0.3],
            [{"res1": 1.0, "res2": 1.0}, 1.0],
            method=diff_method,
            argnum=[0, 1],
        )

    catalyst_res = workflow()
    jax_res = J_jvp(f, [{"res1": 0.1, "res2": 0.2}, 0.3], [{"res1": 1.0, "res2": 1.0}, 1.0])

    catalyst_res_flatten, tree_cat = jax.tree_util.tree_flatten(catalyst_res)
    jax_res_flatten, tree_jax = jax.tree_util.tree_flatten(jax_res)
    assert tree_cat == tree_jax
    assert_allclose(catalyst_res_flatten, jax_res_flatten)


@pytest.mark.parametrize("diff_method", diff_methods)
def test_jvp_pytrees_args_and_return(diff_method):
    """Test that a JVP with pytrees as args."""

    def f(x, y):
        return [x["res1"] + y * x["res2"], {"res": y}, x["res1"]]

    @qjit
    def workflow():
        return C_jvp(
            f,
            [{"res1": 0.1, "res2": 0.2}, 0.3],
            [{"res1": 1.0, "res2": 1.0}, 1.0],
            method=diff_method,
            argnum=[0, 1],
        )

    catalyst_res = workflow()
    jax_res = J_jvp(f, [{"res1": 0.1, "res2": 0.2}, 0.3], [{"res1": 1.0, "res2": 1.0}, 1.0])

    catalyst_res_flatten, tree_cat = jax.tree_util.tree_flatten(catalyst_res)
    jax_res_flatten, tree_jax = jax.tree_util.tree_flatten(jax_res)
    assert tree_cat == tree_jax
    assert_allclose(catalyst_res_flatten, jax_res_flatten)


@pytest.mark.parametrize("diff_method", diff_methods)
def test_jvp_multi_returns(diff_method):
    """Test that a JVP with multiple arg as return."""

    def f(x):
        return x, x**2, x**4

    @qjit
    def workflow():
        return C_jvp(f, [0.3], [1.1], method=diff_method, argnum=[0])

    catalyst_res = workflow()
    jax_res = J_jvp(f, [0.3], [1.1])

    catalyst_res_flatten, tree_cat = jax.tree_util.tree_flatten(catalyst_res)
    jax_res_flatten, tree_jax = jax.tree_util.tree_flatten(jax_res)
    assert tree_cat == tree_jax
    assert_allclose(catalyst_res_flatten, jax_res_flatten, rtol=1e-6)


@pytest.mark.parametrize("diff_method", diff_methods)
def test_vjp_against_jax_full_argnum_case_S_SS(diff_method):
    """Numerically tests Catalyst's jvp against the JAX version."""

    x, ct = (
        [-0.1, 0.5],
        [0.111],
    )

    @qjit
    def C_workflow():
        f = qml.QNode(circuit_rx, device=qml.device("lightning.qubit", wires=1))
        return C_vjp(f, x, ct, method=diff_method, argnum=list(range(len(x))))

    @jax.jit
    def J_workflow():
        f = qml.QNode(circuit_rx, device=qml.device("default.qubit.jax", wires=1), interface="jax")
        y, ft = J_vjp(f, *x)
        ct2 = tree_unflatten(tree_flatten(y)[1], ct)
        return (y, ft(ct2))

    r1 = C_workflow()
    r2 = J_workflow()
    res_jax, tree_jax = jax.tree_util.tree_flatten(r1)
    res_cat, tree_cat = jax.tree_util.tree_flatten(r2)
    assert tree_jax == tree_cat
    assert_allclose(res_jax, res_cat)


@pytest.mark.parametrize("diff_method", diff_methods)
def test_vjp_against_jax_full_argnum_case_T_T(diff_method):
    """Numerically tests Catalyst's jvp against the JAX version."""

    def f(x):
        return jnp.stack([1 * x, 2 * x, 3 * x])

    x, ct = (
        [jnp.zeros([4], dtype=float)],
        [jnp.ones([3, 4], dtype=float)],
    )

    @qjit
    def C_workflow():
        return C_vjp(f, x, ct, method=diff_method, argnum=list(range(len(x))))

    @jax.jit
    def J_workflow():
        y, ft = J_vjp(f, *x)
        ct2 = tree_unflatten(tree_flatten(y)[1], ct)
        return (y, ft(ct2))

    r1 = C_workflow()
    r2 = J_workflow()
    res_jax, tree_jax = jax.tree_util.tree_flatten(r1)
    res_cat, tree_cat = jax.tree_util.tree_flatten(r2)
    assert tree_jax == tree_cat
    for r_j, r_c in zip(res_jax, res_cat):
        assert_allclose(r_j, r_c)


@pytest.mark.parametrize("diff_method", diff_methods)
def test_vjp_against_jax_full_argnum_case_TT_T(diff_method):
    """Numerically tests Catalyst's jvp against the JAX version."""

    def f(x1, x2):
        return jnp.stack(
            [
                3 * jnp.reshape(x1, [6]) + 4 * jnp.reshape(x2, [6]),
                3 * jnp.reshape(x1, [6]) + 4 * jnp.reshape(x2, [6]),
            ]
        )

    x, ct = (
        [jnp.zeros([3, 2], dtype=float), jnp.zeros([2, 3], dtype=float)],
        [jnp.ones([2, 6], dtype=float)],
    )

    @qjit
    def C_workflow():
        return C_vjp(f, x, ct, method=diff_method, argnum=list(range(len(x))))

    @jax.jit
    def J_workflow():
        y, ft = J_vjp(f, *x)
        ct2 = tree_unflatten(tree_flatten(y)[1], ct)
        return (y, ft(ct2))

    r1 = C_workflow()
    r2 = J_workflow()
    res_jax, tree_jax = jax.tree_util.tree_flatten(r1)
    res_cat, tree_cat = jax.tree_util.tree_flatten(r2)
    assert tree_jax == tree_cat
    for r_j, r_c in zip(res_jax, res_cat):
        assert_allclose(r_j, r_c)


@pytest.mark.parametrize("diff_method", diff_methods)
def test_vjp_against_jax_full_argnum_case_T_TT(diff_method):
    """Numerically tests Catalyst's jvp against the JAX version."""

    def f(x):
        return (x, jnp.stack([1 * x, 2 * x, 3 * x]))

    x, ct = (
        [jnp.zeros([4], dtype=float)],
        [jnp.ones([4], dtype=float), jnp.ones([3, 4], dtype=float)],
    )

    @qjit
    def C_workflow():
        return C_vjp(f, x, ct, method=diff_method, argnum=list(range(len(x))))

    @jax.jit
    def J_workflow():
        y, ft = J_vjp(f, *x)
        ct2 = tree_unflatten(tree_flatten(y)[1], ct)
        return (y, ft(ct2))

    r1 = C_workflow()
    r2 = J_workflow()
    res_jax, tree_jax = jax.tree_util.tree_flatten(r1)
    res_cat, tree_cat = jax.tree_util.tree_flatten(r2)
    assert tree_jax == tree_cat
    for r_j, r_c in zip(res_jax, res_cat):
        assert_allclose(r_j, r_c)


@pytest.mark.parametrize("diff_method", diff_methods)
def test_vjp_against_jax_full_argnum_case_TT_TT(diff_method):
    """Numerically tests Catalyst's jvp against the JAX version."""

    def f(x1, x2):
        return (
            1 * jnp.reshape(x1, [6]) + 2 * jnp.reshape(x2, [6]),
            jnp.stack(
                [
                    3 * jnp.reshape(x1, [6]) + 4 * jnp.reshape(x2, [6]),
                    3 * jnp.reshape(x1, [6]) + 4 * jnp.reshape(x2, [6]),
                ]
            ),
        )

    x, ct = (
        [jnp.zeros([3, 2], dtype=float), jnp.zeros([2, 3], dtype=float)],
        [jnp.ones([6], dtype=float), jnp.ones([2, 6], dtype=float)],
    )

    @qjit
    def C_workflow():
        return C_vjp(f, x, ct, method=diff_method, argnum=list(range(len(x))))

    @jax.jit
    def J_workflow():
        y, ft = J_vjp(f, *x)
        ct2 = tree_unflatten(tree_flatten(y)[1], ct)
        return (y, ft(ct2))

    r1 = C_workflow()
    r2 = J_workflow()
    res_jax, tree_jax = jax.tree_util.tree_flatten(r1)
    res_cat, tree_cat = jax.tree_util.tree_flatten(r2)
    assert tree_jax == tree_cat
    for r_j, r_c in zip(res_jax, res_cat):
        assert_allclose(r_j, r_c)


@pytest.mark.parametrize("diff_method", diff_methods)
def test_jvpvjp_argument_checks(diff_method):
    """Numerically tests Catalyst's jvp against the JAX version."""

    def f(x1, x2):
        return jnp.stack(
            [
                3 * jnp.reshape(x1, [6]) + 4 * jnp.reshape(x2, [6]),
                3 * jnp.reshape(x1, [6]) + 4 * jnp.reshape(x2, [6]),
            ]
        )

    x, t, ct = (
        [jnp.zeros([3, 2], dtype=float), jnp.zeros([2, 3], dtype=float)],
        [jnp.ones([3, 2], dtype=float), jnp.ones([2, 3], dtype=float)],
        [jnp.ones([2, 6], dtype=float)],
    )

    @qjit
    def C_workflow1():
        return C_jvp(f, x, tuple(t), method=diff_method, argnum=list(range(len(x))))

    @qjit
    def C_workflow2():
        return C_jvp(f, tuple(x), t, method=diff_method, argnum=tuple(range(len(x))))

    r1 = C_workflow1()
    r2 = C_workflow2()
    res_jax, tree_jax = jax.tree_util.tree_flatten(r1)
    res_cat, tree_cat = jax.tree_util.tree_flatten(r2)
    assert tree_jax == tree_cat
    for r_j, r_c in zip(res_jax, res_cat):
        assert_allclose(r_j, r_c)

    with pytest.raises(ValueError, match="argument must be an iterable"):

        @qjit
        def C_workflow_bad1():
            return C_jvp(f, 33, tuple(t), argnum=list(range(len(x))))

    with pytest.raises(ValueError, match="argument must be an iterable"):

        @qjit
        def C_workflow_bad2():
            return C_vjp(f, list(x), 33, argnum=list(range(len(x))))

    with pytest.raises(ValueError, match="argnum should be integer or a list of integers"):

        @qjit
        def C_workflow_bad3():
            return C_vjp(f, x, ct, argnum="invalid")


@pytest.mark.parametrize("diff_method", diff_methods)
def test_jvp_against_jax_argnum0_case_TT_TT(diff_method):
    """Numerically tests Catalyst's jvp against the JAX version, in case of empty or singular
    argnum argument."""

    def f(x1, x2):
        return (
            1 * jnp.reshape(x1, [6]) + 2 * jnp.reshape(x2, [6]),
            jnp.stack(
                [
                    3 * jnp.reshape(x1, [6]) + 4 * jnp.reshape(x2, [6]),
                    3 * jnp.reshape(x1, [6]) + 4 * jnp.reshape(x2, [6]),
                ]
            ),
        )

    x, t = (
        [jnp.zeros([3, 2], dtype=float), jnp.zeros([2, 3], dtype=float)],
        [jnp.ones([3, 2], dtype=float), jnp.ones([2, 3], dtype=float)],
    )

    @qjit
    def C_workflowA():
        return C_jvp(f, x, t[0:1], method=diff_method)

    @qjit
    def C_workflowB():
        return C_jvp(f, x, t[0:1], method=diff_method, argnum=[0])

    @jax.jit
    def J_workflow():
        # Emulating `argnum=[0]` in JAX
        def _f(a):
            return f(a, *x[1:])

        return J_jvp(_f, x[0:1], t[0:1])

    ra = C_workflowA()
    rb = C_workflowB()
    rj = J_workflow()

    res_cat_a, tree_cat_a = jax.tree_util.tree_flatten(ra)
    res_cat_b, tree_cat_b = jax.tree_util.tree_flatten(rb)
    res_jax, tree_jax = jax.tree_util.tree_flatten(rj)

    assert tree_cat_a == tree_jax
    assert tree_cat_a == tree_cat_b

    for r_j, r_c in zip(res_cat_a, res_cat_b):
        assert_allclose(r_j, r_c)
    for r_j, r_c in zip(res_cat_a, res_jax):
        assert_allclose(r_j, r_c)


@pytest.mark.parametrize("diff_method", diff_methods)
def test_vjp_against_jax_argnum0_case_TT_TT(diff_method):
    """Numerically tests Catalyst's vjp against the JAX version, in case of empty or singular
    argnum argument."""

    def f(x1, x2):
        return (
            1 * jnp.reshape(x1, [6]) + 2 * jnp.reshape(x2, [6]),
            jnp.stack(
                [
                    3 * jnp.reshape(x1, [6]) + 4 * jnp.reshape(x2, [6]),
                    3 * jnp.reshape(x1, [6]) + 4 * jnp.reshape(x2, [6]),
                ]
            ),
        )

    x, ct = (
        [jnp.zeros([3, 2], dtype=float), jnp.zeros([2, 3], dtype=float)],
        [jnp.ones([6], dtype=float), jnp.ones([2, 6], dtype=float)],
    )

    @qjit
    def C_workflowA():
        return C_vjp(f, x, ct, method=diff_method)

    @qjit
    def C_workflowB():
        return C_vjp(f, x, ct, method=diff_method, argnum=[0])

    @jax.jit
    def J_workflow():
        # Emulating `argnum=[0]` in JAX
        def _f(a):
            return f(a, *x[1:])

        y, ft = J_vjp(_f, *x[0:1])
        ct2 = tree_unflatten(tree_flatten(y)[1], ct)
        return (y, ft(ct2))

    ra = C_workflowA()
    rb = C_workflowB()
    rj = J_workflow()

    res_cat_a, tree_cat_a = jax.tree_util.tree_flatten(ra)
    res_cat_b, tree_cat_b = jax.tree_util.tree_flatten(rb)
    res_jax, tree_jax = jax.tree_util.tree_flatten(rj)

    assert tree_cat_a == tree_jax
    assert tree_cat_a == tree_cat_b

    for r_j, r_c in zip(res_cat_a, res_cat_b):
        assert_allclose(r_j, r_c)
    for r_j, r_c in zip(res_cat_a, res_jax):
        assert_allclose(r_j, r_c)


@pytest.mark.parametrize("diff_method", diff_methods)
def test_vjp_pytrees_return(diff_method):
    """Test VJP with pytree return."""

    def f(x, y):
        return [x, {"res": y}, x + y]

    @qjit
    def C_workflowA():
        ct2 = [1.0, {"res": 1.0}, 1.0]
        return C_vjp(f, [0.1, 0.2], ct2, method=diff_method, argnum=[0, 1])

    @jax.jit
    def J_workflow():
        y, ft = J_vjp(f, *[0.1, 0.2])
        ct2 = [1.0, {"res": 1.0}, 1.0]
        return (y, ft(ct2))

    r1 = C_workflowA()
    r2 = J_workflow()
    res_jax, tree_jax = jax.tree_util.tree_flatten(r1)
    res_cat, tree_cat = jax.tree_util.tree_flatten(r2)
    assert tree_jax == tree_cat
    for r_j, r_c in zip(res_jax, res_cat):
        assert_allclose(r_j, r_c)


@pytest.mark.parametrize("diff_method", diff_methods)
def test_vjp_pytrees_args(diff_method):
    """Test VJP with pytree args."""

    def f(x, y):
        return x["res1"] + y * x["res2"], y

    @qjit
    def C_workflowA():
        ct2 = [1.0, 1.0]
        return C_vjp(f, [{"res1": 0.1, "res2": 0.2}, 0.3], ct2, method=diff_method, argnum=[0, 1])

    @jax.jit
    def J_workflow():
        y, ft = J_vjp(f, *[{"res1": 0.1, "res2": 0.2}, 0.3])
        ct2 = (1.0, 1.0)
        return (y, ft(ct2))

    r1 = C_workflowA()
    r2 = J_workflow()
    res_jax, tree_jax = jax.tree_util.tree_flatten(r1)
    res_cat, tree_cat = jax.tree_util.tree_flatten(r2)
    assert tree_jax == tree_cat
    for r_j, r_c in zip(res_jax, res_cat):
        assert_allclose(r_j, r_c)


@pytest.mark.parametrize("diff_method", diff_methods)
def test_VJP_pytrees_args_and_return(diff_method):
    """Test that a VJP with pytrees as args."""

    def f(x, y):
        return [x["res1"] + y * x["res2"], {"res": y}, x["res1"]]

    @qjit
    def C_workflowA():
        ct2 = [1.0, {"res": 1.0}, 1.0]
        return C_vjp(f, [{"res1": 0.1, "res2": 0.2}, 0.3], ct2, method=diff_method, argnum=[0, 1])

    @jax.jit
    def J_workflow():
        y, ft = J_vjp(f, *[{"res1": 0.1, "res2": 0.2}, 0.3])
        ct2 = [1.0, {"res": 1.0}, 1.0]
        return (y, ft(ct2))

    r1 = C_workflowA()
    r2 = J_workflow()
    res_jax, tree_jax = jax.tree_util.tree_flatten(r1)
    res_cat, tree_cat = jax.tree_util.tree_flatten(r2)
    assert tree_jax == tree_cat
    for r_j, r_c in zip(res_jax, res_cat):
        assert_allclose(r_j, r_c)


@pytest.mark.parametrize("diff_method", diff_methods)
def test_vjp_multi_return(diff_method):
    """Test VJP with multiple returns."""

    def f(x):
        return x, x**2

    @qjit
    def C_workflowA():
        return C_vjp(f, [0.1], [1.0, 1.0], method=diff_method, argnum=[0])

    @jax.jit
    def J_workflow():
        y, ft = J_vjp(f, *[0.1])
        return (y, ft(tuple([1.0, 1.0])))

    r1 = C_workflowA()
    r2 = J_workflow()
    res_jax, tree_jax = jax.tree_util.tree_flatten(r1)
    res_cat, tree_cat = jax.tree_util.tree_flatten(r2)
    assert tree_jax == tree_cat
    for r_j, r_c in zip(res_jax, res_cat):
        assert_allclose(r_j, r_c)


@pytest.mark.parametrize("diff_method", diff_methods)
def test_jvp_argument_type_checks_correct_inputs(diff_method):
    """Test that Catalyst's jvp can JIT compile when given the correct types."""

    @qjit
    def C_workflow_f():
        x = (1.0,)
        tangents = (1.0,)
        return C_jvp(f_R1_to_R2, x, tangents, method=diff_method, argnum=[0])

    @qjit
    def C_workflow_g():
        x = jnp.array([2.0, 3.0, 4.0])
        tangents = jnp.ones([3], dtype=float)
        return C_jvp(g_R3_to_R2, [1, x], [tangents], method=diff_method, argnum=[1])


@pytest.mark.parametrize("diff_method", diff_methods)
def test_jvp_argument_type_checks_incompatible_n_inputs(diff_method):
    """Tests error handling of Catalyst's jvp when the number of differentiable params
    and tangent arguments are incompatible.
    """

    with pytest.raises(
        TypeError,
        match=(
            "number of tangent and number of differentiable parameters in catalyst.jvp "
            "do not match"
        ),
    ):

        @qjit
        def C_workflow():
            # If `f` takes one differentiable param (argnum=[0]), then `tangents` must have length 1
            x = (1.0,)
            tangents = (1.0, 1.0)
            return C_jvp(f_R1_to_R2, x, tangents, method=diff_method, argnum=[0])


@pytest.mark.parametrize("diff_method", diff_methods)
def test_jvp_argument_type_checks_incompatible_input_types(diff_method):
    """Tests error handling of Catalyst's jvp when the types of the differentiable
    params and tangent arguments are incompatible.
    """

    with pytest.raises(
        TypeError, match="function params and tangents arguments to catalyst.jvp do not match"
    ):

        @qjit
        def C_workflow():
            # If `x` has type float, then `tangents` should also have type float
            x = (1.0,)
            tangents = (1,)
            return C_jvp(f_R1_to_R2, x, tangents, method=diff_method, argnum=[0])


@pytest.mark.parametrize("diff_method", diff_methods)
def test_jvp_argument_type_checks_incompatible_input_shapes(diff_method):
    """Tests error handling of Catalyst's jvp when the shapes of the differentiable
    params and tangent arguments are incompatible.
    """

    with pytest.raises(
        ValueError, match="catalyst.jvp called with different function params and tangent shapes"
    ):

        @qjit
        def C_workflow():
            # If `x` has shape (3,), then `tangents` must also have shape (3,),
            # but it has shape (4,)
            x = jnp.array([2.0, 3.0, 4.0])
            tangents = jnp.ones([4], dtype=float)
            return C_jvp(g_R3_to_R2, [1, x], [tangents], method=diff_method, argnum=[1])


@pytest.mark.parametrize("diff_method", diff_methods)
def test_vjp_argument_type_checks_correct_inputs(diff_method):
    """Test that Catalyst's vjp can JIT compile when given the correct types."""

    @qjit
    def C_workflow_f():
        x = (1.0,)
        cotangents = (1.0, 1.0)
        return C_vjp(f_R1_to_R2, x, cotangents, method=diff_method, argnum=[0])

    @qjit
    def C_workflow_g():
        x = jnp.array([2.0, 3.0, 4.0])
        cotangents = jnp.ones([2], dtype=float)
        return C_vjp(g_R3_to_R2, [1, x], [cotangents], method=diff_method, argnum=[1])


@pytest.mark.parametrize("diff_method", diff_methods)
def test_vjp_argument_type_checks_incompatible_n_inputs(diff_method):
    """Tests error handling of Catalyst's vjp when the number of function output params
    and cotangent arguments are incompatible.
    """

    with pytest.raises(
        TypeError,
        match=(
            "number of cotangent and number of function output parameters in catalyst.vjp "
            "do not match"
        ),
    ):

        @qjit
        def C_workflow():
            # If `f` returns two outputs, then `cotangents` must have length 2
            x = (1.0,)
            cotangents = (1.0,)
            return C_vjp(f_R1_to_R2, x, cotangents, method=diff_method, argnum=[0])


@pytest.mark.parametrize("diff_method", diff_methods)
def test_vjp_argument_type_checks_incompatible_input_types(diff_method):
    """Tests error handling of Catalyst's vjp when the types of the function output params
    and cotangent arguments are incompatible.
    """

    with pytest.raises(
        TypeError,
        match="function output params and cotangents arguments to catalyst.vjp do not match",
    ):

        @qjit
        def C_workflow():
            # If `x` has type float, then `cotangents` should also have type float
            x = (1.0,)
            cotangents = (1, 1)
            return C_vjp(f_R1_to_R2, x, cotangents, method=diff_method, argnum=[0])


@pytest.mark.parametrize("diff_method", diff_methods)
def test_vjp_argument_type_checks_incompatible_input_shapes(diff_method):
    """Tests error handling of Catalyst's vjp when the shapes of the function output params
    and cotangent arguments are incompatible.
    """

    with pytest.raises(
        ValueError,
        match="catalyst.vjp called with different function output params and cotangent shapes",
    ):

        @qjit
        def C_workflow():
            # If `f` returns object with shape (2,), then `cotangents` must also have
            # shape (2,), but it has shape (3,)
            x = jnp.array([2.0, 3.0, 4.0])
            cotangents = jnp.ones([3], dtype=float)
            return C_vjp(g_R3_to_R2, [1, x], [cotangents], method=diff_method, argnum=[1])


if __name__ == "__main__":
    pytest.main(["-x", __file__])
