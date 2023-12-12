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

from typing import Iterable, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import pennylane as qml
import pytest
from jax import linearize as J_linearize
from jax import vjp as J_vjp
from jax.tree_util import tree_flatten, tree_unflatten
from numpy.testing import assert_allclose

from catalyst import jvp as C_jvp
from catalyst import qjit
from catalyst import vjp as C_vjp

X = TypeVar("X")
T = TypeVar("T")


def flatten_if_tuples(x: Union[X, Tuple[Union[T, Tuple[T]]]]) -> Union[X, Tuple[T]]:
    """Flatten first layer of Python tuples."""
    return (
        sum(((i if isinstance(i, tuple) else (i,)) for i in x), ()) if isinstance(x, tuple) else x
    )


def circuit_rx(x1, x2):
    """A test quantum function"""
    qml.RX(x1, wires=0)
    qml.RX(x2, wires=0)
    return qml.expval(qml.PauliY(0))


def assert_elements_allclose(a, b, **kwargs):
    """Checks all elements of tuples, one by one, for approximate equality"""
    assert all(
        isinstance(i, Iterable) for i in [a, b]
    ), f"Some of {[type(a),type(b)]} is not a tuple"
    assert len(a) == len(b), f"len(a) ({len(a)}) != len(b) ({type(b)})"
    for i, j in zip(a, b):
        assert_allclose(i, j, **kwargs)


diff_methods = ["auto", "fd"]


def test_vjp_outside_qjit_scalar_scalar():
    """Test that vjp can be used outside of a jitting context on a scalar-scalar function."""

    def f(x):
        return x**2

    x = (4.0,)
    ct = (1.0,)

    expected = jax.vjp(f, *x)[1](*ct)
    result = C_vjp(f, x, ct)

    assert_allclose(expected, result)


def test_vjp_outside_qjit_tuple_scalar():
    """Test that vjp can be used outside of a jitting context on a tuple-scalar function."""

    def f(x, y):
        return x**2 + y**2

    x = (4.0, 4.0)
    ct = (1.0,)

    expected = jax.vjp(f, *x)[1](*ct)
    result = C_vjp(f, x, ct)

    assert_allclose(expected, result)


def test_vjp_outside_qjit_tuple_tuple():
    """Test that vjp can be used outside of a jitting context on a tuple-tuple function."""

    def f(x, y):
        return x**2, y**2

    x = (4.0, 4.0)
    ct = (1.0, 1.0)

    expected = jax.vjp(f, *x)[1](ct)
    result = C_vjp(f, x, ct)

    assert_allclose(expected, result)


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
        y, ft = J_linearize(f, *x)
        return flatten_if_tuples((y, ft(*t)))

    r1 = C_workflow()
    r2 = J_workflow()
    assert_elements_allclose(r1, r2, rtol=1e-6, atol=1e-6)


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
        y, ft = J_linearize(f, *x)
        return flatten_if_tuples((y, ft(*t)))

    r1 = C_workflow()
    r2 = J_workflow()
    assert_elements_allclose(r1, r2, rtol=1e-6, atol=1e-6)


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
        y, ft = J_linearize(f, *x)
        return flatten_if_tuples((y, ft(*t)))

    r1 = C_workflow()
    r2 = J_workflow()
    assert_elements_allclose(r1, r2, rtol=1e-6, atol=1e-6)


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
        y, ft = J_linearize(f, *x)
        return flatten_if_tuples((y, ft(*t)))

    r1 = C_workflow()
    r2 = J_workflow()
    assert_elements_allclose(r1, r2, rtol=1e-6, atol=1e-6)


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
        y, ft = J_linearize(f, *x)
        return flatten_if_tuples((y, ft(*t)))

    r1 = C_workflow()
    r2 = J_workflow()
    assert_elements_allclose(r1, r2, rtol=1e-6, atol=1e-6)


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
        return flatten_if_tuples((y, ft(ct2)))

    r1 = C_workflow()
    r2 = J_workflow()
    assert_elements_allclose(r1, r2, rtol=1e-6, atol=1e-6)


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
        return flatten_if_tuples((y, ft(ct2)))

    r1 = C_workflow()
    r2 = J_workflow()
    assert_elements_allclose(r1, r2, rtol=1e-6, atol=1e-6)


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
        return flatten_if_tuples((y, ft(ct2)))

    r1 = C_workflow()
    r2 = J_workflow()
    assert_elements_allclose(r1, r2, rtol=1e-6, atol=1e-6)


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
        return flatten_if_tuples((y, ft(ct2)))

    r1 = C_workflow()
    r2 = J_workflow()
    assert_elements_allclose(r1, r2, rtol=1e-6, atol=1e-6)


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
        return flatten_if_tuples((y, ft(ct2)))

    r1 = C_workflow()
    r2 = J_workflow()
    assert_elements_allclose(r1, r2, rtol=1e-6, atol=1e-6)
    assert_elements_allclose(r1, r2, rtol=1e-6, atol=1e-6)


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

    assert_elements_allclose(C_workflow1(), C_workflow2(), rtol=1e-6, atol=1e-6)

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

        y, ft = J_linearize(_f, *x[0:1])
        return flatten_if_tuples((y, ft(*t[0:1])))

    r1a = C_workflowA()
    r1b = C_workflowB()
    r2 = J_workflow()
    assert_elements_allclose(r1a, r1b, rtol=1e-6, atol=1e-6)
    assert_elements_allclose(r1a, r2, rtol=1e-6, atol=1e-6)


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
        return flatten_if_tuples((y, ft(ct2)))

    r1a = C_workflowA()
    r1b = C_workflowB()
    r2 = J_workflow()
    assert_elements_allclose(r1a, r1b, rtol=1e-6, atol=1e-6)
    assert_elements_allclose(r1a, r2, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
