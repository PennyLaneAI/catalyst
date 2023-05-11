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
""" Test JVP/VJP operation lowering """

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import pennylane as qml
import pytest
from jax import grad as J_grad
from jax import linearize as J_jvp
from jax import vjp as J_vjp
from jax.tree_util import tree_flatten, tree_unflatten
from numpy.testing import assert_allclose
from pennylane import qnode

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


TensorLike = Any
TensorFunction = Callable[[TensorLike, ...], TensorLike]


@dataclass
class PureFunction:
    """Pure function that does not require any decorators."""

    f: TensorFunction


@dataclass
class QuantumFunction:
    """Quantum function that does require a decorator."""

    f: TensorFunction


def _bless_jittable(
    quantum_decorator: Callable[[Callable], Callable], f: Union[PureFunction, QuantumFunction]
) -> Callable:
    """Wraps quantum functions with a proper quantum decorator, known by the caller. Pure functions
    are returned as-is."""

    if isinstance(f, PureFunction):
        return f.f
    elif isinstance(f, QuantumFunction):
        return quantum_decorator(f.f)
    else:
        raise ValueError("Expecting either PureFunction or QuantumFunction")


C_decorator = partial(qml.QNode, device=qml.device("lightning.qubit", wires=1))
# J_decorator = partial(qml.QNode, device=qml.device("lightning.qubit", wires=1),
#                       interface="jax")
J_decorator = partial(qml.QNode, device=qml.device("default.qubit.jax", wires=1), interface="jax")


def circuit_rx(x1, x2):
    """A test quantum function"""
    qml.RX(x1, wires=0)
    qml.RX(x2, wires=0)
    return qml.expval(qml.PauliY(0))


testvec = [
    (QuantumFunction(circuit_rx), [-0.1, 0.5], [0.1, 0.33], [0.111]),
    (
        PureFunction(
            lambda x1, x2: (
                jnp.stack(
                    [
                        3 * jnp.reshape(x1, [6]) + 4 * jnp.reshape(x2, [6]),
                        3 * jnp.reshape(x1, [6]) + 4 * jnp.reshape(x2, [6]),
                    ]
                )
            )
        ),
        [jnp.zeros([3, 2], dtype=float), jnp.zeros([2, 3], dtype=float)],
        [jnp.ones([3, 2], dtype=float), jnp.ones([2, 3], dtype=float)],
        [jnp.ones([2, 6], dtype=float)],
    ),
    (
        PureFunction(lambda x: (x, jnp.stack([1 * x, 2 * x, 3 * x]))),
        [jnp.zeros([4], dtype=float)],
        [jnp.ones([4], dtype=float)],
        [jnp.ones([4], dtype=float), jnp.ones([3, 4], dtype=float)],
    ),
    (
        PureFunction(lambda x: jnp.stack([1 * x, 2 * x, 3 * x])),
        [jnp.zeros([4], dtype=float)],
        [jnp.ones([4], dtype=float)],
        [jnp.ones([3, 4], dtype=float)],
    ),
    (
        PureFunction(
            lambda x1, x2: (
                1 * jnp.reshape(x1, [6]) + 2 * jnp.reshape(x2, [6]),
                jnp.stack(
                    [
                        3 * jnp.reshape(x1, [6]) + 4 * jnp.reshape(x2, [6]),
                        3 * jnp.reshape(x1, [6]) + 4 * jnp.reshape(x2, [6]),
                    ]
                ),
            )
        ),
        [jnp.zeros([3, 2], dtype=float), jnp.zeros([2, 3], dtype=float)],
        [jnp.ones([3, 2], dtype=float), jnp.ones([2, 3], dtype=float)],
        [jnp.ones([6], dtype=float), jnp.ones([2, 6], dtype=float)],
    ),
]


def assert_elements_allclose(a, b, **kwargs):
    """Checks all elements of tuples, one by one, for approximate equality"""
    assert all(isinstance(i, tuple) for i in [a, b]), f"Some of {[type(a),type(b)]} is not a tuple"
    assert len(a) == len(b), f"len(a) ({len(a)}) != len(b) ({type(b)})"
    for i, j in zip(a, b):
        assert_allclose(i, j, **kwargs)


@pytest.mark.parametrize("f, x, t, _", testvec)
def test_jvp_against_jax_full_argnum(f: callable, x: list, t: list, _):
    """Numerically tests Catalyst's jvp against the JAX version."""

    @qjit
    def C_workflow():
        return C_jvp(_bless_jittable(C_decorator, f), x, t, method="fd", argnum=list(range(len(x))))

    @jax.jit
    def J_workflow():
        y, ft = J_jvp(_bless_jittable(J_decorator, f), *x)
        return flatten_if_tuples((y, ft(*t)))

    r1 = C_workflow()
    r2 = J_workflow()
    assert_elements_allclose(r1, r2, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("f, x, _, ct", testvec)
def test_vjp_against_jax_full_argnum(f: callable, x: list, _, ct: list):
    """Numerically tests Catalyst's jvp against the JAX version."""

    @qjit
    def C_workflow():
        return C_vjp(
            _bless_jittable(C_decorator, f), x, ct, method="fd", argnum=list(range(len(x)))
        )

    @jax.jit
    def J_workflow():
        y, ft = J_vjp(_bless_jittable(J_decorator, f), *x)
        ct2 = tree_unflatten(tree_flatten(y)[1], ct)
        return flatten_if_tuples((y, ft(ct2)))

    r1 = C_workflow()
    r2 = J_workflow()
    assert_elements_allclose(r1, r2, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("f, x, t, ct", testvec[0:1])
def test_jvpvjp_argument_checks(f: callable, x: list, t: list, ct: list):
    """Numerically tests Catalyst's jvp against the JAX version."""

    C_f = _bless_jittable(C_decorator, f)

    @qjit
    def C_workflow1():
        return C_jvp(C_f, list(x), t, method="fd", argnum=list(range(len(x))))

    @qjit
    def C_workflow2():
        return C_jvp(C_f, tuple(x), t, method="fd", argnum=list(range(len(x))))

    @qjit
    def C_workflow3():
        return C_vjp(C_f, list(x), list(ct), method="fd", argnum=list(range(len(x))))

    @qjit
    def C_workflow4():
        return C_vjp(C_f, tuple(x), tuple(ct), method="fd", argnum=list(range(len(x))))

    assert_elements_allclose(C_workflow1(), C_workflow2(), rtol=1e-6, atol=1e-6)
    assert_elements_allclose(C_workflow3(), C_workflow4(), rtol=1e-6, atol=1e-6)

    with pytest.raises(ValueError, match="argument must be a list or a tuple"):

        @qjit
        def C_workflow_bad1():
            return C_jvp(C_f, 33, tuple(t), method="fd", argnum=list(range(len(x))))

    with pytest.raises(ValueError, match="argument must be a list or a tuple"):

        @qjit
        def C_workflow_bad2():
            return C_vjp(C_f, 33, tuple(ct), method="fd", argnum=list(range(len(x))))


@pytest.mark.parametrize("f, x, t, _", testvec)
def test_jvp_against_jax_argnum0(f: callable, x: list, t: list, _):
    """Numerically tests Catalyst's jvp against the JAX version, in case of empty or singular
    argnum argument."""

    C_f = _bless_jittable(C_decorator, f)
    J_f = _bless_jittable(J_decorator, f)

    @qjit
    def C_workflowA():
        return C_jvp(C_f, x, t[0:1], method="fd")

    @qjit
    def C_workflowB():
        return C_jvp(C_f, x, t[0:1], method="fd", argnum=[0])

    @jax.jit
    def J_workflow():
        # Emulating `argnum=[0]` in JAX
        def _f(a):
            return J_f(a, *x[1:])

        y, ft = J_jvp(_f, *x[0:1])
        return flatten_if_tuples((y, ft(*t[0:1])))

    r1a = C_workflowA()
    r1b = C_workflowB()
    r2 = J_workflow()
    assert_elements_allclose(r1a, r1b, rtol=1e-6, atol=1e-6)
    assert_elements_allclose(r1a, r2, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("f, x, _, ct", testvec)
def test_vjp_against_jax_argnum0(f: callable, x: list, _, ct: list):
    """Numerically tests Catalyst's vjp against the JAX version, in case of empty or singular
    argnum argument."""

    C_f = _bless_jittable(C_decorator, f)
    J_f = _bless_jittable(J_decorator, f)

    @qjit
    def C_workflowA():
        return C_vjp(C_f, x, ct, method="fd")

    @qjit
    def C_workflowB():
        return C_vjp(C_f, x, ct, method="fd", argnum=[0])

    @jax.jit
    def J_workflow():
        # Emulating `argnum=[0]` in JAX
        def _f(a):
            return J_f(a, *x[1:])

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
