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

"""Integration tests for the async execution of QNodes features."""
import numpy as np
import pennylane as qml
import pytest
from jax import numpy as jnp

from catalyst import adjoint, cond, for_loop, grad, measure, qjit, while_loop

# We are explicitly testing that when something is not assigned
# the use is awaited.
# pylint: disable=expression-not-assigned
# pylint: disable=missing-function-docstring


def test_qnode_execution(backend):
    """The two first QNodes are executed in parrallel."""
    dev = qml.device(backend, wires=2)

    def multiple_qnodes(params):
        @qml.qnode(device=dev)
        def circuit1(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(wires=0))

        @qml.qnode(device=dev)
        def circuit2(params):
            qml.RY(params[0], wires=0)
            qml.RZ(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliX(wires=0))

        @qml.qnode(device=dev)
        def circuit3(params):
            qml.RZ(params[0], wires=0)
            qml.RX(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(wires=0))

        new_params = jnp.array([circuit1(params), circuit2(params)])
        return circuit3(new_params)

    params = jnp.array([1.0, 2.0])
    compiled = qjit(async_qnodes=True)(multiple_qnodes)
    observed = compiled(params)
    expected = qjit()(multiple_qnodes)(params)
    assert "async_execute_fn" in compiled.qir
    assert np.allclose(expected, observed)


# TODO: add the following diff_methods once issue #419 is fixed:
# ("parameter-shift", "auto"), ("adjoint", "auto")]
@pytest.mark.parametrize("diff_methods", [("finite-diff", "fd")])
@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_gradient(inp, diff_methods, backend):
    """Parameter shift and finite diff generate multiple QNode that are run async."""

    def f(x):
        qml.RX(x * 2, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit(async_qnodes=True)
    def compiled(x: float):
        g = qml.qnode(qml.device(backend, wires=1), diff_method=diff_methods[0])(f)
        h = grad(g, method=diff_methods[1])
        return h(x)

    def interpreted(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f, device, diff_method="backprop")
        h = qml.grad(g, argnums=0)
        return h(x)

    assert "async_execute_fn" in compiled.qir
    assert np.allclose(compiled(inp), interpreted(inp))


def test_exception(backend):
    "Test exception."

    @qml.qnode(qml.device(backend, wires=2))
    def circuit(x: int):
        qml.CNOT(wires=[x, 0])
        return qml.probs()

    @qjit(async_qnodes=True)
    def wrapper():
        return circuit(0)

    # TODO: Better error messages.
    msg = "Unrecoverable error"
    with pytest.raises(RuntimeError, match=msg):
        wrapper()


def test_exception2(backend):
    "Test exception in multiple async executions."

    @qml.qnode(qml.device(backend, wires=2))
    def circuit(x: int):
        qml.CNOT(wires=[x, 0])
        return qml.probs()

    @qjit(async_qnodes=True)
    def wrapper():
        return circuit(0) + circuit(0)

    # TODO: Better error messages.
    msg = "Unrecoverable error"
    with pytest.raises(RuntimeError, match=msg):
        wrapper()


def test_exception3(backend):
    "Test exception when not used in python."

    @qml.qnode(qml.device(backend, wires=2))
    def circuit(x: int):
        qml.CNOT(wires=[x, 0])
        return qml.probs()

    @qjit(async_qnodes=True)
    def wrapper():
        circuit(0) + circuit(0)
        return None

    # TODO: Better error messages.
    msg = "Unrecoverable error"
    with pytest.raises(RuntimeError, match=msg):
        wrapper()


def test_exception4(backend):
    "Test exception happening on two different circuits."

    @qml.qnode(qml.device(backend, wires=2))
    def circuit(x: int):
        qml.CNOT(wires=[x, 0])
        return qml.probs()

    @qml.qnode(qml.device(backend, wires=2))
    def circuit2(x: int):
        qml.Hadamard(wires=[0])
        qml.CNOT(wires=[x, 0])
        return qml.probs()

    @qjit(async_qnodes=True)
    def wrapper():
        circuit(0) + circuit2(0)
        return None

    # TODO: Better error messages.
    msg = "Unrecoverable error"
    with pytest.raises(RuntimeError, match=msg):
        wrapper()


def test_exception_adjoint(backend):
    "Test exception happening on two different circuits both adjointed."

    def bad_cnot(x):
        qml.CNOT(wires=[x, 0])

    @qml.qnode(qml.device(backend, wires=2))
    def circuit(x: int):
        adjoint(bad_cnot)(x)
        return qml.probs()

    @qml.qnode(qml.device(backend, wires=2))
    def circuit2(x: int):
        qml.Hadamard(wires=[0])
        adjoint(bad_cnot)(x)
        return qml.probs()

    @qjit(async_qnodes=True)
    def wrapper():
        circuit(0) + circuit2(0)
        return None

    # TODO: Better error messages.
    msg = "Unrecoverable error"
    with pytest.raises(RuntimeError, match=msg):
        wrapper()


def test_exception_conditional(backend):
    @qml.qnode(qml.device(backend, wires=2))
    def circuit(x: int):
        qml.CNOT(wires=[x, 0])
        return qml.probs()

    @qjit(async_qnodes=True)
    def wrapper(x: int):
        @cond(x == 1)
        def cond_fn():
            return circuit(1)

        @cond_fn.otherwise
        def cond_fn():
            return circuit(0)

        return cond_fn()

    # TODO: Better error messages.
    msg = "Unrecoverable error"
    with pytest.raises(RuntimeError, match=msg):
        wrapper(0)


def test_exception_conditional_1(backend):
    "Test exception happening in else and outside else."

    @qml.qnode(qml.device(backend, wires=2))
    def circuit(x: int):
        qml.CNOT(wires=[x, 0])
        return qml.probs()

    @qjit(async_qnodes=True)
    def wrapper(x: int):
        y = circuit(1)

        @cond(x == 1)
        def cond_fn():
            return circuit(1)

        @cond_fn.otherwise
        def cond_fn():
            return circuit(0)

        return y + cond_fn()

    # TODO: Better error messages.
    msg = "Unrecoverable error"
    with pytest.raises(RuntimeError, match=msg):
        wrapper(0)


def test_exception_conditional_2(backend):
    "Test exception happening in the presence of an if statement but in another."

    @qml.qnode(qml.device(backend, wires=2))
    def circuit(x: int):
        qml.CNOT(wires=[x, 0])
        return qml.probs()

    @qjit(async_qnodes=True)
    def wrapper(x: int):
        y = circuit(0)

        @cond(x == 1)
        def cond_fn():
            return circuit(1)

        @cond_fn.otherwise
        def cond_fn():
            return circuit(1)

        return y + cond_fn()

    # TODO: Better error messages.
    msg = "Unrecoverable error"
    with pytest.raises(RuntimeError, match=msg):
        wrapper(0)


@pytest.mark.parametrize(
    "order", [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 1, 0), (1, 0, 1), (1, 1, 1)]
)
def test_qnode_exception_dependency(order, backend):
    """The two first QNodes are executed in parrallel."""
    dev = qml.device(backend, wires=2)
    x = order[0]
    y = order[1]
    z = order[2]

    def multiple_qnodes(params):
        @qml.qnode(device=dev)
        def circuit1(params, x):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[x, 1])
            return qml.expval(qml.PauliZ(wires=0))

        @qml.qnode(device=dev)
        def circuit2(params, x):
            qml.RY(params[0], wires=0)
            qml.RZ(params[1], wires=1)
            qml.CNOT(wires=[x, 1])
            return qml.expval(qml.PauliX(wires=0))

        @qml.qnode(device=dev)
        def circuit3(params, x):
            qml.RZ(params[0], wires=0)
            qml.RX(params[1], wires=1)
            qml.CNOT(wires=[x, 1])
            return qml.expval(qml.PauliY(wires=0))

        new_params = jnp.array([circuit1(params, x), circuit2(params, y)])
        return circuit3(new_params, z)

    params = jnp.array([1.0, 2.0])
    msg = "Unrecoverable error"
    with pytest.raises(RuntimeError, match=msg):
        qjit(async_qnodes=True)(multiple_qnodes)(params)


# TODO: add the following diff_methods once issue #419 is fixed:
# ("parameter-shift", "auto"), ("adjoint", "auto")]
@pytest.mark.parametrize("diff_methods", [("finite-diff", "fd")])
@pytest.mark.parametrize("inp", [(1.0)])
def test_gradient_exception(inp, diff_methods, backend):
    """Parameter shift and finite diff generate multiple QNode that are run async."""

    def f(x, y):
        qml.RX(x * 2, wires=0)
        qml.CNOT(wires=[0, y])
        return qml.expval(qml.PauliY(0))

    @qjit(async_qnodes=True)
    def compiled(x: float):
        g = qml.qnode(qml.device(backend, wires=1), diff_method=diff_methods[0])(f)
        h = grad(g, method=diff_methods[1], argnums=[0])
        return h(x, 0)

    msg = "Unrecoverable error"
    with pytest.raises(RuntimeError, match=msg):
        compiled(inp)


def test_exception_in_loop(backend):
    "Test exception happening in a loop."

    @qjit(async_qnodes=True)
    @qml.qnode(qml.device(backend, wires=3))
    def circuit(n):
        @while_loop(lambda v: v[0] < v[1])
        def loop(v):
            qml.CNOT(wires=[n, v[0]])
            return v[0] + 1, v[1]

        loop((0, 3))
        return measure(wires=0)

    msg = "Unrecoverable error"
    # Exception in beginning
    with pytest.raises(RuntimeError, match=msg):
        circuit(0)
    # Exception in middle
    with pytest.raises(RuntimeError, match=msg):
        circuit(1)
    # Exception in end
    with pytest.raises(RuntimeError, match=msg):
        circuit(2)


def test_exception_in_for_loop(backend):
    "Test exception happening in a loop."

    @qjit(async_qnodes=True)
    @qml.qnode(qml.device(backend, wires=1))
    def circuit(n):
        @for_loop(0, 1, n)
        def loop(i):
            qml.CNOT(wires=[0, i])

        return loop()

    msg = "Unrecoverable error"
    # Exception in beginning
    with pytest.raises(RuntimeError, match=msg):
        circuit(1)


def test_exception_in_loop2(backend):
    "Test exception happening in a loop while one qnode succeeds."

    @qml.qnode(qml.device(backend, wires=3))
    def bad(n):
        @while_loop(lambda v: v[0] < v[1])
        def loop(v):
            qml.CNOT(wires=[n, v[0]])
            return v[0] + 1, v[1]

        loop((0, 3))
        return measure(wires=0)

    @qml.qnode(qml.device(backend, wires=3))
    def good():
        return qml.state()

    @qjit(async_qnodes=True)
    def wrapper(n):
        x = good()
        y = bad(n)
        return x + y

    msg = "Unrecoverable error"
    # Exception in beginning
    with pytest.raises(RuntimeError, match=msg):
        wrapper(0)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
