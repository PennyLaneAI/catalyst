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

"""Test built-in differentiation support in Catalyst."""

import jax
import numpy as np
import pennylane as qml
import pytest
from jax import numpy as jnp
from jax.tree_util import tree_flatten

import catalyst.utils.calculate_grad_shape as infer
from catalyst import (
    CompileError,
    DifferentiableCompileError,
    cond,
    for_loop,
    grad,
    jacobian,
    measure,
    mitigate_with_zne,
    pure_callback,
    qjit,
    value_and_grad,
)

# pylint: disable=too-many-lines


class TestGradShape:
    """Unit tests for the calculate_grad_shape module."""

    @pytest.mark.parametrize("sig", [infer.Signature([int], [int])])
    def test_repr(self, sig):
        """Sanity check to make sure that we have a human readable representation."""
        assert str(sig) == "[<class 'int'>] -> [<class 'int'>]"

    def test_deduction(self):
        """Test case from https://github.com/PennyLaneAI/catalyst/issues/83"""
        params = [jax.core.ShapedArray([], float)]
        returns = [jax.core.ShapedArray([2], float), jax.core.ShapedArray([], float)]
        in_signature = infer.Signature(params, returns)
        observed_output = infer.calculate_grad_shape(in_signature, [0])
        expected_output = infer.Signature(params, returns)
        assert observed_output == expected_output

    def test_deduction_float(self):
        """Test case for non tensor type."""
        params = [float]
        returns = [float, float]
        in_signature = infer.Signature(params, returns)
        with pytest.raises(TypeError, match="Inputs and results must be tensor type."):
            infer.calculate_grad_shape(in_signature, [0])


def test_grad_outside_qjit():
    """Test that grad can be used outside of a jitting context."""

    def f(x):
        return x**2

    x = 4.0

    expected = jax.grad(f)(x)
    result = grad(f)(x)

    assert np.allclose(expected, result)


def test_value_and_grad_outside_qjit():
    """Test that value_and_grad can be used outside of a jitting context."""

    def f(x):
        return x**2

    x = 4.0

    expected_val, expected_grad = jax.value_and_grad(f)(x)
    result_val, result_grad = value_and_grad(f)(x)

    assert np.allclose(expected_val, result_val)
    assert np.allclose(expected_grad, result_grad)


@pytest.mark.parametrize("argnum", (None, 0, [1], (0, 1)))
def test_grad_outside_qjit_argnum(argnum):
    """Test that argnums work correctly outside of a jitting context."""

    def f(x, y):
        return x**2 + y**2

    x, y = 4.0, 4.0

    expected = jax.grad(f, argnums=argnum if argnum is not None else 0)(x, y)
    result = grad(f, argnum=argnum)(x, y)

    assert np.allclose(expected, result)


@pytest.mark.parametrize("argnum", (None, 0, [1], (0, 1)))
def test_value_and_grad_outside_qjit_argnum(argnum):
    """Test that argnums work correctly outside of a jitting context."""

    def f(x, y):
        return x**2 + y**2

    x, y = 4.0, 4.0

    expected_val, expected_grad = jax.value_and_grad(
        f, argnums=argnum if argnum is not None else 0
    )(x, y)
    result_val, result_grad = value_and_grad(f, argnum=argnum)(x, y)

    assert np.allclose(expected_val, result_val)
    assert np.allclose(expected_grad, result_grad)


def test_jacobian_outside_qjit():
    """Test that jacobian can be used outside of a jitting context."""

    def f(x):
        return x**2, 2 * x

    x = jnp.array([4.0, 5.0])

    expected = jax.jacobian(f)(x)
    result = jacobian(f)(x)

    assert len(expected) == len(result) == 2
    assert np.allclose(expected[0], result[0])
    assert np.allclose(expected[1], result[1])


@pytest.mark.parametrize("argnum", (None, 0, [1], (0, 1)))
def test_jacobian_outside_qjit_argnum(argnum):
    """Test that argnums work correctly outside of a jitting context."""

    def f(x, y):
        return x**2 + y**2, 2 * x + 2 * y

    x, y = jnp.array([4.0, 5.0]), jnp.array([4.0, 5.0])

    expected = jax.jacobian(f, argnums=argnum if argnum is not None else 0)(x, y)
    result = jacobian(f, argnum=argnum)(x, y)

    assert len(expected) == len(result) == 2
    assert np.allclose(expected[0], result[0])
    assert np.allclose(expected[1], result[1])


def test_non_differentiable_qnode():
    """Check for an error message when the QNode is explicitly marked non-differentiable."""

    @qml.qnode(qml.device("lightning.qubit", wires=1), diff_method=None)
    def f(x: float):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(wires=0))

    # Ensure None allows forward-pass to succeed
    assert np.allclose(qjit(f)(1.0), np.cos(1.0))

    @qjit
    def grad_f(x):
        return grad(f, method="auto")(x)

    with pytest.raises(
        DifferentiableCompileError,
        match="Cannot differentiate a QNode explicitly marked non-differentiable",
    ):
        grad_f(1.0)


def test_param_shift_on_non_expval(backend):
    """Check for an error message when parameter-shift is used on QNodes that return anything but
    qml.expval or qml.probs.
    """

    @qml.qnode(qml.device(backend, wires=1), diff_method="parameter-shift")
    def func(p):
        x = qml.expval(qml.PauliZ(0))
        y = p**2
        return x, y

    def workflow(p: float):
        return jacobian(func, method="auto")(p)

    with pytest.raises(
        DifferentiableCompileError, match="The parameter-shift method can only be used"
    ):
        qjit(workflow)


def test_adjoint_on_non_expval(backend):
    """Check for an error message when adjoint is used on QNodes that return anything but
    qml.expval or qml.probs.
    """

    @qml.qnode(qml.device(backend, wires=1), diff_method="adjoint")
    def func(p):
        x = qml.expval(qml.PauliZ(0))
        y = p**2
        return x, y

    def workflow(p: float):
        return jacobian(func, method="auto")(p)

    with pytest.raises(DifferentiableCompileError, match="The adjoint method can only be used"):
        qjit(workflow)


def test_grad_on_qjit():
    """Check that grad works when called on an existing qjit object that does not wrap a QNode."""

    @qjit
    def f(x: float):
        return x * x

    result = qjit(grad(f))(3.0)
    expected = 6.0

    assert np.allclose(result, expected)


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_finite_diff(inp, backend):
    """Test finite diff."""

    def f(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled_grad_default(x: float):
        g = qml.qnode(qml.device(backend, wires=1))(f)
        h = grad(g, method="fd")
        return h(x)

    def interpretted_grad_default(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f, device, diff_method="finite-diff")
        h = qml.grad(g, argnum=0)
        return h(x)

    assert np.allclose(compiled_grad_default(inp), interpretted_grad_default(inp))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_finite_diff_mul(inp, backend):
    """Test finite diff with mul."""

    def f(x):
        qml.RX(3 * x, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled_grad_default(x: float):
        g = qml.qnode(qml.device(backend, wires=1))(f)
        h = grad(g, method="fd")
        return h(x)

    def interpretted_grad_default(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f, device, diff_method="finite-diff")
        h = qml.grad(g, argnum=0)
        return h(x)

    assert np.allclose(compiled_grad_default(inp), interpretted_grad_default(inp))


@pytest.mark.parametrize("inp", [1.0, 2.0, 3.0, 4.0])
def test_finite_diff_in_loop(inp, backend):
    """Test finite diff in loop."""

    @qml.qnode(qml.device(backend, wires=1))
    def f(x):
        qml.RX(3 * x, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit
    def compiled_grad_default(params, ntrials):
        diff = grad(f, argnum=0, method="fd")

        def fn(i, g):
            return diff(params)

        return for_loop(0, ntrials, 1)(fn)(params)

    def interpretted_grad_default(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f, device, diff_method="finite-diff")
        h = qml.grad(g, argnum=0)
        return h(x)

    assert np.allclose(compiled_grad_default(inp, 5), interpretted_grad_default(inp))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_adj(inp, backend):
    """Test the adjoint method."""

    def f(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled(x: float):
        g = qml.qnode(qml.device(backend, wires=1), diff_method="adjoint")(f)
        h = grad(g, method="auto")
        return h(x)

    def interpreted(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f, device, diff_method="backprop")
        h = qml.grad(g, argnum=0)
        return h(x)

    assert np.allclose(compiled(inp), interpreted(inp))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_adj_mult(inp, backend):
    """Test the adjoint method with mult."""

    def f(x):
        qml.RX(x * 2, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled(x: float):
        g = qml.qnode(qml.device(backend, wires=1), diff_method="adjoint")(f)
        h = grad(g, method="auto")
        return h(x)

    def interpreted(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f, device, diff_method="backprop")
        h = qml.grad(g, argnum=0)
        return h(x)

    assert np.allclose(compiled(inp), interpreted(inp))


@pytest.mark.parametrize("inp", [1.0, 2.0, 3.0, 4.0])
def test_adj_in_loop(inp, backend):
    """Test the adjoint method in loop."""

    @qml.qnode(qml.device(backend, wires=1), diff_method="adjoint")
    def f(x):
        qml.RX(3 * x, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled_grad_default(params, ntrials):
        diff = grad(f, argnum=0, method="auto")

        def fn(i, g):
            return diff(params)

        return for_loop(0, ntrials, 1)(fn)(params)

    def interpretted_grad_default(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f, device, diff_method="backprop")
        h = qml.grad(g, argnum=0)
        return h(x)

    assert np.allclose(compiled_grad_default(inp, 5), interpretted_grad_default(inp))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_ps(inp, backend):
    """Test the ps method."""

    def f(x):
        qml.RX(x * 2, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled(x: float):
        g = qml.qnode(qml.device(backend, wires=1), diff_method="parameter-shift")(f)
        h = grad(g, method="auto")
        return h(x)

    def interpreted(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f, device, diff_method="backprop")
        h = qml.grad(g, argnum=0)
        return h(x)

    assert np.allclose(compiled(inp), interpreted(inp))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_ps_conditionals(inp, backend):
    """Test the ps method and conditionals."""

    def f_compiled(x, y):
        @cond(y > 1.5)
        def true_path():
            qml.RX(x * 2, wires=0)

        @true_path.otherwise
        def false_path():
            qml.RX(x, wires=0)

        true_path()
        return qml.expval(qml.PauliY(0))

    def f_interpreted(x, y):
        if y > 1.5:
            qml.RX(x * 2, wires=0)
        else:
            qml.RX(x, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled(x: float, y: float):
        g = qml.qnode(qml.device(backend, wires=1), diff_method="parameter-shift")(f_compiled)
        h = grad(g, method="auto", argnum=0)
        return h(x, y)

    def interpreted(x, y):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f_interpreted, device, diff_method="backprop")
        h = qml.grad(g, argnum=0)
        return h(x, y)

    assert np.allclose(compiled(inp, 0.0), interpreted(inp, 0.0))
    assert np.allclose(compiled(inp, 2.0), interpreted(inp, 2.0))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_ps_for_loops(inp, backend):
    """Test the ps method with for loops."""

    def f_compiled(x, y):
        @for_loop(0, y, 1)
        def loop_fn(i):
            qml.RX(x * i * 1.5, wires=0)

        loop_fn()
        return qml.expval(qml.PauliY(0))

    def f_interpreted(x, y):
        for i in range(0, y, 1):
            qml.RX(x * i * 1.5, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled(x: float, y: int):
        g = qml.qnode(qml.device(backend, wires=1), diff_method="parameter-shift")(f_compiled)
        h = grad(g, method="auto", argnum=0)
        return h(x, y)

    def interpreted(x, y):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f_interpreted, device, diff_method="backprop")
        h = qml.grad(g, argnum=0)
        return h(x, y)

    assert np.allclose(compiled(inp, 1), interpreted(inp, 1))
    assert np.allclose(compiled(inp, 2), interpreted(inp, 2))
    assert np.allclose(compiled(inp, 3), interpreted(inp, 3))
    assert np.allclose(compiled(inp, 4), interpreted(inp, 4))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_ps_for_loops_entangled(inp, backend):
    """Test the ps method with for loops and entangled."""

    def f_compiled(x, y, z):
        qml.RX(x, wires=0)
        qml.Hadamard(wires=0)

        @for_loop(1, y, 1)
        def loop_fn(i):
            qml.RX(x, wires=i)
            qml.CNOT(wires=[0, i])

        loop_fn()
        return qml.expval(qml.PauliY(z))

    def f_interpreted(x, y, z):
        qml.RX(x, wires=0)
        qml.Hadamard(wires=0)
        for i in range(1, y, 1):
            qml.RX(x, wires=i)
            qml.CNOT(wires=[0, i])
        return qml.expval(qml.PauliY(z))

    @qjit()
    def compiled(x: float, y: int, z: int):
        g = qml.qnode(qml.device(backend, wires=3), diff_method="parameter-shift")(f_compiled)
        h = grad(g, method="auto", argnum=0)
        return h(x, y, z)

    def interpreted(x, y, z):
        device = qml.device("default.qubit", wires=3)
        g = qml.QNode(f_interpreted, device, diff_method="backprop")
        h = qml.grad(g, argnum=0)
        return h(x, y, z)

    assert np.allclose(compiled(inp, 1, 1), interpreted(inp, 1, 1))
    assert np.allclose(compiled(inp, 2, 2), interpreted(inp, 2, 2))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_ps_qft(inp, backend):
    """Test the ps method in QFT."""

    def qft_compiled(x, n, z):
        # Input state: equal superposition
        @for_loop(0, n, 1)
        def init(i):
            qml.Hadamard(wires=i)

        # QFT
        @for_loop(0, n, 1)
        def qft(i):
            qml.Hadamard(wires=i)

            @for_loop(i + 1, n, 1)
            def inner(j):
                qml.RY(x, wires=j)
                qml.ControlledPhaseShift(jnp.pi / 2 ** (n - j + 1), [i, j])

            inner()

        init()
        qft()

        # Expected output: |100...>
        return qml.expval(qml.PauliZ(z))

    def qft_interpreted(x, n, z):
        # Input state: equal superposition
        for i in range(0, n, 1):
            qml.Hadamard(wires=i)

        for i in range(0, n, 1):
            qml.Hadamard(wires=i)

            for j in range(i + 1, n, 1):
                qml.RY(x, wires=j)
                qml.ControlledPhaseShift(jnp.pi / 2 ** (n - j + 1), [i, j])

        return qml.expval(qml.PauliZ(z))

    @qjit()
    def compiled(x: float, y: int, z: int):
        g = qml.qnode(qml.device(backend, wires=3), diff_method="parameter-shift")(qft_compiled)
        h = grad(g, method="auto", argnum=0)
        return h(x, y, z)

    def interpreted(x, y, z):
        device = qml.device("default.qubit", wires=3)
        g = qml.QNode(qft_interpreted, device, diff_method="backprop")
        h = qml.grad(g, argnum=0)
        return h(x, y, z)

    assert np.allclose(compiled(inp, 2, 2), interpreted(inp, 2, 2))


def test_ps_probs(backend):
    """Check that the parameter-shift method works for qml.probs."""

    @qml.qnode(qml.device(backend, wires=1), diff_method="parameter-shift")
    def func(p):
        qml.RY(p, wires=0)
        return qml.probs(wires=0)

    @qjit
    def workflow(p: float):
        return jacobian(func, method="auto")(p)

    result = workflow(0.5)
    reference = qml.jacobian(func, argnum=0)(0.5)

    assert np.allclose(result, reference)


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_finite_diff_h(inp, backend):
    """Test finite diff."""

    def f(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled_grad_h(x: float):
        g = qml.qnode(qml.device(backend, wires=1))(f)
        h = grad(g, method="fd", h=0.1)
        return h(x)

    def interpretted_grad_h(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f, device, diff_method="finite-diff", h=0.1)
        h = qml.grad(g, argnum=0)
        return h(x)

    assert np.allclose(compiled_grad_h(inp), interpretted_grad_h(inp))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_finite_diff_argnum(inp, backend):
    """Test finite diff."""

    def f2(x, y):
        qml.RX(x**y, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled_grad_argnum(x: float):
        g = qml.qnode(qml.device(backend, wires=1))(f2)
        h = grad(g, method="fd", argnum=1)
        return h(x, 2.0)

    def interpretted_grad_argnum(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f2, device, diff_method="finite-diff")
        h = qml.grad(g, argnum=1)
        return h(x, 2.0)

    assert np.allclose(compiled_grad_argnum(inp), interpretted_grad_argnum(inp))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_finite_diff_argnum_list(inp, backend):
    """Test finite diff."""

    def f2(x, y):
        qml.RX(x**y, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled_grad_argnum_list(x: float):
        g = qml.qnode(qml.device(backend, wires=1))(f2)
        h = grad(g, method="fd", argnum=[1])
        return h(x, 2.0)

    def interpretted_grad_argnum_list(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f2, device, diff_method="finite-diff")
        h = qml.grad(g, argnum=[1])
        # Slightly different behaviour. If argnum is a list
        # it doesn't matter if it is a single number,
        # the return value will be a n-tuple of size of the
        # argnum list.
        return h(x, 2.0)[0]

    assert np.allclose(compiled_grad_argnum_list(inp), interpretted_grad_argnum_list(inp))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_finite_grad_range_change(inp, backend):
    """Test finite diff."""

    def f2(x, y):
        qml.RX(x**y, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled_grad_range_change(x: float):
        g = qml.qnode(qml.device(backend, wires=1))(f2)
        h = grad(g, method="fd", argnum=[0, 1])
        return h(x, 2.0)

    def interpretted_grad_range_change(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f2, device, diff_method="finite-diff")
        h = qml.grad(g, argnum=[0, 1])
        return h(x, 2.0)

    assert np.allclose(compiled_grad_range_change(inp), interpretted_grad_range_change(inp))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_ps_grad_range_change(inp, backend):
    """Test param shift."""

    def f2(x, y):
        qml.RX(x**y, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled_grad_range_change(x: float):
        g = qml.qnode(qml.device(backend, wires=1), diff_method="parameter-shift")(f2)
        h = grad(g, method="auto", argnum=[0, 1])
        return h(x, 2.0)

    def interpretted_grad_range_change(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f2, device, diff_method="backprop")
        h = qml.grad(g, argnum=[0, 1])
        return h(x, 2.0)

    assert np.allclose(compiled_grad_range_change(inp), interpretted_grad_range_change(inp))


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_ps_tensorinp(inp, backend):
    """Test param shift."""

    def f2(x, y):
        qml.RX(x[0] ** y, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled(x: jax.core.ShapedArray([1], float)):
        g = qml.qnode(qml.device(backend, wires=1), diff_method="parameter-shift")(f2)
        h = grad(g, method="auto", argnum=[0, 1])
        return h(x, 2.0)

    def interpretted(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f2, device, diff_method="backprop")
        h = qml.grad(g, argnum=[0, 1])
        return h(x, 2.0)

    for dydx_c, dydx_i in zip(compiled(jnp.array([inp])), interpretted(np.array([inp]))):
        assert np.allclose(dydx_c, dydx_i)


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_adjoint_grad_range_change(inp, backend):
    """Test adjoint."""

    def f2(x, y):
        qml.RX(x**y, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled_grad_range_change(x: float):
        g = qml.qnode(qml.device(backend, wires=1), diff_method="adjoint")(f2)
        h = grad(g, method="auto", argnum=[0, 1])
        return h(x, 2.0)

    def interpretted_grad_range_change(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f2, device, diff_method="backprop")
        h = qml.grad(g, argnum=[0, 1])
        return h(x, 2.0)

    assert np.allclose(compiled_grad_range_change(inp), interpretted_grad_range_change(inp))


@pytest.mark.parametrize("method", [("parameter-shift"), ("adjoint")])
def test_assert_no_higher_order_without_fd(method, backend):
    """Test input validation for gradients"""

    def f(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliY(0))

    with pytest.raises(DifferentiableCompileError, match="higher order derivatives"):

        @qjit()
        def workflow(x: float):
            g = qml.qnode(qml.device(backend, wires=1), diff_method=method)(f)
            h = grad(g, method="auto")
            i = grad(h, method="auto")
            return i(x)


def test_assert_invalid_diff_method():
    """Test invalid diff method detection"""

    def f(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliY(0))

    with pytest.raises(ValueError, match="Invalid differentiation method"):

        @qjit()
        def workflow(x: float):
            g = qml.qnode(qml.device("lightning.qubit", wires=1))(f)
            h = grad(g, method="non-existent method")
            return h(x)


def test_assert_invalid_h_type():
    """Test invalid h type detection"""

    def f(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliY(0))

    with pytest.raises(ValueError, match="Invalid h value"):

        @qjit()
        def workflow(x: float):
            g = qml.qnode(qml.device("lightning.qubit", wires=1))(f)
            h = grad(g, method="fd", h="non-integer")
            return h(x)


def test_assert_non_differentiable():
    """Test non-differentiable parameter detection"""
    with pytest.raises(DifferentiableCompileError, match="Non-differentiable object passed"):

        @qjit()
        def workflow(x: float):
            h = grad("string!", method="fd")
            return h(x)


def test_finite_diff_arbitrary_functions():
    """Test gradients on non-qnode functions."""

    @qjit
    def workflow(x):
        def _f(x):
            return 2 * x

        return grad(_f, method="fd")(x)

    assert workflow(0.0) == 2.0


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_finite_diff_higher_order(inp, backend):
    """Test finite diff."""

    def f(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled_grad2_default(x: float):
        g = qml.qnode(qml.device(backend, wires=1))(f)
        h = grad(g, method="fd")
        i = grad(h, method="fd")
        return i(x)

    def interpretted_grad2_default(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f, device, diff_method="backprop", max_diff=2)
        h = qml.grad(g, argnum=0)
        i = qml.grad(h, argnum=0)
        return i(x)

    assert np.allclose(compiled_grad2_default(inp), interpretted_grad2_default(inp), rtol=0.1)


@pytest.mark.parametrize("g_method", ["fd", "auto"])
@pytest.mark.parametrize(
    "h_coeffs", [[0.2, -0.53], np.array([0.2, -0.53]), jnp.array([0.2, -0.53])]
)
def test_jax_consts(h_coeffs, g_method, backend):
    """Test jax constants."""

    def circuit(params):
        qml.CRX(params[0], wires=[0, 1])
        qml.CRX(params[0], wires=[0, 2])
        h_obs = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.Hadamard(2)]
        return qml.expval(qml.Hamiltonian(h_coeffs, h_obs))

    @qjit()
    def compile_grad(params):
        diff_method = "adjoint" if g_method == "auto" else "finite-diff"
        g = qml.qnode(qml.device(backend, wires=3), diff_method=diff_method)(circuit)
        h = grad(g, method=g_method)
        return h(params)

    def interpret_grad(params):
        device = qml.device("default.qubit", wires=3)
        g = qml.QNode(circuit, device, diff_method="backprop")
        h = jax.grad(g, argnums=0)
        return h(params)

    inp = jnp.array([1.0, 2.0])
    assert np.allclose(compile_grad(jnp.array(inp)), interpret_grad(inp))


def test_non_float_arg(backend):
    """Test a function which attempts to differentiate non-floating point arguments."""

    @qml.qnode(qml.device(backend, wires=2))
    def circuit(x: complex, y: float):
        qml.RX(jnp.real(x), wires=0)
        qml.RY(y, wires=0)
        return qml.expval(qml.PauliZ(0))

    @qjit
    def cost_fn(x, y):
        return grad(circuit)(x, y)

    with pytest.raises(
        DifferentiableCompileError,
        match="only supports differentiation on floating-point arguments",
    ):
        cost_fn(1j, 2.0)


def test_non_float_res(backend):
    """Test a function which attempts to differentiate non-floating point results."""

    @qml.qnode(qml.device(backend, wires=2))
    def circuit(x: float, y: float):
        qml.RX(x, wires=0)
        qml.RY(y, wires=0)
        return qml.expval(qml.PauliZ(0))

    @qjit
    @grad
    def cost_fn(x, y):
        return 1j * circuit(x, y)

    with pytest.raises(
        DifferentiableCompileError, match="only supports differentiation on floating-point results"
    ):
        cost_fn(1.0, 2.0)


@pytest.mark.parametrize("diff_method", ["fd", "auto"])
@pytest.mark.parametrize("inp", [(1.0), (2.0)])
def test_finite_diff_multiple_devices(inp, diff_method, backend):
    """Test gradient methods using multiple backend devices."""
    qnode_diff_method = "adjoint" if diff_method == "auto" else "finite-diff"

    @qml.qnode(qml.device(backend, wires=1), diff_method=qnode_diff_method)
    def f(x):
        qml.RX(3 * x, wires=0)
        return qml.expval(qml.PauliY(0))

    @qml.qnode(qml.device("lightning.qubit", wires=1), diff_method=qnode_diff_method)
    def g(x):
        qml.RX(3 * x, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled_grad_default(params, ntrials):
        d_f = grad(f, argnum=0, method=diff_method)

        def fn_f(_i, _g):
            return d_f(params)

        d_g = grad(g, argnum=0, method=diff_method)

        def fn_g(_i, _g):
            return d_g(params)

        d1 = for_loop(0, ntrials, 1)(fn_f)(params)
        d2 = for_loop(0, ntrials, 1)(fn_g)(params)
        return d1, d2

    result = compiled_grad_default(inp, 5)
    assert np.allclose(result[0], result[1])


def test_grad_on_non_scalar_output(backend):
    """Test a function which attempts to use `grad` on a function that returns a non-scalar."""

    @qml.qnode(qml.device(backend, wires=1), diff_method="parameter-shift")
    def f(x):
        qml.RX(3 * x, wires=0)
        return qml.probs()

    @qjit
    def compiled(x):
        return grad(f)(x)

    with pytest.raises(DifferentiableCompileError, match="only supports scalar-output functions"):
        compiled(1.0)


def test_grad_on_multi_result_function(backend):
    """Test a function which attempts to use `grad` on a function that returns multiple values."""

    @qml.qnode(qml.device(backend, wires=2), diff_method="parameter-shift")
    def f(x):
        qml.RX(3 * x, wires=0)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

    @qjit
    def compiled(x):
        return grad(f)(x)

    with pytest.raises(DifferentiableCompileError, match="only supports scalar-output functions"):
        compiled(1.0)


def test_multiple_grad_invocations(backend):
    """Test a function that uses grad multiple times."""

    @qml.qnode(qml.device(backend, wires=2), diff_method="parameter-shift")
    def f(x, y):
        qml.RX(3 * x, wires=0)
        qml.RX(y, wires=0)
        return qml.expval(qml.PauliZ(0))

    @qjit
    def compiled(x: float, y: float):
        g1 = grad(f, argnum=0, method="auto")(x, y)
        g2 = grad(f, argnum=1, method="auto")(x, y)
        return jnp.array([g1, g2])

    actual = compiled(0.1, 0.2)
    expected = jax.jacobian(f, argnums=(0, 1))(0.1, 0.2)
    for actual_entry, expected_entry in zip(actual, expected):
        assert actual_entry == pytest.approx(expected_entry)


def test_loop_with_dyn_wires(backend):
    """Test the gradient on a function with a loop and modular wire arithmetic."""
    num_wires = 4
    dev = qml.device(backend, wires=num_wires)

    @qml.qnode(dev)
    def cat(phi):
        @for_loop(0, 3, 1)
        def loop(i):
            qml.RY(phi, wires=jnp.mod(i, num_wires))

        loop()

        return qml.expval(qml.prod(*[qml.PauliZ(i) for i in range(num_wires)]))

    @qml.qnode(dev)
    def pl(phi):
        @for_loop(0, 3, 1)
        def loop(i):
            qml.RY(phi, wires=i % num_wires)

        loop()

        return qml.expval(qml.prod(*[qml.PauliZ(i) for i in range(num_wires)]))

    arg = 0.75
    result = qjit(grad(cat))(arg)
    expected = qml.grad(pl, argnum=0)(arg)

    assert np.allclose(result, expected)


def test_pytrees_return_qnode(backend):
    """Test the gradient on a function with a return including list and dictionnaries"""
    num_wires = 1
    dev = qml.device(backend, wires=num_wires)

    @qml.qnode(dev)
    def circuit(phi, psi):
        qml.RY(phi, wires=0)
        qml.RX(psi, wires=0)
        return [{"expval0": qml.expval(qml.PauliZ(0))}, qml.expval(qml.PauliZ(0))]

    psi = 0.1
    phi = 0.2
    result = qjit(jacobian(circuit, argnum=[0, 1]))(psi, phi)

    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], dict)
    assert isinstance(result[0]["expval0"], tuple)
    assert len(result[0]["expval0"]) == 2
    assert isinstance(result[1], tuple)
    assert len(result[1]) == 2


def test_pytrees_return_classical_function(backend):
    """Test the jacobian on a qnode with a return including list and dictionnaries."""
    num_wires = 1
    dev = qml.device(backend, wires=num_wires)

    @qml.qnode(dev)
    def circuit(phi, psi):
        qml.RY(phi, wires=0)
        qml.RX(psi, wires=0)
        return [{"expval0": qml.expval(qml.PauliZ(0))}, qml.expval(qml.PauliZ(0))]

    psi = 0.1
    phi = 0.2
    result = qjit(jacobian(circuit, argnum=[0, 1]))(psi, phi)

    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], dict)
    assert isinstance(result[0]["expval0"], tuple)
    assert len(result[0]["expval0"]) == 2
    assert isinstance(result[1], tuple)
    assert len(result[1]) == 2


def test_pytrees_return_classical():
    """Test the jacobian on a function with a return including list and dictionnaries."""

    def f(x, y):
        return [x, {"a": x**2}, x + y]

    x = 0.4
    y = 0.2

    jax_expected_results = jax.jit(jax.jacobian(f, argnums=[0, 1]))(x, y)
    catalyst_results = qjit(jacobian(f, argnum=[0, 1]))(x, y)

    flatten_res_jax, tree_jax = tree_flatten(jax_expected_results)
    flatten_res_catalyst, tree_catalyst = tree_flatten(catalyst_results)

    assert tree_jax == tree_catalyst
    assert np.allclose(flatten_res_jax, flatten_res_catalyst)


def test_pytrees_args_classical():
    """Test the jacobian on a function with a return including list and dictionnaries."""

    def f(x, y):
        return x["res1"], x["res2"] + y

    x = {"res1": 0.3, "res2": 0.4}
    y = 0.2

    jax_expected_results = jax.jit(jax.jacobian(f, argnums=[0, 1]))(x, y)
    catalyst_results = qjit(jacobian(f, argnum=[0, 1]))(x, y)

    flatten_res_jax, tree_jax = tree_flatten(jax_expected_results)
    flatten_res_catalyst, tree_catalyst = tree_flatten(catalyst_results)

    assert tree_jax == tree_catalyst
    assert np.allclose(flatten_res_jax, flatten_res_catalyst)


def test_pytrees_args_return_classical():
    """Test the jacobian on a function with a args and return including list and dictionnaries."""

    def f(x, y):
        return [{"res": x["args1"], "res2": x["args2"] + y}, x["args1"] + y]

    x = {"args1": 0.3, "args2": 0.4}
    y = 0.2

    jax_expected_results = jax.jit(jax.jacobian(f, argnums=[0, 1]))(x, y)
    catalyst_results = qjit(jacobian(f, argnum=[0, 1]))(x, y)

    flatten_res_jax, tree_jax = tree_flatten(jax_expected_results)
    flatten_res_catalyst, tree_catalyst = tree_flatten(catalyst_results)

    assert tree_jax == tree_catalyst
    assert np.allclose(flatten_res_jax, flatten_res_catalyst)


@pytest.mark.xfail(reason="QubitUnitrary is not support with catalyst.grad")
@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_adj_qubitunitary(inp, backend):
    """Test the adjoint method."""

    def f(x):
        qml.RX(x, wires=0)
        U1 = 1 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex)
        qml.QubitUnitary(U1, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled(x: float):
        g = qml.qnode(qml.device(backend, wires=1), diff_method="adjoint")(f)
        h = grad(g, method="auto")
        return h(x)

    def interpreted(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f, device, diff_method="backprop")
        h = qml.grad(g, argnum=0)
        return h(x)

    assert np.allclose(compiled(inp), interpreted(inp))


class TestGradientErrors:
    """Test errors when an operation which does not have a valid gradient is reachable
    from the grad op"""

    def test_measure_error(self):
        """Test with measure"""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f(x):
            qml.RX(x, wires=0)
            _bool = measure(0)
            qml.RX(_bool + 1, wires=0)
            return qml.expval(qml.PauliX(0))

        with pytest.raises(CompileError, match=".*Compilation failed.*"):

            @qml.qjit
            def cir(x: float):
                return grad(f)(x)

    def test_callback_error(self):
        """Test with callback"""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f(x):
            y = pure_callback(jnp.sin, float)(x)
            qml.RX(y, wires=0)
            return qml.expval(qml.PauliX(0))

        with pytest.raises(CompileError, match=".*Compilation failed.*"):

            @qml.qjit
            def cir(x: float):
                return grad(f)(x)

    def test_with_zne(self):
        """Test with ZNE"""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliX(0))

        def g(x):
            return mitigate_with_zne(f, scale_factors=jax.numpy.array([1, 2, 3]), deg=2)(x)

        with pytest.raises(CompileError, match=".*Compilation failed.*"):

            @qml.qjit
            def cir(x: float):
                return grad(g)(x)


class TestGradientUsagePatterns:
    """Test usage patterns of Gradient functions"""

    def test_grad_usage_patterns(self):
        """Test usage patterns of catalyst.grad."""

        def fn(x):
            return x**2

        x = 4.0

        res_pattern_fn_as_argument = grad(fn)(x)
        res_pattern_partial = grad()(fn)(x)
        expected = jax.grad(fn)(x)
        
        assert np.allclose(res_pattern_fn_as_argument, expected)
        assert np.allclose(res_pattern_partial, expected)

    def test_value_and_grad_usage_patterns(self):
        """Test usage patterns of catalyst.value_and_grad."""

        def fn(x):
            return x**2

        x = 4.0

        fn_as_argument_val, fn_as_argument_grad = value_and_grad(fn)(x)
        partial_val, partial_grad = value_and_grad()(fn)(x)
        expected_val, expected_grad = jax.value_and_grad(fn)(x)

        assert np.allclose(fn_as_argument_val, expected_val)
        assert np.allclose(partial_val, expected_val)
        assert np.allclose(fn_as_argument_grad, expected_grad)
        assert np.allclose(partial_grad, expected_grad)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
