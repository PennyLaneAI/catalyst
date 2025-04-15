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

import platform

import jax
import numpy as np
import pennylane as qml
import pytest
from jax import numpy as jnp
from jax.tree_util import tree_all, tree_flatten, tree_map, tree_structure

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
    vmap,
)
from catalyst.compiler import get_lib_path
from catalyst.device.op_support import (
    _are_param_frequencies_same_as_catalyst,
    _has_grad_recipe,
    _has_parameter_frequencies,
    _is_grad_recipe_same_as_catalyst,
    _paramshift_op_checker,
)
from catalyst.jax_tracer import HybridOp

# pylint: disable=too-many-lines,missing-function-docstring,missing-class-docstring


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


def test_gradient_generate_once():
    """Test that gradients are only generated once even if
    they are called multiple times. This is already tested
    in lit tests, but lit tests are not counted in coverage
    """

    def identity(x):
        return x

    @qjit
    def wrap(x: float):
        diff = grad(identity)
        return diff(x) + diff(x)

    assert "@identity_0" not in wrap.mlir


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


@pytest.mark.parametrize("argnums", (None, 0, [1], (0, 1)))
def test_grad_outside_qjit_argnum(argnums):
    """Test that argnums work correctly outside of a jitting context."""

    def f(x, y):
        return x**2 + y**2

    x, y = 4.0, 4.0

    expected = jax.grad(f, argnums=argnums if argnums is not None else 0)(x, y)
    result = grad(f, argnums=argnums)(x, y)

    assert np.allclose(expected, result)


@pytest.mark.parametrize("argnums", (None, 0, [1], (0, 1)))
def test_value_and_grad_outside_qjit_argnum(argnums):
    """Test that argnums work correctly outside of a jitting context."""

    def f(x, y):
        return x**2 + y**2

    x, y = 4.0, 4.0

    expected_val, expected_grad = jax.value_and_grad(
        f, argnums=argnums if argnums is not None else 0
    )(x, y)
    result_val, result_grad = value_and_grad(f, argnums=argnums)(x, y)

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


@pytest.mark.parametrize("argnums", (None, 0, [1], (0, 1)))
def test_jacobian_outside_qjit_argnums(argnums):
    """Test that argnums work correctly outside of a jitting context."""

    def f(x, y):
        return x**2 + y**2, 2 * x + 2 * y

    x, y = jnp.array([4.0, 5.0]), jnp.array([4.0, 5.0])

    expected = jax.jacobian(f, argnums=argnums if argnums is not None else 0)(x, y)
    result = jacobian(f, argnums=argnums)(x, y)

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


def test_value_and_grad_on_qjit_classical():
    """Check that value_and_grad works when called on an qjit object that does not wrap a QNode."""

    @qjit
    def f1(x: float):
        return x * x

    result = qjit(value_and_grad(f1))(3.0)
    expected = (9.0, 6.0)
    assert np.allclose(result, expected)

    @qjit
    def f2(x: float):
        return [x * x]

    result = qjit(value_and_grad(f2))(3.0)
    expected = ([9.0], [6.0])
    assert np.allclose(result, expected)

    @qjit
    def f3(x: float):
        return {"helloworld": x * x}

    result = qjit(value_and_grad(f3))(3.0)
    expected = ({"helloworld": 9.0}, {"helloworld": 6.0})
    assert np.allclose(result[0]["helloworld"], expected[0]["helloworld"])
    assert np.allclose(result[1]["helloworld"], expected[1]["helloworld"])

    @qjit
    def f4(x: float, y: float, z: float):
        return 100 * x + 200 * y + 300 * z

    result = qjit(value_and_grad(f4))(0.1, 0.2, 0.3)
    expected = (140, 100)
    assert np.allclose(result, expected)

    @qjit
    def f5(x: float, y: float, z: float):
        return 100 * x + 200 * y + 300 * z

    result = qjit(value_and_grad(f5, argnums=(0, 1, 2)))(0.1, 0.2, 0.3)
    expected = (140, (100, 200, 300))
    assert np.allclose(result[0], expected[0])
    assert np.allclose(result[1], expected[1])


def test_value_and_grad_on_qjit_classical_vector():
    """Check that value_and_grad works when called on an qjit object that does not wrap a QNode
    and takes in a vector.
    """

    @qjit
    def f(vec):
        # Takes in a 2D vector (x,y) and computes 30x+40y
        prod = jnp.array([30, 40]) * vec
        return prod[0] + prod[1]

    x = jnp.array([1.0, 1.0])
    result = qjit(value_and_grad(f))(x)
    expected = (70.0, [30.0, 40.0])

    assert np.allclose(result[0], expected[0])
    assert np.allclose(result[1], expected[1])


def test_value_and_grad_on_qjit_classical_dict():
    """Check that value_and_grad works when called on an qjit object that does not wrap a QNode
    and takes in a dictionary.
    """

    @qjit
    def f(tree):
        # Takes in two 2D vectors (x1, x2) and (y1, y2) and computes x1y1+x2y2
        hello = tree["hello"]  # (x1, x2)
        world = tree["world"]  # (y1, y2)
        return (hello * world).sum()

    x = {"hello": jnp.array([1.0, 2.0]), "world": jnp.array([3.0, 4.0])}
    result = qjit(value_and_grad(f))(x)
    expected = (11.0, {"hello": jnp.array([3.0, 4.0]), "world": jnp.array([1.0, 2.0])})

    assert np.allclose(result[0], expected[0])
    assert np.allclose(result[1]["hello"], expected[1]["hello"])
    assert np.allclose(result[1]["world"], expected[1]["world"])


def test_value_and_grad_on_qjit_quantum():
    """Check that value_and_grad works when called on an qjit object that does wrap a QNode."""

    @qjit
    def workflow(x: float):
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circuit():
            qml.CNOT(wires=[0, 1])
            qml.RX(0, wires=[2])
            return qml.probs()  # This is [1, 0, 0, ...]

        return x * (circuit()[0])

    result = qjit(value_and_grad(workflow))(3.0)
    expected = (3.0, 1.0)
    assert np.allclose(result, expected)


def test_value_and_grad_on_qjit_quantum_variant():
    """
    Check that value_and_grad works when called on a QNode with trainable parameters.
    """

    def workflow_variant(x: float):
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(xx):
            qml.PauliX(wires=0)
            qml.RX(xx, wires=0)
            return qml.probs()

        return circuit(x)[0]

    result = qjit(value_and_grad(workflow_variant))(1.1)
    expected = (workflow_variant(1.1), qjit(grad(workflow_variant))(1.1))
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "argnum", [(0, 1, 2), (0), (1), (2), (0, 1), (0, 2), (1, 2), (1, 0, 2), (2, 0, 1)]
)
def test_value_and_grad_on_qjit_quantum_variant_argnum(argnum):
    """
    Check that value_and_grad works when called on a QNode with multiple trainable parameters.
    """

    def workflow_variant(x: float, y: float, z: float):
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(xx, yy, zz):
            qml.PauliX(wires=0)
            qml.RX(xx, wires=0)
            qml.RY(yy, wires=0)
            qml.RZ(zz, wires=0)
            return qml.probs()

        return circuit(x, y, z)[0]

    result = qjit(value_and_grad(workflow_variant, argnums=argnum))(1.1, 2.2, 3.3)
    expected = (
        workflow_variant(1.1, 2.2, 3.3),
        qjit(grad(workflow_variant, argnums=argnum))(1.1, 2.2, 3.3),
    )
    assert np.allclose(result[0], expected[0])
    assert np.allclose(result[1], expected[1])


def test_value_and_grad_on_qjit_quantum_variant_tree():
    """
    Check that value_and_grad works when called on an qjit object that does wrap a QNode
    with trainable parameters and a general pytree input.
    """

    def workflow_variant_tree(params):
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(params):
            qml.RX(params["x"], wires=0)
            qml.RY(params["y"], wires=0)
            return qml.probs()

        return circuit(params)[0]

    params = {"x": 0.12, "y": 0.34}
    result = qjit(value_and_grad(qjit(workflow_variant_tree)))(params)
    expected = (workflow_variant_tree(params), qjit(grad(workflow_variant_tree))(params))
    assert np.allclose(result[0], expected[0])
    assert np.allclose(result[1]["x"], expected[1]["x"])
    assert np.allclose(result[1]["y"], expected[1]["y"])


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
        diff = grad(f, argnums=0, method="fd")

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
        diff = grad(f, argnums=0, method="auto")

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
        h = grad(g, method="auto", argnums=0)
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
        h = grad(g, method="auto", argnums=0)
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
        h = grad(g, method="auto", argnums=0)
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
        h = grad(g, method="auto", argnums=0)
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


@pytest.mark.xfail(reason="Issue #1571 https://github.com/PennyLaneAI/catalyst/issues/1571")
@pytest.mark.parametrize("gate_n_inputs", [(qml.CRX, [1]), (qml.CRot, [1, 2, 3])])
def test_ps_four_term_rule(backend, gate_n_inputs):
    """Operations with the 4-term shift rule need to be decomposed to be differentiated."""
    gate, inputs = gate_n_inputs

    @qml.qnode(qml.device(backend, wires=2), diff_method="parameter-shift")
    def f(x):
        qml.RY(0.321, wires=0)
        gate(*(x * i for i in inputs), wires=[0, 1])
        return qml.expval(0.5 * qml.Z(1) @ qml.X(0) - 0.4 * qml.Y(1) @ qml.H(0))

    @qjit
    def main(x: float):
        return qml.grad(f)(x)

    result = main(0.1)
    reference = main.original_function(qml.numpy.array(0.1))

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
        g = qml.QNode(f, device, diff_method="finite-diff", gradient_kwargs={"h": 0.1})
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
        h = grad(g, method="fd", argnums=1)
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
        h = grad(g, method="fd", argnums=[1])
        return h(x, 2.0)

    def interpretted_grad_argnum_list(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f2, device, diff_method="finite-diff")
        h = qml.grad(g, argnum=[1])
        # Slightly different behaviour. If argnums is a list
        # it doesn't matter if it is a single number,
        # the return value will be a n-tuple of size of the
        # argnums list.
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
        h = grad(g, method="fd", argnums=[0, 1])
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
        h = grad(g, method="auto", argnums=[0, 1])
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
        h = grad(g, method="auto", argnums=[0, 1])
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
        h = grad(g, method="auto", argnums=[0, 1])
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

    def workflow(x: float):
        h = grad("string!", method="fd")
        return h(x)

    with pytest.raises(TypeError, match="Differentiation target must be callable"):
        qjit(workflow)


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
        d_f = grad(f, argnums=0, method=diff_method)

        def fn_f(_i, _g):
            return d_f(params)

        d_g = grad(g, argnums=0, method=diff_method)

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
        g1 = grad(f, argnums=0, method="auto")(x, y)
        g2 = grad(f, argnums=1, method="auto")(x, y)
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
    result = qjit(jacobian(circuit, argnums=[0, 1]))(psi, phi)

    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], dict)
    assert isinstance(result[0]["expval0"], tuple)
    assert len(result[0]["expval0"]) == 2
    assert isinstance(result[1], tuple)
    assert len(result[1]) == 2


def test_calssical_kwargs():
    """Test the gradient on a classical function with keyword arguments"""

    @qjit
    def f1(x, y, z):
        return x * (y - z)

    result = qjit(grad(f1, argnums=0))(3.0, y=1.0, z=2.0)
    expected = qjit(grad(f1, argnums=0))(3.0, 1.0, 2.0)
    assert np.allclose(expected, result)


def test_calssical_kwargs_switched_arg_order():
    """Test the gradient on classical function with keyword arguments and switched argument order"""

    @qjit
    def f1(x, y, z):
        return x * (y - z)

    result = qjit(grad(f1, argnums=0))(3.0, z=2.0, y=1.0)
    expected = qjit(grad(f1, argnums=0))(3.0, 1.0, 2.0)
    assert np.allclose(expected, result)


def test_qnode_kwargs(backend):
    """Test the gradient on a qnode with keyword arguments"""
    num_wires = 1
    dev = qml.device(backend, wires=num_wires)

    @qml.qnode(dev)
    def circuit(x, y, z):
        qml.RY(x, wires=0)
        qml.RX(y, wires=0)
        qml.RX(z, wires=0)
        return qml.expval(qml.PauliZ(0))

    result = qjit(jacobian(circuit, argnums=[0]))(0.1, y=0.2, z=0.3)
    expected = qjit(jacobian(circuit, argnums=[0]))(0.1, 0.2, 0.3)
    assert np.allclose(expected, result)
    result = qjit(grad(circuit, argnums=[0]))(0.1, y=0.2, z=0.3)
    expected = qjit(grad(circuit, argnums=[0]))(0.1, 0.2, 0.3)
    assert np.allclose(expected, result)
    result_val, result_grad = qjit(value_and_grad(circuit, argnums=[0]))(0.1, y=0.2, z=0.3)
    expected_val = qjit(circuit)(0.1, 0.2, 0.3)
    expected_grad = qjit(grad(circuit, argnums=[0]))(0.1, 0.2, 0.3)
    print(result_val, result_grad)
    print(expected_val, expected_grad)
    assert np.allclose(expected_val, result_val)
    assert np.allclose(expected_grad, result_grad)


def test_qnode_kwargs_switched_arg_order(backend):
    """Test the gradient on a qnode with keyword arguments and switched argument order"""
    num_wires = 1
    dev = qml.device(backend, wires=num_wires)

    @qml.qnode(dev)
    def circuit(x, y, z):
        qml.RY(x, wires=0)
        qml.RX(y, wires=0)
        qml.RX(z, wires=0)
        return qml.expval(qml.PauliZ(0))

    switched_order = qjit(jacobian(circuit, argnums=[0]))(0.1, z=0.3, y=0.2)
    expected = qjit(jacobian(circuit, argnums=[0]))(0.1, 0.2, 0.3)
    assert np.allclose(expected[0], switched_order[0])
    switched_order = qjit(grad(circuit, argnums=[0]))(0.1, z=0.3, y=0.2)
    expected = qjit(grad(circuit, argnums=[0]))(0.1, 0.2, 0.3)
    assert np.allclose(expected[0], switched_order[0])
    switched_order_val, switched_order_grad = qjit(value_and_grad(circuit, argnums=[0]))(
        0.1, z=0.3, y=0.2
    )
    expected_val = qjit(circuit)(0.1, 0.2, 0.3)
    expected_grad = qjit(grad(circuit, argnums=[0]))(0.1, 0.2, 0.3)
    assert np.allclose(expected_val, switched_order_val)
    assert np.allclose(expected_grad, switched_order_grad)


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
    result = qjit(jacobian(circuit, argnums=[0, 1]))(psi, phi)

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
    catalyst_results = qjit(jacobian(f, argnums=[0, 1]))(x, y)

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
    catalyst_results = qjit(jacobian(f, argnums=[0, 1]))(x, y)

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
    catalyst_results = qjit(jacobian(f, argnums=[0, 1]))(x, y)

    flatten_res_jax, tree_jax = tree_flatten(jax_expected_results)
    flatten_res_catalyst, tree_catalyst = tree_flatten(catalyst_results)

    assert tree_jax == tree_catalyst
    assert np.allclose(flatten_res_jax, flatten_res_catalyst)


def test_non_parametrized_circuit(backend):
    """Test that the derivate of non parametrized circuit is null."""
    dev = qml.device(backend, wires=1)

    def cost(x):
        @qml.qnode(dev)
        def circuit(x):  # pylint: disable=unused-argument
            qml.PauliX(wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        return circuit(x)

    assert np.allclose(qjit(grad(cost))(1.1), 0.0)


@pytest.mark.xfail(reason="The verifier currently doesn't distinguish between active/inactive ops")
@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_adj_qubitunitary(inp, backend):
    """Test the adjoint method."""

    def f(x):
        qml.RX(x, wires=0)
        U1 = np.array([[0.5 + 0.5j, -0.5 - 0.5j], [0.5 - 0.5j, 0.5 - 0.5j]], dtype=complex)
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


@pytest.mark.xfail(reason="Need PR 332.")
@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_preprocessing_outside_qnode(inp, backend):
    """Test the preprocessing outside qnode."""

    @qml.qnode(qml.device(backend, wires=1))
    def f(y):
        qml.RX(y, wires=0)
        return qml.expval(qml.PauliZ(0))

    @qjit
    def g(x):
        return grad(lambda y: f(jnp.cos(y)) ** 2)(x)

    def h(x):
        return jax.grad(lambda y: f(jnp.cos(y)) ** 2)(x)

    assert np.allclose(g(inp), h(inp))


@pytest.mark.xfail(reason="Need PR 332.")
def test_gradient_slice(backend):
    """Test the differentation when the qnode generates memref with non identity layout."""
    n_wires = 5
    data = jnp.sin(jnp.mgrid[-2:2:0.2].reshape(n_wires, -1)) ** 3

    weights = jnp.ones([n_wires, 3])

    bias = jnp.array(0.0)
    params = {"weights": weights, "bias": bias}

    dev = qml.device(backend, wires=n_wires)

    @qml.qnode(dev)
    def circuit(data, weights):
        """Quantum circuit ansatz"""

        for i in range(n_wires):
            qml.RY(data[i], wires=i)

        for i in range(n_wires):
            qml.RX(weights[i, 0], wires=i)
            qml.RY(weights[i, 1], wires=i)
            qml.RX(weights[i, 2], wires=i)
            qml.CNOT(wires=[i, (i + 1) % n_wires])

        return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(n_wires)]))

    def my_model(data, weights, bias):
        return circuit(data[:, 0], weights) + bias

    cat_res = qjit(
        jacobian(
            my_model,
            argnums=1,
        )
    )(data, params["weights"], params["bias"])
    jax_res = jax.jacobian(my_model, argnums=1)(data, params["weights"], params["bias"])
    assert np.allclose(cat_res, jax_res)


def test_ellipsis_differentiation(backend):
    """Test circuit diff with ellipsis in the preprocessing."""
    dev = qml.device(backend, wires=3)

    @qml.qnode(dev)
    def circuit(weights):
        r = weights[..., 1, 2, 0]
        qml.RY(r, wires=0)
        return qml.expval(qml.PauliZ(0))

    weights = jnp.ones([5, 3, 3])

    cat_res = qjit(grad(circuit, argnums=0))(weights)
    jax_res = jax.grad(circuit, argnums=0)(weights)
    assert np.allclose(cat_res, jax_res)


@pytest.mark.xfail(reason="First need #332, then Vmap yields wrong results when differentiated")
def test_vmap_worflow_derivation(backend):
    """Check the gradient of a vmap workflow"""
    n_wires = 5
    data = jnp.sin(jnp.mgrid[-2:2:0.2].reshape(n_wires, -1)) ** 3

    targets = jnp.array([-0.2, 0.4, 0.35, 0.2], dtype=jax.numpy.float64)

    dev = qml.device(backend, wires=n_wires)

    @qml.qnode(dev, diff_method="adjoint")
    def circuit(data, weights):
        """Quantum circuit ansatz"""

        @for_loop(0, n_wires, 1)
        def data_embedding(i):
            qml.RY(data[i], wires=i)

        data_embedding()

        @for_loop(0, n_wires, 1)
        def ansatz(i):
            qml.RX(weights[i, 0], wires=i)
            qml.RY(weights[i, 1], wires=i)
            qml.RX(weights[i, 2], wires=i)
            qml.CNOT(wires=[i, (i + 1) % n_wires])

        ansatz()

        return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(n_wires)]))

    circuit = vmap(circuit, in_axes=(1, None))

    def my_model(data, weights, bias):
        return circuit(data, weights) + bias

    def loss_fn(params, data, targets):
        predictions = my_model(data, params["weights"], params["bias"])
        loss = jnp.sum((targets - predictions) ** 2 / len(data))
        return loss

    weights = jnp.ones([n_wires, 3])
    bias = jnp.array(0.0, dtype=jax.numpy.float64)
    params = {"weights": weights, "bias": bias}

    results_cat = qjit(grad(loss_fn))(params, data, targets)
    results_jax = jax.grad(loss_fn)(params, data, targets)

    data_cat, pytree_enzyme = tree_flatten(results_cat)
    data_jax, pytree_fd = tree_flatten(results_jax)

    assert pytree_enzyme == pytree_fd
    assert jnp.allclose(data_cat[0], data_jax[0])
    assert jnp.allclose(data_cat[1], data_jax[1])


@pytest.mark.xfail(reason="First need #332, then Vmap yields wrong results when differentiated")
def test_forloop_vmap_worflow_derivation(backend):
    """Test a forloop vmap."""
    n_wires = 5
    data = jnp.sin(jnp.mgrid[-2:2:0.2].reshape(n_wires, -1)) ** 3
    weights = jnp.ones([n_wires, 3])

    bias = jnp.array(0.0)
    params = {"weights": weights, "bias": bias}

    dev = qml.device(backend, wires=n_wires)

    @qml.qnode(dev)
    def circuit(data, weights):
        """Quantum circuit ansatz"""

        for i in range(n_wires):
            qml.RY(data[i], wires=i)

        for i in range(n_wires):
            qml.RX(weights[i, 0], wires=i)
            qml.RY(weights[i, 1], wires=i)
            qml.RX(weights[i, 2], wires=i)
            qml.CNOT(wires=[i, (i + 1) % n_wires])

        return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(n_wires)]))

    def my_model(data, weights):
        transposed_data = jnp.transpose(data, [1, 0])
        result_0 = circuit(transposed_data[0], weights)

        transposed_result = jnp.empty((data.shape[1], *result_0.shape), result_0.dtype)
        transposed_result = transposed_result.at[0].set(result_0)

        @for_loop(1, data.shape[1], 1)
        def body(i, result_array):
            result_i = circuit(transposed_data[i], weights)
            return result_array.at[i].set(result_i)

        return body(transposed_result)

    cat_res = qjit(
        jacobian(
            my_model,
            argnums=1,
        )
    )(data, params["weights"])
    jax_res = jax.jacobian(my_model, argnums=1)(data, params["weights"])

    data_cat, pytree_enzyme = tree_flatten(jax_res)
    data_jax, pytree_fd = tree_flatten(cat_res)

    assert pytree_enzyme == pytree_fd

    assert jnp.allclose(data_cat[0], data_jax[0])
    assert jnp.allclose(data_cat[1], data_jax[1])


@pytest.mark.parametrize(
    "gate,state", ((qml.BasisState, np.array([1])), (qml.StatePrep, np.array([0, 1])))
)
def test_paramshift_with_gates(gate, state):
    """Test parameter shift works with a variety of gates present in the circuit."""

    dev = qml.device("lightning.qubit", wires=1)

    @grad
    @qml.qnode(dev, diff_method="parameter-shift")
    def cost(x):
        gate(state, wires=0)
        qml.RY(x, wires=0)
        return qml.expval(qml.PauliZ(0))

    param = 0.1
    expected = cost(param)
    observed = qjit(cost)(param)
    assert np.allclose(expected, observed)


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

        with pytest.raises(DifferentiableCompileError, match="MidCircuitMeasure is not allowed"):

            @qjit
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

            @qjit
            def cir(x: float):
                return grad(f)(x)

    def test_with_zne(self):
        """Test with ZNE"""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliX(0))

        def g(x):
            return mitigate_with_zne(f, scale_factors=[1, 3, 5])(x)

        with pytest.raises(CompileError, match=".*Compilation failed.*"):

            @qjit
            def cir(x: float):
                return grad(g)(x)


class TestGradientUsagePatterns:
    """Test usage patterns of Gradient functions"""

    def test_grad_usage_patterns(self):
        """Test usage patterns of catalyst.grad."""

        def fn(x):
            return x**2

        x = 4.0

        res_pattern_fn_as_argument = grad(fn, method="fd")(x)
        res_pattern_partial = grad(method="fd")(fn)(x)
        expected = jax.grad(fn)(x)

        assert np.allclose(res_pattern_fn_as_argument, expected)
        assert np.allclose(res_pattern_partial, expected)

    def test_value_and_grad_usage_patterns(self):
        """Test usage patterns of catalyst.value_and_grad."""

        def fn(x):
            return x**2

        x = 4.0

        fn_as_argument_val, fn_as_argument_grad = value_and_grad(fn, method="fd")(x)
        partial_val, partial_grad = value_and_grad(method="fd")(fn)(x)
        expected_val, expected_grad = jax.value_and_grad(fn)(x)

        assert np.allclose(fn_as_argument_val, expected_val)
        assert np.allclose(partial_val, expected_val)
        assert np.allclose(fn_as_argument_grad, expected_grad)
        assert np.allclose(partial_grad, expected_grad)

    def test_jacobian_usage_patterns(self):
        """Test usage patterns of catalyst.jacobian."""

        def fn(x):
            return x**2

        x = 4.0

        res_pattern_fn_as_argument = jacobian(fn, method="fd")(x)
        res_pattern_partial = jacobian(method="fd")(fn)(x)
        expected = jax.jacobian(fn)(x)

        assert np.allclose(res_pattern_fn_as_argument, expected)
        assert np.allclose(res_pattern_partial, expected)


@pytest.mark.parametrize("argnums", [0, 1, (0, 1)])
def test_grad_argnums(argnums):
    """Tests https://github.com/PennyLaneAI/catalyst/issues/1477"""

    @qjit
    @qml.qnode(device=qml.device("lightning.qubit", wires=4), interface="jax")
    def circuit(inputs, weights):
        qml.AngleEmbedding(features=inputs, wires=range(4), rotation="X")
        for i in range(1, 4):
            qml.CRX(weights[i - 1], wires=[i, 0])
        return qml.expval(qml.PauliZ(wires=0))

    weights = jnp.array([3.0326467, 0.98860157, 1.9887222])
    inputs = jnp.array([0.9653214, 0.31468165, 0.63302994])

    def compare_structure_and_value(o1, o2):
        return tree_structure(o1) == tree_structure(o2) and tree_all(tree_map(jnp.allclose, o1, o2))

    result = grad(circuit, argnums=argnums)(weights, inputs)
    expected = jax.grad(circuit.original_function, argnums=argnums)(weights, inputs)
    assert compare_structure_and_value(result, expected)

    _, result = value_and_grad(circuit, argnums=argnums)(weights, inputs)
    _, expected = jax.value_and_grad(circuit.original_function, argnums=argnums)(weights, inputs)
    assert compare_structure_and_value(result, expected)


class TestGradientMethodErrors:
    """Test errors for different gradient methods."""

    @staticmethod
    def get_custom_device(grad_method="fd", **kwargs):
        """Generate a custom device with specified gradient method."""
        lightning_device = qml.device("lightning.qubit", wires=0)

        class CustomDevice(qml.devices.Device):
            """Custom Gate Set Device"""

            def __init__(self, shots=None, wires=None):
                super().__init__(wires=wires, shots=shots)
                self.qjit_capabilities = lightning_device.capabilities

            def preprocess(self, execution_config=None):
                """Device preprocessing function."""
                program, config = lightning_device.preprocess(execution_config)
                config.gradient_method = grad_method
                return program, config

            @staticmethod
            def get_c_interface():
                """Returns a tuple consisting of the device name, and
                the location to the shared object with the C/C++ device implementation.
                """
                system_extension = ".dylib" if platform.system() == "Darwin" else ".so"
                lib_path = (
                    get_lib_path("runtime", "RUNTIME_LIB_DIR")
                    + "/librtd_null_qubit"
                    + system_extension
                )
                return "NullQubit", lib_path

            def execute(self, _circuits, _execution_config):
                """Raises: RuntimeError"""
                raise RuntimeError("QJIT devices cannot execute tapes.")

            def supports_derivatives(self, config, circuit=None):  # pylint: disable=unused-argument
                """Pretend we support any derivatives"""
                return True

        return CustomDevice(**kwargs)

    def test_device_grad_method_error(self):
        """Test that using 'device' grad method raises appropriate error."""

        @qml.qnode(self.get_custom_device(grad_method="device", wires=1))
        def f(x: float):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        with pytest.raises(
            ValueError, match="The device does not provide a catalyst compatible gradient method"
        ):
            qjit(grad(f))

    def test_finite_diff_grad_method_error(self):
        """Test that using 'finite-diff' grad method raises appropriate error."""

        @qml.qnode(self.get_custom_device(grad_method="finite-diff", wires=1))
        def f(x: float):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        with pytest.raises(
            ValueError, match="Finite differences at the QNode level is not supported"
        ):
            qjit(grad(f))

    def test_invalid_grad_method_error(self):
        """Test that using an invalid grad method raises appropriate error."""

        @qml.qnode(self.get_custom_device(grad_method="invalid_method", wires=1))
        def f(x: float):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        with pytest.raises(ValueError, match="Invalid gradient method: invalid_method"):
            qjit(grad(f))


class TestParameterShiftVerificationUnitTests:
    """Unit tests for parameter shift verification"""

    def test_check_grad_recipe_no_grad_recipe(self):
        """Check that if grad recipe is not defined, no exception gets triggered"""

        # a family of ops that do not have grad recipe are control flow ops
        class DummyOp(qml.operation.Operator): ...

        assert not _has_grad_recipe(DummyOp(wires=[0]))

    def test_check_grad_recipe_empty(self):
        """Some grad recipes are defined but are filled with Nones"""

        class DummyOp(qml.operation.Operator):
            @property
            def grad_recipe(self):
                return [None]

        assert not _has_grad_recipe(DummyOp(wires=[0]))

    def test_check_grad_recipe_different_size(self):
        """len(grad_recipe) != len(op.data)"""

        class DummyOp(qml.operation.Operator):
            def __init__(self, wires=None):
                param = 0.0
                super().__init__(param, wires=wires)

            @property
            def num_params(self):
                return 1

            @property
            def grad_recipe(self):
                return (
                    [[0.5, 1.0, np.pi / 2], [-0.5, 1.0, -np.pi / 2]],
                    [[0.5, 1.0, np.pi / 2], [-0.5, 1.0, -np.pi / 2]],
                )

        assert not _is_grad_recipe_same_as_catalyst(DummyOp(wires=[0]))

    def test_check_grad_recipe_different(self):
        """Check exception is raised when invalid grad_recipe is found"""

        class DummyOp(qml.operation.Operator):
            def __init__(self, wires=None):
                param = 0.0
                super().__init__(param, wires=wires)

            @property
            def num_params(self):
                return 1

            @property
            def grad_recipe(self):
                return ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],)

        assert not _is_grad_recipe_same_as_catalyst(DummyOp(wires=[0]))

    def test_check_grad_recipe_dynamic(self):
        """Check exception is raised when dynamic grad recipe is found"""

        class DummyOp(qml.operation.Operator):
            def __init__(self, param, wires=None):
                self.param = param
                super().__init__(param, wires=wires)

            @property
            def num_params(self):
                return 1

            @property
            def grad_recipe(self):
                return ([[0.5, self.param, jnp.pi / 2], [-0.5, 1.0, -jnp.pi / 2]],)

        def program(x):
            assert not _is_grad_recipe_same_as_catalyst(DummyOp(x, wires=[0]))

        jax.make_jaxpr(program)(0.0)

    def test_check_param_frequencies(self):
        """No param frequencies attr"""

        class DummyOp(qml.operation.Operator): ...

        assert not hasattr(DummyOp, "parameter_frequencies")
        assert not _has_parameter_frequencies(DummyOp(wires=[0]))

    def test_check_param_frequencies_different_length(self):
        """Check exception is raised when frequencies length mismatches parameter length"""

        class DummyOp(qml.operation.Operator):
            def __init__(self, wires=None):
                super().__init__(0.0, wires=wires)

            @property
            def num_params(self):
                return 1

            @property
            def parameter_frequencies(self):
                return [1.0, 1.0]

        assert not _are_param_frequencies_same_as_catalyst(DummyOp(wires=[0]))

    def test_check_invalid_frequencies(self):
        """Check exception is raised when invalid frequencies are found"""

        class DummyOp(qml.operation.Operator):
            def __init__(self, wires=None):
                super().__init__(0.0, wires=wires)

            @property
            def num_params(self):
                return 1

            @property
            def parameter_frequencies(self):
                return [(0.0,)]

        assert not _are_param_frequencies_same_as_catalyst(DummyOp(wires=[0]))

    def test_undefined_frequencies(self):
        """Test ParameterFrequenciesUndefinedError"""

        class DummyOp(qml.operation.Operator):
            def __init__(self, wires=None):
                super().__init__(0.0, wires=wires)

            @property
            def num_params(self):
                return 1

            @property
            def parameter_frequencies(self):
                raise qml.operation.ParameterFrequenciesUndefinedError()

        assert not _has_parameter_frequencies(DummyOp(wires=[0]))

    def test_qubit_unitary(self):
        """QubitUnitary is not a differentiable gate in Catalyst"""
        op = qml.QubitUnitary(jnp.array([[1, 1], [1, -1]]), wires=0)
        assert not _paramshift_op_checker(op)

    def test_no_grad_recipe_no_param_frequencies(self):
        """No grad recipe, no param shift, not hybrid op => no grad"""

        class DummyOp(qml.operation.Operator):
            def __init__(self, wires=None):
                super().__init__(0.0, wires=wires)

            @property
            def num_params(self):
                return 1

        assert not _paramshift_op_checker(DummyOp(wires=[0]))

    def test_hybrid_op(self):
        """HybridOp => grad"""

        class DummyOp(HybridOp):
            def __init__(self):
                super().__init__([], [], [])

        assert _paramshift_op_checker(DummyOp())


class TestParameterShiftVerificationIntegrationTests:
    """Test to verify operations / observables / measurements when doing parameter shift.

    Source of truth obtained from shortcut story: 84819
    """

    def test_is_mcm(self, backend):
        """No mcm"""

        device = qml.device(backend, wires=1)

        with pytest.raises(DifferentiableCompileError, match="MidCircuitMeasure is not allowed"):

            @qjit
            @grad
            @qml.qnode(device, diff_method="parameter-shift")
            def circuit(_: float):
                measure(0)
                return qml.expval(qml.PauliZ(wires=0))

    def test_all_arguments_are_constant(self, backend):
        """When all arguments are constant they do not contribute to the gradient"""
        device = qml.device(backend, wires=1)

        # Yes, this test does not have an assertion.
        # The test is that this does not produce an assertion.

        @qjit
        @grad
        @qml.qnode(device, diff_method="parameter-shift")
        def circuit(_: float):
            qml.RX(0.0, wires=[0])
            return qml.expval(qml.PauliZ(wires=0))

    def test_grad_recipe_dynamic(self, backend):
        """Raise exception when there is an op with a grad_recipe that's dynamic"""
        device = qml.device(backend, wires=1)

        class RX(qml.RX):
            @property
            def grad_recipe(self):
                x = self.data[0]
                c = 0.5 / jnp.sin(x)
                return ([[c, 0.0, 2 * x], [-c, 0.0, 0.0]],)

        with pytest.raises(CompileError):

            @qjit
            @grad
            @qml.qnode(device, diff_method="parameter-shift")
            def circuit(x: float):
                RX(x, wires=[0])
                return qml.expval(qml.PauliZ(wires=0))

    def test_grad_recipe_static(self, backend):
        """Raise exception when there is an op with a mismatching grad_recipe"""
        device = qml.device(backend, wires=1)

        class RX(qml.RX):
            @property
            def grad_recipe(self):
                return ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],)

        with pytest.raises(CompileError):

            @qjit
            @grad
            @qml.qnode(device, diff_method="parameter-shift")
            def circuit(x: float):
                RX(x, wires=[0])
                return qml.expval(qml.PauliZ(wires=0))

    def test_parameter_frequencies(self, backend):
        """Raise exception when when there is an lengths are mismatched."""
        device = qml.device(backend, wires=1)

        class RX(qml.RX):
            @property
            def parameter_frequencies(self):
                # Only one parameter but two frequencies is an error
                return (1.0, 1.0)

        with pytest.raises(CompileError):

            @qjit
            @grad
            @qml.qnode(device, diff_method="parameter-shift")
            def circuit(x: float):
                RX(x, wires=[0])
                return qml.expval(qml.PauliZ(wires=0))

    def test_parameter_frequencies_not_one(self, backend):
        """When there is an op without parameter_frequencies, ps gradient should fail"""
        device = qml.device(backend, wires=1)

        class RX(qml.RX):
            @property
            def parameter_frequencies(self):
                # Only one parameter but two frequencies is an error
                return [(2.0,)]

        with pytest.raises(CompileError):

            @qjit
            @grad
            @qml.qnode(device, diff_method="parameter-shift")
            def circuit(x: float):
                RX(x, wires=[0])
                return qml.expval(qml.PauliZ(wires=0))


def test_closure_variable_grad():
    """Test that grad can take closure variables"""

    @qml.qjit
    def workflow_closure(x, y):

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(jnp.pi * x, wires=0)
            qml.RX(jnp.pi * y, wires=0)
            return qml.expval(qml.PauliY(0))

        g = grad(circuit)
        return g(x)

    @qml.qjit
    def workflow_no_closure(x, y):

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(jnp.pi * x, wires=0)
            qml.RX(jnp.pi * y, wires=0)
            return qml.expval(qml.PauliY(0))

        g = grad(circuit)
        return g(x, y)

    expected = workflow_no_closure(1.0, 0.25)
    observed = workflow_closure(1.0, 0.25)
    assert np.allclose(expected, observed)


def test_closure_variable_value_and_grad():
    """Test that value and grad can take closure variables"""

    @qml.qjit
    def workflow_closure(x, y):

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(jnp.pi * x, wires=0)
            qml.RX(jnp.pi * y, wires=0)
            return qml.expval(qml.PauliY(0))

        g = value_and_grad(circuit)
        return g(x)

    @qml.qjit
    def workflow_no_closure(x, y):

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(jnp.pi * x, wires=0)
            qml.RX(jnp.pi * y, wires=0)
            return qml.expval(qml.PauliY(0))

        g = value_and_grad(circuit)
        return g(x, y)

    x, y = 1.0, 0.25
    expected = workflow_no_closure(x, y)
    observed = workflow_closure(x, y)
    assert np.allclose(expected, observed)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
