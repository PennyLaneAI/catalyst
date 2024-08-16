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

"""Test built-in differentiation support of hybrid programs with classical postprocessing."""

import jax
import jax.numpy as jnp
import pennylane as qml
import pytest

from catalyst import DifferentiableCompileError, grad, jacobian, qjit

SUPPORTED_DIFF_METHODS = ["parameter-shift", "adjoint"]


@pytest.mark.parametrize("diff_method", SUPPORTED_DIFF_METHODS)
def test_scalar_scalar(backend, diff_method):
    """Test a hybrid scalar -> scalar (internally a point tensor -> point tensor) workflow"""

    @qml.qnode(qml.device(backend, wires=1), diff_method=diff_method)
    def workflow(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(wires=0))

    def postprocess(x):
        w = workflow(x)
        return jnp.cos(w)

    @qjit
    def jac_postprocess(x):
        return grad(postprocess, method="auto")(x)

    jax_jacobian = jax.jacobian(postprocess)(0.5)
    catalyst_jacobian = jac_postprocess(0.5)
    assert catalyst_jacobian == pytest.approx(jax_jacobian)


@pytest.mark.parametrize("diff_method", SUPPORTED_DIFF_METHODS)
def test_one_to_many(backend, diff_method):
    """Test a tall Jacobian (one input to many outputs)"""

    @qml.qnode(qml.device(backend, wires=1), diff_method=diff_method)
    def workflow(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(wires=0))

    def postprocess(x):
        w = workflow(x)
        return jnp.array([jnp.cos(w), w, w * 2])

    @qjit
    def jac_postprocess(x):
        return jacobian(postprocess, method="auto")(x)

    jax_jacobian = jax.jacobian(postprocess)(0.5)
    catalyst_jacobian = jac_postprocess(0.5)
    assert catalyst_jacobian == pytest.approx(jax_jacobian)


@pytest.mark.parametrize("diff_method", SUPPORTED_DIFF_METHODS)
def test_many_to_one(backend, diff_method):
    """Test a wide Jacobian (many inputs to one output)"""

    @qml.qnode(qml.device(backend, wires=1), diff_method=diff_method)
    def workflow(x):
        qml.RX(x[0] * 3, wires=0)
        qml.RX(x[1] * 2, wires=0)
        qml.RX(x[2] * 1.5, wires=0)
        qml.RZ(jnp.exp(x[3]), wires=0)
        return qml.expval(qml.PauliZ(wires=0))

    def postprocess(x):
        w = workflow(x)
        return jnp.cos(w)

    @qjit
    def jac_postprocess(x):
        return grad(postprocess, method="auto")(x)

    x = jnp.array([0.5, 0.4, 0.3, 0.2])
    jax_jacobian = jax.jacobian(postprocess)(x)
    catalyst_jacobian = jac_postprocess(x)
    assert catalyst_jacobian == pytest.approx(jax_jacobian)


def test_tensor_measure(backend):
    """Tests correctness of a derivative of a qnode that returns a tensor"""

    @qml.qnode(qml.device(backend, wires=2), diff_method="parameter-shift")
    def workflow(x):
        qml.RX(x, wires=0)
        qml.RX(x * 2, wires=1)
        return qml.probs()

    def postprocess(x):
        probs = workflow(x)
        return jnp.sum(probs) / jnp.prod(probs)

    @qjit
    def jac_postprocess(x):
        return jacobian(postprocess, method="auto")(x)

    jax_jacobian = jax.jacobian(postprocess)(0.5)
    catalyst_jacobian = jac_postprocess(0.5)
    assert catalyst_jacobian == pytest.approx(jax_jacobian)


def test_multi_measure(backend):
    """Tests correctness of a derivative of a qnode with multiple measurements"""

    @qml.qnode(qml.device(backend, wires=1), diff_method="parameter-shift")
    def workflow(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(wires=0)), qml.probs()

    def postprocess(x):
        w, probs = workflow(x)
        return jnp.cos(w) + jnp.prod(probs)

    @qjit
    def jac_postprocess(x):
        return grad(postprocess, method="auto")(x)

    jax_jacobian = jax.jacobian(postprocess)(0.5)
    catalyst_jacobian = jac_postprocess(0.5)
    assert catalyst_jacobian == pytest.approx(jax_jacobian)


def test_purely_classical():
    """Test the behaviour of the grad op on a purely classical function"""

    def postprocess(x):
        return x**2

    @qjit
    def classical_grad(x):
        return grad(postprocess, method="auto")(x)

    assert classical_grad(4.5) == 9


@pytest.mark.parametrize("diff_method", SUPPORTED_DIFF_METHODS)
def test_jacobian(backend, diff_method):
    """Tests correctness of a full Jacobian with multiple inputs and outputs"""

    @qml.qnode(qml.device(backend, wires=1), diff_method=diff_method)
    def workflow(x):
        qml.RX(x[0] * 3, wires=0)
        qml.RX(x[1] * 2, wires=0)
        qml.RX(x[2] * 1.5, wires=0)
        qml.RZ(jnp.exp(x[3]), wires=0)
        return qml.expval(qml.PauliZ(wires=0))

    def postprocess(x):
        w = workflow(x)
        return jnp.array([[jnp.sin(w), jnp.cos(w)], [w, w * 2], [w / 2, w]])

    @qjit
    def jac_postprocess(x):
        return jacobian(postprocess, method="auto")(x)

    x = jnp.array([0.5, 0.4, 0.3, 0.2])
    assert jac_postprocess(x) == pytest.approx(jax.jacobian(postprocess)(x))


@pytest.mark.parametrize("diff_method", SUPPORTED_DIFF_METHODS)
def test_multi_result(backend, diff_method):
    """Tests the correctness of multiple Jacobians from multiple results"""

    @qml.qnode(qml.device(backend, wires=1), diff_method=diff_method)
    def workflow(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(wires=0))

    def postprocess(x):
        w = workflow(x)
        return jnp.cos(w), jnp.array([w, x * 2.454])

    @qjit
    def jac_postprocess(x):
        return jacobian(postprocess, method="auto")(x)

    jax_jacobian = jax.jacobian(postprocess)(0.5)
    catalyst_jacobian = jac_postprocess(0.5)
    for catalyst_result, jax_result in zip(catalyst_jacobian, jax_jacobian):
        assert catalyst_result == pytest.approx(jax_result)


@pytest.mark.parametrize("diff_method", SUPPORTED_DIFF_METHODS)
def test_multi_arg_multi_result(backend, diff_method):
    """Tests multiple tensor arguments and results"""

    @qml.qnode(qml.device(backend, wires=1), diff_method=diff_method)
    def workflow(x, y):
        qml.RX(x[0], wires=0)
        qml.RY(y, wires=0)
        return qml.expval(qml.PauliZ(wires=0))

    def postprocess(x, y):
        w = workflow(x, y)
        return jnp.cos(w), jnp.array([w, x[0] * 2.454])

    @qjit
    def jac_postprocess(x, y):
        return jacobian(postprocess, argnums=[0, 1], method="auto")(x, y)

    args = (jnp.array([0.5, 0, 0]), 0.4)
    jax_jacobian = jax.jacobian(postprocess, argnums=[0, 1])(*args)
    catalyst_jacobian = jac_postprocess(*args)

    for i in range(2):
        for j in range(2):
            assert jax_jacobian[i][j] == pytest.approx(catalyst_jacobian[i][j])


def test_multi_qnode(backend):
    """Test a multi-QNode workflow where each QNode has a different diff_method"""
    device = qml.device(backend, wires=2)

    @qml.qnode(device, diff_method="adjoint")
    def first_qnode(a):
        qml.RX(a[0], wires=0)
        qml.CNOT(wires=(0, 1))
        qml.RY(a[1], wires=1)
        qml.RZ(a[2], wires=1)
        return qml.expval(qml.PauliX(1))

    @qml.qnode(device, diff_method="parameter-shift")
    def second_qnode(x):
        qml.RX(x[2] * 0.4, wires=0)
        return qml.expval(qml.PauliZ(0))

    def postprocess(x):
        return jnp.tanh(second_qnode(x)) * jnp.cos(first_qnode(x))

    @qjit
    def grad_workflow(x):
        return grad(postprocess, method="auto")(x)

    x = jnp.array([0.1, 0.2, 0.3])
    assert grad_workflow(x) == pytest.approx(jax.jacobian(postprocess)(x))


def test_qnode_different_returns(backend):
    """Test a multi-QNode workflow where the QNodes have different diff_methods and return
    different shapes.
    """

    @qml.qnode(qml.device(backend, wires=1), diff_method="parameter-shift")
    def circuit_A(params):
        qml.RX(jnp.exp(params[0] ** 2) / jnp.cos(params[1] / 4), wires=0)
        return qml.probs()

    @qml.qnode(qml.device(backend, wires=1), diff_method="adjoint")
    def circuit_B(params):
        qml.RX(jnp.exp(params[1] ** 2) / jnp.cos(params[0] / 4), wires=0)
        return qml.expval(qml.PauliZ(wires=0))

    def loss(params):
        return jnp.prod(circuit_A(params)) + circuit_B(params)

    @qjit
    def grad_loss(theta):
        return grad(loss, method="auto")(theta)

    x = jnp.array([1.0, 2.0])
    assert grad_loss(x) == pytest.approx(jax.jacobian(loss)(x))


def test_no_nested_grad_without_fd():
    """Test input validation for higher order derivatives where outer grad ops don't have
    method='fd'.
    """

    def inner(x: float):
        return x

    def middle(x: float):
        return grad(inner, method="fd")(x)

    with pytest.raises(DifferentiableCompileError, match="higher order derivatives"):

        @qjit
        def outer(x: float):
            return grad(middle, method="auto")(x)

        outer(9.0)


def test_nested_qnode(backend):
    """Test that Enzyme is able to compile a hybrid program containing nested QNodes"""
    device = qml.device(backend, wires=2)

    @qml.qnode(device, diff_method="adjoint")
    def inner(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0))

    @qml.qnode(device, diff_method="parameter-shift")
    def outer(x):
        qml.RX(inner(x), wires=0)
        return qml.expval(qml.PauliY(0))

    def post(x):
        return outer(x)

    @qjit(target="mlir")
    def _grad_qnode_direct(x: float):
        return grad(outer, method="auto")(x)

    @qjit(target="mlir")
    def _grad_postprocess(x: float):
        return grad(post, method="auto")(x)

    # The runtime doesn't support actually executing nested QNodes, so we just make sure they
    # compile without issues.
    # assert _grad_qnode_direct(1.0) == pytest.approx(jax.grad(post)(1.0))


if __name__ == "__main__":
    pytest.main(["-x", __file__])
