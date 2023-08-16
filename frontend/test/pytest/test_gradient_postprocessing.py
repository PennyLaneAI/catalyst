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

"""Test built-in differentiation support of hybrid programs with classical postprocessing."""

import jax
import jax.numpy as jnp
import pennylane as qml
import pytest

from catalyst import qjit, grad


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
    def jacobian(x):
        return grad(postprocess, method="defer")(x)

    jax_jacobian = jax.jacobian(postprocess)(0.5)
    catalyst_jacobian = jacobian(0.5)
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
    def jacobian(x):
        return grad(postprocess, method="defer")(x)

    jax_jacobian = jax.jacobian(postprocess)(0.5)
    catalyst_jacobian = jacobian(0.5)
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
        return jnp.array([jnp.cos(w), w, w * 2])

    @qjit
    def jacobian(x):
        return grad(postprocess, method="defer")(x)

    x = jnp.array([0.5, 0.4, 0.3, 0.2])
    jax_jacobian = jax.jacobian(postprocess)(x)
    catalyst_jacobian = jacobian(x)
    assert catalyst_jacobian == pytest.approx(jax_jacobian.mT)


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
    def jacobian(x):
        return grad(postprocess, method="defer")(x)

    jax_jacobian = jax.jacobian(postprocess)(0.5)
    catalyst_jacobian = jacobian(0.5)
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
    def jacobian(x):
        return grad(postprocess, method="defer")(x)

    jax_jacobian = jax.jacobian(postprocess)(0.5)
    catalyst_jacobian = jacobian(0.5)
    assert catalyst_jacobian == pytest.approx(jax_jacobian)


def test_purely_classical():
    """Test the behaviour of the grad op on a purely classical function"""

    def postprocess(x):
        return x**2

    @qjit
    def classical_grad(x):
        return grad(postprocess, method="defer")(x)

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
    def jacobian(x):
        return grad(postprocess, method="defer")(x)

    x = jnp.array([0.5, 0.4, 0.3, 0.2])
    assert jacobian(x) == pytest.approx(jnp.transpose(jax.jacobian(postprocess)(x), axes=(2, 0, 1)))


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
    def jacobian(x):
        return grad(postprocess, method="defer")(x)

    jax_jacobian = jax.jacobian(postprocess)(0.5)
    catalyst_jacobian = jacobian(0.5)
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
    def jacobian(x, y):
        return grad(postprocess, argnum=(0, 1), method="defer")(x, y)

    args = (jnp.array([0.5, 0, 0]), 0.4)
    jax_jacobian = jax.jacobian(postprocess, argnums=(0, 1))(*args)
    catalyst_jacobian = jacobian(*args)
    # Catalyst returns a list of 4 values while JAX returns a 2x2 list of lists
    for i, row in enumerate(jax_jacobian):
        for j, jax_entry in enumerate(row):
            # With multiple arguments and results, the Catalyst jacobian are transposed
            # w.r.t. the JAX jacobian. This is why the i and j are switched.
            catalyst_entry = catalyst_jacobian[j * len(row) + i]
            assert catalyst_entry == pytest.approx(jax_entry.T)


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
        return grad(postprocess, method="defer")(x)

    x = jnp.array([0.1, 0.2, 0.3])
    assert grad_workflow(x) == pytest.approx(jax.jacobian(postprocess)(x))
