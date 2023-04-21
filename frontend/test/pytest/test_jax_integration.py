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

"""Test QJIT compatibility with JAX transformations such as jax.jit and jax.grad."""

from functools import partial

import jax
import jax.numpy as jnp
import pennylane as qml
import pytest

from catalyst import measure, qjit


class TestJAXJIT:
    """Test QJIT compatibility with JAX compilation."""

    def test_simple_circuit(self):
        """Test a basic use case of jax.jit on top of qjit."""

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(x: jax.ShapedArray((3,), dtype=float)):
            qml.RX(jnp.pi * x[0], wires=0)
            qml.RY(x[1] ** 2, wires=0)
            qml.RX(x[1] * x[2], wires=0)
            return qml.expval(qml.PauliZ(0))

        @jax.jit
        def cost_fn(x):
            result = circuit(x)
            return jnp.cos(result) ** 2

        result = cost_fn(jnp.array([0.1, 0.2, 0.3]))
        assert jnp.allclose(result, 0.3573519347253657)

    def test_multiple_arguments(self):
        """Test a circuit with multiple arguments using jax.jit on top of qjit."""

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(x: jax.ShapedArray((3,), dtype=float), y: jax.ShapedArray((2,), dtype=float)):
            qml.RX(jnp.pi * x[0], wires=0)
            qml.RY(x[1] ** 2, wires=0)
            qml.RX(y[1] * x[2], wires=0)
            return qml.probs(wires=0)

        @jax.jit
        def cost_fn(x, y):
            result = circuit(x, y)
            return jnp.sum(jnp.cos(result) ** 2)

        result = cost_fn(jnp.array([0.1, 0.2, 0.3]), jnp.array([0.1, 0.2]))
        assert jnp.allclose(result, 1.3229868863865866)

    def test_multiple_results(self):
        """Test a circuit with multiple results using jax.jit on top of qjit."""

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(x: jax.ShapedArray((3,), dtype=float), y: jax.ShapedArray((2,), dtype=float)):
            qml.RX(jnp.pi * x[0], wires=0)
            qml.RY(x[1] ** 2, wires=0)
            qml.RX(y[1] * x[2], wires=0)
            return qml.probs(wires=0), qml.expval(qml.PauliZ(0))

        @jax.jit
        def cost_fn(x, y):
            result = circuit(x, y)
            return jnp.sum(jnp.cos(result[0]) ** 2) + jnp.sin(result[1]) ** 2

        result = cost_fn(jnp.array([0.1, 0.2, 0.3]), jnp.array([0.1, 0.2]))
        assert jnp.allclose(result, 1.9656349516612208)

    def test_without_precompilation(self):
        """Test a function without type hints (pre-compilation) using jax.jit on top of qjit."""

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(x, y):
            qml.RX(jnp.pi * x[0], wires=0)
            qml.RY(x[1] ** 2, wires=0)
            qml.RX(y[1] * x[2], wires=0)
            return qml.probs(wires=0), qml.expval(qml.PauliZ(0))

        @jax.jit
        def cost_fn(x, y):
            result = circuit(x, y)
            return jnp.sum(jnp.cos(result[0]) ** 2) + jnp.sin(result[1]) ** 2

        result = cost_fn(jnp.array([0.1, 0.2, 0.3]), jnp.array([0.1, 0.2]))
        assert jnp.allclose(result, 1.9656349516612208)

    def test_multiple_calls(self):
        """Test a jax.jit function which repeatedly calls a qjit function."""

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(x):
            qml.RY(x, wires=0)
            return measure(0)

        @jax.jit
        def cost_fn(x, y):
            m1 = circuit(x)
            m2 = circuit(y)
            return m1 == m2

        result1 = cost_fn(0.0, 0.0)
        result2 = cost_fn(0.0, jnp.pi)
        assert result1 == True
        assert result2 == False


class TestJAXGrad:
    """Test QJIT compatibility with JAX differentiation."""

    def test_simple_circuit(self):
        """Test a basic use case of jax.grad on top of qjit."""

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(x: jax.ShapedArray((3,), dtype=float)):
            qml.RX(jnp.pi * x[0], wires=0)
            qml.RY(x[1] ** 2, wires=0)
            qml.RX(x[1] * x[2], wires=0)
            return qml.expval(qml.PauliZ(0))

        @jax.grad
        def cost_fn(x):
            result = circuit(x)
            return jnp.cos(result) ** 2

        result = cost_fn(jnp.array([0.1, 0.2, 0.3]))
        assert jnp.allclose(result, jnp.array([1.09975732, 0.11963127, 0.07005128]))

    def test_multiple_arguments(self):
        """Test a circuit with multiple arguments using jax.grad on top of qjit."""

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(x: jax.ShapedArray((3,), dtype=float), y: jax.ShapedArray((2,), dtype=float)):
            qml.RX(jnp.pi * x[0], wires=0)
            qml.RY(x[1] ** 2, wires=0)
            qml.RX(y[1] * x[2], wires=0)
            return qml.probs(wires=0)

        @partial(jax.grad, argnums=[0, 1])
        def cost_fn(x, y):
            result = circuit(x, y)
            return jnp.sum(jnp.cos(result) ** 2)

        result = cost_fn(jnp.array([0.1, 0.2, 0.3]), jnp.array([0.1, 0.2]))
        assert len(result) == 2
        assert jnp.allclose(result[0], jnp.array([0.49699912, 0.00657736, 0.03165737]))
        assert jnp.allclose(result[1], jnp.array([0.0, 0.04748605]))

    def test_multiple_results(self):
        """Test a circuit with multiple results using jax.grad on top of qjit."""

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(x: jax.ShapedArray((3,), dtype=float), y: jax.ShapedArray((2,), dtype=float)):
            qml.RX(jnp.pi * x[0], wires=0)
            qml.RY(x[1] ** 2, wires=0)
            qml.RX(y[1] * x[2], wires=0)
            return qml.probs(wires=0), qml.expval(qml.PauliZ(0))

        @partial(jax.grad, argnums=[0, 1])
        def cost_fn(x, y):
            result = circuit(x, y)
            return jnp.sum(jnp.cos(result[0]) ** 2) + jnp.sin(result[1]) ** 2

        result = cost_fn(jnp.array([0.1, 0.2, 0.3]), jnp.array([0.1, 0.2]))
        assert len(result) == 2
        assert jnp.allclose(result[0], jnp.array([-0.6027582, -0.00797699, -0.03839391]))
        assert jnp.allclose(result[1], jnp.array([0.0, -0.05759086]))

    def test_without_precompilation(self):
        """Test a function without type hints (pre-compilation) using jax.grad on top of qjit."""

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(x, y):
            qml.RX(jnp.pi * x[0], wires=0)
            qml.RY(x[1] ** 2, wires=0)
            qml.RX(y[1] * x[2], wires=0)
            return qml.probs(wires=0), qml.expval(qml.PauliZ(0))

        @partial(jax.grad, argnums=[0, 1])
        def cost_fn(x, y):
            result = circuit(x, y)
            return jnp.sum(jnp.cos(result[0]) ** 2) + jnp.sin(result[1]) ** 2

        result = cost_fn(jnp.array([0.1, 0.2, 0.3]), jnp.array([0.1, 0.2]))
        assert len(result) == 2
        assert jnp.allclose(result[0], jnp.array([-0.6027582, -0.00797699, -0.03839391]))
        assert jnp.allclose(result[1], jnp.array([0.0, -0.05759086]))

    def test_non_differentiable_arguments(self):
        """Test a circuit with non-differentiable arguments using jax.grad on top of qjit."""

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(x: jax.ShapedArray((3,), dtype=float), y: int):
            qml.RX(jnp.pi * x[0], wires=0)
            qml.RY(x[1] ** 2, wires=0)
            qml.RX(y * x[2], wires=0)
            return qml.probs(wires=0), qml.expval(qml.PauliZ(0))

        @partial(jax.grad)
        def cost_fn(x, y):
            result = circuit(x, y)
            return jnp.sum(jnp.cos(result[0]) ** 2) + jnp.sin(result[1]) ** 2

        result = cost_fn(jnp.array([0.1, 0.2, 0.3]), 3)
        assert jnp.allclose(result, jnp.array([-1.34682245, -0.00432698, -1.2855136]))

    def test_multiple_calls(self):
        """Test a jax.grad function which repeatedly calls a qjit function."""

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(x):
            qml.RY(x, wires=0)
            return jnp.asarray(measure(0), dtype=float)

        @jax.grad
        def cost_fn(x, y):
            m1 = circuit(x)
            m2 = circuit(y)
            return m1 + m2

        result1 = cost_fn(0.0, 0.0)
        result2 = cost_fn(0.0, jnp.pi)
        assert jnp.allclose(result1, 0.0)
        assert jnp.allclose(result2, 0.0)

    def test_efficient_Jacobian(self):
        """Test a jax.grad function does not compute Jacobians for arguments not in argnum."""

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(x: float, y: float):
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            return qml.expval(qml.PauliZ(0))

        @partial(jax.grad, argnums=0)
        def cost_fn(x, y):
            return circuit(x, y)

        assert jnp.allclose(cost_fn(0.1, 0.2), -0.09784339500725572)

        assert len(circuit.jaxed_qfunc.deriv_qfuncs) == 1
        assert "0" in circuit.jaxed_qfunc.deriv_qfuncs
        assert len(circuit.jaxed_qfunc.deriv_qfuncs["0"].jaxpr.out_avals) == 1


if __name__ == "__main__":
    pytest.main(["-x", __file__])
