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

from catalyst import for_loop, measure, qjit
from catalyst.jit import JAX_QJIT


class TestJAXJIT:
    """Test QJIT compatibility with JAX compilation."""

    def test_simple_circuit_with_pytree_input(self, backend):
        """Test a basic use case of jax.jit with a dictionary as input."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(x):
            qml.RX(jnp.pi * x["a"][0], wires=0)
            qml.RY(x["a"][1] ** 2, wires=0)
            qml.RX(x["a"][1] * x["a"][2], wires=0)
            return qml.expval(qml.PauliZ(0))

        def cost_fn(x):
            result = circuit(x)
            return jnp.cos(result) ** 2

        x = {"a": jnp.array([0.1, 0.2, 0.3])}
        result = jax.jit(cost_fn)(x)
        reference = cost_fn(x)

        assert jnp.allclose(result, reference)

    def test_simple_circuit_with_pytree_output(self, backend):
        """Test a basic use case of jax.jit with a dictionary as an output."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(x: jax.core.ShapedArray((3,), dtype=float)):
            qml.RX(jnp.pi * x[0], wires=0)
            qml.RY(x[1] ** 2, wires=0)
            qml.RX(x[1] * x[2], wires=0)
            return {"a": qml.expval(qml.PauliZ(0))}

        def cost_fn(x):
            result = circuit(x)
            return jnp.cos(result["a"]) ** 2

        x = jnp.array([0.1, 0.2, 0.3])
        result = jax.jit(cost_fn)(x)
        reference = cost_fn(x)

        assert jnp.allclose(result, reference)

    def test_simple_circuit(self, backend):
        """Test a basic use case of jax.jit on top of qjit."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(x: jax.core.ShapedArray((3,), dtype=float)):
            qml.RX(jnp.pi * x[0], wires=0)
            qml.RY(x[1] ** 2, wires=0)
            qml.RX(x[1] * x[2], wires=0)
            return qml.expval(qml.PauliZ(0))

        def cost_fn(x):
            result = circuit(x)
            return jnp.cos(result) ** 2

        x = jnp.array([0.1, 0.2, 0.3])
        result = jax.jit(cost_fn)(x)
        reference = cost_fn(x)

        assert jnp.allclose(result, reference)

    def test_multiple_arguments(self, backend):
        """Test a circuit with multiple arguments using jax.jit on top of qjit."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(
            x: jax.core.ShapedArray((3,), dtype=float), y: jax.core.ShapedArray((2,), dtype=float)
        ):
            qml.RX(jnp.pi * x[0], wires=0)
            qml.RY(x[1] ** 2, wires=0)
            qml.RX(y[1] * x[2], wires=0)
            return qml.probs(wires=0)

        def cost_fn(x, y):
            result = circuit(x, y)
            return jnp.sum(jnp.cos(result) ** 2)

        x, y = jnp.array([0.1, 0.2, 0.3]), jnp.array([0.1, 0.2])
        result = jax.jit(cost_fn)(x, y)
        reference = cost_fn(x, y)

        assert jnp.allclose(result, reference)

    def test_multiple_results(self, backend):
        """Test a circuit with multiple results using jax.jit on top of qjit."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(
            x: jax.core.ShapedArray((3,), dtype=float), y: jax.core.ShapedArray((2,), dtype=float)
        ):
            qml.RX(jnp.pi * x[0], wires=0)
            qml.RY(x[1] ** 2, wires=0)
            qml.RX(y[1] * x[2], wires=0)
            return qml.probs(wires=0), qml.expval(qml.PauliZ(0))

        def cost_fn(x, y):
            result = circuit(x, y)
            return jnp.sum(jnp.cos(result[0]) ** 2) + jnp.sin(result[1]) ** 2

        x, y = jnp.array([0.1, 0.2, 0.3]), jnp.array([0.1, 0.2])
        result = jax.jit(cost_fn)(x, y)
        reference = cost_fn(x, y)

        assert jnp.allclose(result, reference)

    def test_without_precompilation(self, backend):
        """Test a function without type hints (pre-compilation) using jax.jit on top of qjit."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(x, y):
            qml.RX(jnp.pi * x[0], wires=0)
            qml.RY(x[1] ** 2, wires=0)
            qml.RX(y[1] * x[2], wires=0)
            return qml.probs(wires=0), qml.expval(qml.PauliZ(0))

        def cost_fn(x, y):
            result = circuit(x, y)
            return jnp.sum(jnp.cos(result[0]) ** 2) + jnp.sin(result[1]) ** 2

        x, y = jnp.array([0.1, 0.2, 0.3]), jnp.array([0.1, 0.2])
        result = jax.jit(cost_fn)(x, y)
        reference = cost_fn(x, y)

        assert jnp.allclose(result, reference)

    def test_multiple_calls(self, backend):
        """Test a jax.jit function which repeatedly calls a qjit function."""

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
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


class TestJAXAD:
    """Test QJIT compatibility with JAX differentiation."""

    def test_simple_circuit(self, backend):
        """Test a basic use case of jax.grad on top of qjit."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(x: jax.core.ShapedArray((3,), dtype=float)):
            qml.RX(jnp.pi * x[0], wires=0)
            qml.RY(x[1] ** 2, wires=0)
            qml.RX(x[1] * x[2], wires=0)
            return qml.expval(qml.PauliZ(0))

        @jax.grad
        def cost_fn(x, qfunc):
            result = qfunc(x)
            return jnp.cos(result) ** 2

        x = jnp.array([0.1, 0.2, 0.3])
        result = cost_fn(x, circuit)
        reference = cost_fn(x, circuit.user_function)

        assert jnp.allclose(result, reference)

    @pytest.mark.parametrize("argnums", (0, 1, [0, 1]))
    def test_multiple_arguments(self, backend, argnums):
        """Test a circuit with multiple arguments using jax.grad on top of qjit."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(
            x: jax.core.ShapedArray((3,), dtype=float), y: jax.core.ShapedArray((2,), dtype=float)
        ):
            qml.RX(jnp.pi * x[0], wires=0)
            qml.RY(x[1] ** 2, wires=0)
            qml.RX(y[1] * x[2], wires=0)
            return qml.probs(wires=0)

        @partial(jax.grad, argnums=argnums)
        def cost_fn(x, y, qfunc):
            result = qfunc(x, y)
            return jnp.sum(jnp.cos(result) ** 2)

        x, y = jnp.array([0.1, 0.2, 0.3]), jnp.array([0.1, 0.2])
        result = cost_fn(x, y, circuit)
        reference = cost_fn(x, y, circuit.user_function)

        assert jnp.allclose(result[0], reference[0])
        if isinstance(argnums, list):
            assert jnp.allclose(result[1], reference[1])

    def test_multiple_results(self, backend):
        """Test a circuit with multiple results using jax.grad on top of qjit."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(
            x: jax.core.ShapedArray((3,), dtype=float), y: jax.core.ShapedArray((2,), dtype=float)
        ):
            qml.RX(jnp.pi * x[0], wires=0)
            qml.RY(x[1] ** 2, wires=0)
            qml.RX(y[1] * x[2], wires=0)
            return qml.probs(wires=0), qml.expval(qml.PauliZ(0))

        @partial(jax.grad, argnums=[0, 1])
        def cost_fn(x, y, qfunc):
            result = qfunc(x, y)
            return jnp.sum(jnp.cos(result[0]) ** 2) + jnp.sin(result[1]) ** 2

        x, y = jnp.array([0.1, 0.2, 0.3]), jnp.array([0.1, 0.2])
        result = cost_fn(x, y, circuit)
        reference = cost_fn(x, y, circuit.user_function)

        assert len(result) == 2
        assert jnp.allclose(result[0], reference[0])
        assert jnp.allclose(result[1], reference[1])

    def test_jacobian(self, backend):
        """Test a circuit with vector return type using jax.jacobian on top of qjit."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(
            x: jax.core.ShapedArray((3,), dtype=float), y: jax.core.ShapedArray((2,), dtype=float)
        ):
            qml.RX(jnp.pi * x[0], wires=0)
            qml.RY(x[1] ** 2, wires=0)
            qml.RX(y[1] * x[2], wires=0)
            return qml.probs(wires=0)

        @partial(jax.jacobian, argnums=[0, 1])
        def cost_fn(x, y, qfunc):
            result = qfunc(x, y)
            return jnp.cos(result) ** 2

        x, y = jnp.array([0.1, 0.2, 0.3]), jnp.array([0.1, 0.2])
        result = cost_fn(x, y, circuit)
        reference = cost_fn(x, y, circuit.user_function)

        assert len(result) == 2
        assert jnp.allclose(result[0], reference[0])
        assert jnp.allclose(result[1], reference[1])

    def test_without_precompilation(self, backend):
        """Test a function without type hints (pre-compilation) using jax.grad on top of qjit."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(x, y):
            qml.RX(jnp.pi * x[0], wires=0)
            qml.RY(x[1] ** 2, wires=0)
            qml.RX(y[1] * x[2], wires=0)
            return qml.probs(wires=0), qml.expval(qml.PauliZ(0))

        @partial(jax.grad, argnums=[0, 1])
        def cost_fn(x, y, qfunc):
            result = qfunc(x, y)
            return jnp.sum(jnp.cos(result[0]) ** 2) + jnp.sin(result[1]) ** 2

        x, y = jnp.array([0.1, 0.2, 0.3]), jnp.array([0.1, 0.2])
        result = cost_fn(x, y, circuit)
        reference = cost_fn(x, y, circuit.user_function)

        assert len(result) == 2
        assert jnp.allclose(result[0], reference[0])
        assert jnp.allclose(result[1], reference[1])

    def test_non_differentiable_arguments(self, backend):
        """Test a circuit with non-differentiable arguments using jax.grad on top of qjit."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(x: jax.core.ShapedArray((3,), dtype=float), y: int):
            qml.RX(jnp.pi * x[0], wires=0)
            qml.RY(x[1] ** 2, wires=0)
            qml.RX(y * x[2], wires=0)
            return qml.probs(wires=0), qml.expval(qml.PauliZ(0))

        @jax.grad
        def cost_fn(x, y, qfunc):
            result = qfunc(x, y)
            return jnp.sum(jnp.cos(result[0]) ** 2) + jnp.sin(result[1]) ** 2

        x, y = jnp.array([0.1, 0.2, 0.3]), 3
        result = cost_fn(x, y, circuit)
        reference = cost_fn(x, y, circuit.user_function)

        assert jnp.allclose(result, reference)

    def test_multiple_calls(self, backend):
        """Test a jax.grad function which repeatedly calls a qjit function."""

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @jax.grad
        def cost_fn(x, y):
            m1 = circuit(x)
            m2 = circuit(y)
            return m1 + m2

        result1 = cost_fn(0.0, 0.0)
        result2 = cost_fn(0.0, jnp.pi)
        assert jnp.allclose(result1, 0.0)
        assert jnp.allclose(result2, 0.0)

    @pytest.mark.parametrize("shape", ([2, 3], [3, 2], [1, 6]))
    def test_multiD_calls(self, shape):
        """Test a jax.grad in combination with qjit on non-1D input parameters."""

        def func(p1, p2):
            return jnp.reshape(p1, shape) + 2 * jnp.reshape(p2, shape)

        C_func = qjit(func)
        PL_func = func

        def cost_fn(p1, p2, f):
            m1 = f(p1, p2)
            m2 = f(p1, p2)
            return m1 + m2

        p1 = jnp.array([[0.1, 0.3, 0.5], [0.1, 0.2, 0.8]])
        p2 = jnp.array([[0.3, 0.5], [0.2, 0.8], [0.2, 0.8]])
        result = jax.jacobian(cost_fn, argnums=[0, 1])(p1, p2, C_func)
        reference = jax.jacobian(cost_fn, argnums=[0, 1])(p1, p2, PL_func)
        assert len(result) == len(reference)
        for a, b in zip(result, reference):
            assert a.shape == b.shape
            assert jnp.allclose(a, b, rtol=1e-6, atol=1e-6)

    def test_efficient_Jacobian(self, backend):
        """Test a jax.grad function does not compute Jacobians for arguments not in argnums."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(x: float, y: float):
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            return qml.expval(qml.PauliZ(0))

        @partial(jax.grad, argnums=0)
        def cost_fn(x, y):
            return circuit(x, y)

        cost_fn(0.1, 0.2)

        assert len(circuit.jaxed_function.derivative_functions) == 1
        assert "0" in circuit.jaxed_function.derivative_functions
        assert len(circuit.jaxed_function.derivative_functions["0"].jaxpr.out_avals) == 1

    def test_jit_and_grad(self, backend):
        """Test that argnums determination works correctly when combining jax.jit with jax.grad.
        This was fixed by the introduction of symbolic zero detection for tangent vectors."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(params: jax.core.ShapedArray((2,), dtype=float), n: int):
            qml.RX(n * params[0], wires=0)
            qml.RY(params[1] / n, wires=1)
            qml.CNOT(wires=[0, 1])

            return qml.expval(qml.PauliZ(1))

        n = 3
        params = jnp.array([0.1, 0.2])

        result = jax.grad(jax.jit(circuit), argnums=0)(params, n)
        reference = jax.grad(circuit, argnums=0)(params, n)

        assert jnp.allclose(result, reference)

    def test_argnums_passed(self, backend, monkeypatch):
        """Test that when combining jax.jit and jax.grad, the internal argnums are correctly
        passed to the custom quantum JVP"""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(p1, n, p2):
            def ansatz(_):
                qml.RX(p1, wires=0)
                qml.RY(p2, wires=1)
                qml.CNOT(wires=[0, 1])

            for_loop(0, n, 1)(ansatz)()

            return qml.expval(qml.PauliZ(1))

        # Patch the quantum gradient wrapper to verify the internal argnums
        get_derivative_qjit = JAX_QJIT.get_derivative_qjit

        def get_derivative_qjit_wrapper(self, argnums):
            assert argnums == [0, 2]
            return get_derivative_qjit(self, argnums)

        monkeypatch.setattr(JAX_QJIT, "get_derivative_qjit", get_derivative_qjit_wrapper)

        jax.grad(jax.jit(circuit), argnums=(0, 2))(-4.5, 3, 4.3)


class TestJAXRecompilation:
    """
    Test obtained from ticket: https://github.com/PennyLaneAI/catalyst/issues/149

    JAX is asked the gradient of a function, but the function itself might need recompilation.
    """

    def test_jax_function_has_not_been_jit_compiled(self, backend):
        """Test if function can be used by jax.grad even if it has not been JIT compiled"""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(params, n):
            def ansatz(i, x):
                qml.RX(x[i, 0], wires=0)
                qml.RY(x[i, 1], wires=1)
                qml.CNOT(wires=[0, 1])
                return x

            for_loop(0, n, 1)(ansatz)(jnp.reshape(params, (-1, 2)))

            return qml.expval(qml.PauliZ(1))

        params = jnp.array([0.54, 0.3154, 0.654, 0.123])
        jax.grad(circuit, argnums=0)(params, 2)

    def test_jax_function_needs_recompilation(self, backend):
        """Test if function can be used by jax.grad but it needs recompilation"""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(params, n):
            def ansatz(i, x):
                qml.RX(x[i, 0], wires=0)
                qml.RY(x[i, 1], wires=1)
                qml.CNOT(wires=[0, 1])
                return x

            for_loop(0, n, 1)(ansatz)(jnp.reshape(params, (-1, 2)))

            return qml.expval(qml.PauliZ(1))

        params = jnp.array([0.54, 0.3154, 0.654, 0.123])
        jax.grad(circuit, argnums=0)(params, 2)
        params = jnp.array([0.54, 0.3154, 0.654, 0.123, 0.1, 0.2])
        jax.grad(circuit, argnums=0)(params, 3)

    def test_pytree_qml_counts_simple(self):
        """Test if a single qml.counts() can be used and output correctly."""
        dev = qml.device("lightning.qubit", wires=1, shots=20)

        @qjit
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return {"1": qml.counts()}

        result = circuit(0.5)
        _, result_tree = jax.tree.flatten(result)
        assert "PyTreeDef({'1': (*, *)})" == str(result_tree)

    def test_pytree_qml_counts_nested(self):
        """Test if nested qml.counts() can be used and output correctly."""
        dev = qml.device("lightning.qubit", wires=1, shots=20)

        @qjit
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return {"1": qml.counts()}, {"2": qml.expval(qml.Z(0))}

        result = circuit(0.5)
        _, result_tree = jax.tree.flatten(result)
        assert "PyTreeDef(({'1': (*, *)}, {'2': *}))" == str(result_tree)

        @qjit
        @qml.qnode(dev)
        def circuit2(x):
            qml.RX(x, wires=0)
            return [{"1": qml.expval(qml.Z(0))}, {"2": qml.counts()}], {"3": qml.expval(qml.Z(0))}

        result = circuit2(0.5)
        _, result_tree = jax.tree.flatten(result)

        assert "PyTreeDef(([{'1': *}, {'2': (*, *)}], {'3': *}))" == str(result_tree)

    def test_pytree_qml_counts_2_nested(self):
        """Test if multiple nested qml.counts() can be used and output correctly."""
        dev = qml.device("lightning.qubit", wires=1, shots=20)

        @qjit
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return [{"1": qml.expval(qml.Z(0))}, {"2": qml.counts()}], [
                {"3": qml.expval(qml.Z(0))},
                {"4": qml.counts()},
            ]

        result = circuit(0.5)
        _, result_tree = jax.tree.flatten(result)
        assert "PyTreeDef(([{'1': *}, {'2': (*, *)}], [{'3': *}, {'4': (*, *)}]))" == str(
            result_tree
        )

        @qjit
        @qml.qnode(dev)
        def circuit2(x):
            qml.RX(x, wires=0)
            return [{"1": qml.expval(qml.Z(0))}, {"2": qml.counts()}], [
                {"3": qml.counts()},
                {"4": qml.expval(qml.Z(0))},
            ]

        result = circuit2(0.5)
        _, result_tree = jax.tree.flatten(result)

        assert "PyTreeDef(([{'1': *}, {'2': (*, *)}], [{'3': (*, *)}, {'4': *}]))" == str(
            result_tree
        )

    def test_pytree_qml_counts_longer(self):
        """Test if 3 differently nested qml.counts() can be used and output correctly."""
        dev = qml.device("lightning.qubit", wires=1, shots=20)

        @qjit
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return [
                [{"1": qml.expval(qml.Z(0))}, {"2": qml.counts()}],
                [{"3": qml.expval(qml.Z(0))}, {"4": qml.counts()}],
                {"5": qml.expval(qml.Z(0))},
                {"6": qml.counts()},
            ]

        result = circuit(0.5)
        _, result_tree = jax.tree.flatten(result)
        assert (
            "PyTreeDef([[{'1': *}, {'2': (*, *)}], "
            + "[{'3': *}, {'4': (*, *)}], {'5': *}, {'6': (*, *)}])"
            == str(result_tree)
        )


if __name__ == "__main__":
    pytest.main(["-x", __file__])
