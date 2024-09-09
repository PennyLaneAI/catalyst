# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Integration tests for the the PL capture in Catalyst.
"""
import jax.numpy as jnp
import pennylane as qml
import pytest

import catalyst


class TestCapture:
    """Integration tests for Catalyst adjoint functionality."""

    @pytest.mark.parametrize("theta", (jnp.pi, 0.1, 0.0))
    def test_simple_circuit(self, backend, theta):
        """Test the integration for a simple circuit."""
        dev = qml.device(backend, wires=2)

        @qml.qnode(device=dev)
        def catalyst_circuit(x):
            qml.Hadamard(wires=0)
            qml.RZ(x, wires=0)
            qml.RZ(x, wires=0)
            qml.CNOT(wires=[1, 0])
            qml.Hadamard(wires=1)
            return qml.expval(qml.PauliY(wires=0))

        @qml.qnode(device=dev)
        def pl_circuit(x):
            qml.Hadamard(wires=0)
            qml.RZ(x, wires=0)
            qml.RZ(x, wires=0)
            qml.CNOT(wires=[1, 0])
            qml.Hadamard(wires=1)
            return qml.expval(qml.PauliY(wires=0))

        actual = catalyst_circuit(theta)
        desired = pl_circuit(theta)
        assert jnp.allclose(actual, desired)

    @pytest.mark.parametrize("theta", (jnp.pi, 0.1, 0.0))
    def test_simple_workflow(self, backend, theta):
        """Test the integration for a simple workflow."""
        dev = qml.device(backend, wires=2)

        @qml.qnode(device=dev)
        def circuit(x):
            qml.Hadamard(wires=0)
            qml.RZ(x, wires=0)
            qml.RZ(x, wires=0)
            qml.CNOT(wires=[1, 0])
            qml.Hadamard(wires=1)
            return qml.expval(qml.PauliY(wires=0))

        @catalyst.qjit(experimental_capture=True)
        def f(x):
            return circuit(x**2) ** 2

        @catalyst.qjit
        def g(x):
            return circuit(x**2) ** 2

        actual = f(theta)
        desired = g(theta)

        assert jnp.allclose(actual, desired)

    @pytest.mark.xfail(reason="Adjoint not supported.")
    @pytest.mark.parametrize("theta, val", [(jnp.pi, 0), (-100.0, 1)])
    def test_adjoint(self, backend, theta, val):
        """Test the integration for a circuit with adjoint."""
        device = qml.device(backend, wires=2)

        @qml.qjit(experimental_capture=True)
        @qml.qnode(device)
        def catalyst_circuit(theta, val):
            qml.adjoint(qml.RY)(jnp.pi, val)
            qml.adjoint(qml.RZ)(theta, wires=val)
            return qml.state()

        @qml.qnode(device)
        def pl_circuit(theta, val):
            qml.adjoint(qml.RY)(jnp.pi, val)
            qml.adjoint(qml.RZ)(theta, wires=val)
            return qml.state()

        actual = catalyst_circuit(theta, val)
        desired = pl_circuit(theta, val)
        assert jnp.allclose(actual, desired)

    @pytest.mark.xfail(reason="Ctrl not supported.")
    @pytest.mark.parametrize("theta", (jnp.pi, 0.1, 0.0))
    def test_ctrl(self, backend, theta):
        """Test the integration for a circuit with control."""
        device = qml.device(backend, wires=3)

        @qml.qjit(experimental_capture=True)
        @qml.qnode(device)
        def catalyst_circuit(theta):
            qml.ctrl(qml.RX(theta, wires=0), control=[1], control_values=[False])
            qml.ctrl(qml.RX, control=[1], control_values=[False])(theta, wires=[0])
            return qml.state()

        @qml.qnode(device)
        def pl_circuit(theta):
            qml.ctrl(qml.RX(theta, wires=0), control=[1], control_values=[False])
            qml.ctrl(qml.RX, control=[1], control_values=[False])(theta, wires=[0])
            return qml.state()

        actual = catalyst_circuit(theta)
        desired = pl_circuit(theta)
        assert jnp.allclose(actual, desired)

    @pytest.mark.xfail(reason="For not supported.")
    @pytest.mark.parametrize("theta", (jnp.pi, 0.1, 0.0))
    def test_forloop(self, backend, theta):
        """Test the integration for a circuit with a for loop."""

        @qml.qjit(experimental_capture=True)
        @qml.qnode(qml.device(backend, wires=4))
        def catalyst_capture_circuit(x):

            @qml.for_loop(1, 1, 4)
            def loop(i):
                qml.CNOT(wires=[0, i])
                qml.RX(x, wires=i)

            loop()

            return qml.expval(qml.Z(2))

        @qml.qnode(qml.device(backend, wires=4))
        def catalyst_circuit(x):

            for i in range(1, 4):
                qml.CNOT(wires=[0, i])
                qml.RX(x, wires=i)

            return qml.expval(qml.Z(2))

        actual = catalyst_capture_circuit(theta)
        desired = catalyst_circuit(theta)
        assert jnp.allclose(actual, desired)
