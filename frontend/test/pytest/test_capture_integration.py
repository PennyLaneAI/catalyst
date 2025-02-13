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
"""Integration tests for the the PL capture in Catalyst."""
import jax.numpy as jnp
import pennylane as qml
import pytest

import catalyst


def circuit_aot_builder(dev):
    """Test AOT builder."""

    @catalyst.qjit(experimental_capture=True)
    @qml.qnode(device=dev)
    def catalyst_circuit_aot(x: float):
        qml.Hadamard(wires=0)
        qml.RZ(x, wires=0)
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[1, 0])
        qml.Hadamard(wires=1)
        return qml.expval(qml.PauliY(wires=0))

    return catalyst_circuit_aot


class TestCapture:
    """Integration tests for Catalyst adjoint functionality."""

    @pytest.mark.parametrize("theta", (jnp.pi, 0.1, 0.0))
    def test_simple_circuit_aot(self, backend, theta):
        """Test the integration for a simple circuit."""
        dev = qml.device(backend, wires=2)

        @qml.qnode(device=dev)
        def pl_circuit(x):
            qml.Hadamard(wires=0)
            qml.RZ(x, wires=0)
            qml.RZ(x, wires=0)
            qml.CNOT(wires=[1, 0])
            qml.Hadamard(wires=1)
            return qml.expval(qml.PauliY(wires=0))

        actual = circuit_aot_builder(dev)(theta)
        desired = pl_circuit(theta)
        assert jnp.allclose(actual, desired)

    @pytest.mark.parametrize("capture", (True, False))
    @pytest.mark.parametrize("theta", (jnp.pi, 0.1, 0.0))
    def test_simple_circuit(self, backend, theta, capture):
        """Test the integration for a simple circuit."""
        if capture:
            qml.capture.enable()

        dev = qml.device(backend, wires=2)

        @catalyst.qjit(experimental_capture=True)
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

        if capture:
            assert qml.capture.enabled()
        else:
            assert not qml.capture.enabled()

        qml.capture.disable()
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

    @pytest.mark.parametrize("theta", (jnp.pi, 0.1, 0.0))
    def test_forloop(self, backend, theta):
        """Test the integration for a circuit with a for loop."""

        @qml.qjit(experimental_capture=True)
        @qml.qnode(qml.device(backend, wires=4))
        def catalyst_capture_circuit(x):

            @qml.for_loop(1, 4, 1)
            def loop(i):
                qml.CNOT(wires=[0, i])
                qml.RX(x, wires=i)

            loop()

            return qml.expval(qml.Z(2))

        @qml.qnode(qml.device(backend, wires=4))
        def pl_circuit(x):

            for i in range(1, 4):
                qml.CNOT(wires=[0, i])
                qml.RX(x, wires=i)

            return qml.expval(qml.Z(2))

        actual = catalyst_capture_circuit(theta)
        desired = pl_circuit(theta)
        assert jnp.allclose(actual, desired)

    def test_forloop_workflow(self, backend):
        """Test the integration for a circuit with a for loop primitive."""

        @qml.qnode(qml.device(backend, wires=1))
        def circuit(n, x):

            @qml.for_loop(1, n, 1)
            def loop_rx(_, x):
                qml.RX(x, wires=0)
                return jnp.sin(x)

            # apply the for loop
            loop_rx(x)

            return qml.expval(qml.Z(0))

        default_capture_result = qml.qjit(circuit)(10, 0.3)
        experimental_capture_result = qml.qjit(circuit, experimental_capture=True)(10, 0.3)
        assert default_capture_result == experimental_capture_result

    def test_nested_loops(self, backend):
        """Test the integration for a circuit with a nested for loop primitive."""

        @qml.qnode(qml.device(backend, wires=4))
        def circuit(n):
            # Input state: equal superposition
            @qml.for_loop(0, n, 1)
            def init(i):
                qml.Hadamard(wires=i)

            # QFT
            @qml.for_loop(0, n, 1)
            def qft(i):
                qml.Hadamard(wires=i)

                @qml.for_loop(i + 1, n, 1)
                def inner(j):
                    qml.ControlledPhaseShift(jnp.pi / 2 ** (n - j + 1), [i, j])

                inner()

            init()
            qft()

            # Expected output: |100...>
            return qml.state()

        default_capture_result = qml.qjit(circuit)(4)
        experimental_capture_result = qml.qjit(circuit, experimental_capture=True)(4)
        assert jnp.allclose(default_capture_result, jnp.eye(2**4)[0])
        assert jnp.allclose(experimental_capture_result, default_capture_result)

    def test_while_loop_workflow(self, backend):
        """Test the integration for a circuit with a while_loop primitive."""

        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float):

            @qml.while_loop(lambda x: x < 2.0)
            def loop_rx(x):
                # perform some work and update (some of) the arguments
                qml.RX(x, wires=0)
                return x**2

            # apply the while loop
            loop_rx(x)

            return qml.expval(qml.Z(0))

        default_capture_result = qml.qjit(circuit)(1.5)
        experimental_capture_result = qml.qjit(circuit, experimental_capture=True)(1.5)
        assert default_capture_result == experimental_capture_result

    def test_cond_workflow_if_else(self, backend):
        """Test the integration for a circuit with a cond primitive with true and false branches."""

        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float):

            def ansatz_true():
                qml.RX(x, wires=0)
                qml.Hadamard(wires=0)

            def ansatz_false():
                qml.RY(x, wires=0)

            qml.cond(x > 1.4, ansatz_true, ansatz_false)()

            return qml.expval(qml.Z(0))

        default_capture_result = qml.qjit(circuit)(0.1)
        experimental_capture_result = qml.qjit(circuit, experimental_capture=True)(0.1)
        assert default_capture_result == experimental_capture_result

    def test_cond_workflow_if(self, backend):
        """Test the integration for a circuit with a cond primitive with a true branch only."""

        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float):

            def ansatz_true():
                qml.RX(x, wires=0)
                qml.Hadamard(wires=0)

            qml.cond(x > 1.4, ansatz_true)()

            return qml.expval(qml.Z(0))

        default_capture_result = qml.qjit(circuit)(1.5)
        experimental_capture_result = qml.qjit(circuit, experimental_capture=True)(1.5)
        assert default_capture_result == experimental_capture_result

    def test_cond_workflow_with_custom_primitive(self, backend):
        """Test the integration for a circuit with a cond primitive containing a custom
        primitive."""

        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float):

            def ansatz_true():
                qml.RX(x, wires=0)
                qml.Hadamard(wires=0)
                qml.GlobalPhase(jnp.pi / 4)  # Custom primitive

            def ansatz_false():
                qml.RY(x, wires=0)
                qml.GlobalPhase(jnp.pi / 2)  # Custom primitive

            qml.cond(x > 1.4, ansatz_true, ansatz_false)()

            return qml.expval(qml.Z(0))

        default_capture_result = qml.qjit(circuit)(0.1)
        experimental_capture_result = qml.qjit(circuit, experimental_capture=True)(0.1)
        assert default_capture_result == experimental_capture_result

    def test_cond_workflow_with_abstract_measurement(self, backend):
        """Test the integration for a circuit with a cond primitive containing an
        abstract measurement."""

        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float):

            def ansatz_true():
                qml.RX(x, wires=0)
                qml.Hadamard(wires=0)
                qml.state()  # Abstract measurement

            def ansatz_false():
                qml.RY(x, wires=0)
                qml.state()  # Abstract measurement

            qml.cond(x > 1.4, ansatz_true, ansatz_false)()

            return qml.expval(qml.Z(0))

        default_capture_result = qml.qjit(circuit)(0.1)
        experimental_capture_result = qml.qjit(circuit, experimental_capture=True)(0.1)
        assert default_capture_result == experimental_capture_result

    def test_cond_workflow_with_simple_primitive(self, backend):
        """Test the integration for a circuit with a cond primitive containing an
        simple primitive."""

        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float):

            def ansatz_true():
                qml.RX(x, wires=0)
                qml.Hadamard(wires=0)
                return x + 1  # simple primitive

            def ansatz_false():
                qml.RY(x, wires=0)
                return x + 1  # simple primitive

            qml.cond(x > 1.4, ansatz_true, ansatz_false)()

            return qml.expval(qml.Z(0))

        default_capture_result = qml.qjit(circuit)(0.1)
        experimental_capture_result = qml.qjit(circuit, experimental_capture=True)(0.1)
        assert default_capture_result == experimental_capture_result

    def test_cond_workflow_nested(self, backend):
        """Test the integration for a circuit with a nested cond primitive."""

        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float, y: float):

            def ansatz_true():
                qml.RX(x, wires=0)
                qml.Hadamard(wires=0)

            def ansatz_false():

                def branch_true():
                    qml.RY(y, wires=0)

                def branch_false():
                    qml.RZ(y, wires=0)

                qml.cond(y > 1.4, branch_true, branch_false)()

            qml.cond(x > 1.4, ansatz_true, ansatz_false)()

            return qml.expval(qml.Z(0))

        default_capture_result = qml.qjit(circuit)(0.1, 1.5)
        experimental_capture_result = qml.qjit(circuit, experimental_capture=True)(0.1, 1.5)
        assert default_capture_result == experimental_capture_result

    def test_cond_workflow_operator(self, backend):
        """Test the integration for a circuit with a cond primitive returning
        an Operator."""

        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float):

            qml.cond(x > 1.4, qml.RX, qml.RY)(x, wires=0)

            return qml.expval(qml.Z(0))

        default_capture_result = qml.qjit(circuit)(0.1)
        experimental_capture_result = qml.qjit(circuit, experimental_capture=True)(0.1)
        assert default_capture_result == experimental_capture_result
