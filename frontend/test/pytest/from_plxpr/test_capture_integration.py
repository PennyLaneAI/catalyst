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

# pylint: disable=too-many-lines

from functools import partial

import jax.numpy as jnp
import pennylane as qp
import pytest
from jax.core import ShapedArray

import catalyst
from catalyst import qjit

pytestmark = pytest.mark.usefixtures("disable_capture")


def circuit_aot_builder(dev):
    """Test AOT builder."""

    qp.capture.enable()

    @catalyst.qjit
    @qp.qnode(device=dev)
    def catalyst_circuit_aot(x: float):
        qp.Hadamard(wires=0)
        qp.RZ(x, wires=0)
        qp.RZ(x, wires=0)
        qp.CNOT(wires=[1, 0])
        qp.Hadamard(wires=1)
        return qp.expval(qp.PauliY(wires=0))

    pass

    return catalyst_circuit_aot


def has_catalyst_transforms(mlir):
    """Check in the MLIR if the transforms were scheduled"""
    return (
        'transform.apply_registered_pass "cancel-inverses"' in mlir
        and 'transform.apply_registered_pass "merge-rotations"' in mlir
    )


def is_unitary_rotated(mlir):
    """Check in the MLIR if a unitary was rotated"""
    return (
        "quantum.unitary" not in mlir
        and mlir.count('quantum.custom "RZ"') == 2
        and mlir.count('quantum.custom "RY"') == 1
    )


def is_rot_decomposed(mlir):
    """Check in the MLIR if a rot was decomposed"""
    return (
        'quantum.custom "Rot"' not in mlir
        and mlir.count('quantum.custom "RZ"') == 2
        and mlir.count('quantum.custom "RY"') == 1
    )


def is_wire_mapped(mlir):
    """Check in the MLIR if a wire was mapped"""
    return "quantum.extract %0[ 0]" not in mlir and "quantum.extract %0[ 1]" in mlir


def is_single_qubit_fusion_applied(mlir):
    """Check in the MLIR if 'single_qubit_fusion' was applied"""
    return (
        mlir.count('quantum.custom "Rot"') == 1
        and 'quantum.custom "Hadamard"' not in mlir
        and 'quantum.custom "RZ"' not in mlir
    )


def is_controlled_pushed_back(mlir, non_controlled_string, controlled_string):
    """Check in the MLIR if the controlled gate got pushed after the non-controlled one"""
    non_controlled_pos = mlir.find(non_controlled_string)
    assert non_controlled_pos > 0

    remaining_mlir = mlir[non_controlled_pos + len(non_controlled_string) :]
    return controlled_string in remaining_mlir


def is_amplitude_embedding_merged_and_decomposed(mlir):
    """Check in the MLIR if the amplitude embeddings got merged and decomposed"""
    return (
        "AmplitudeEmbedding" not in mlir
        and mlir.count('quantum.custom "RY"') == 3
        and mlir.count('quantum.custom "CNOT"') == 2
    )


# pylint: disable=too-many-public-methods
class TestCapture:
    """Integration tests for functionality."""

    @pytest.mark.parametrize("theta", (jnp.pi, 0.1, 0.0))
    def test_simple_circuit_aot(self, backend, theta):
        """Test the integration for a simple circuit."""
        dev = qp.device(backend, wires=2)

        @qp.qnode(device=dev)
        def circuit(x):
            qp.Hadamard(wires=0)
            qp.RZ(x, wires=0)
            qp.RZ(x, wires=0)
            qp.CNOT(wires=[1, 0])
            qp.Hadamard(wires=1)
            qp.PCPhase(x, 2, wires=[0])
            return qp.expval(qp.PauliY(wires=0))

        captured_result = circuit_aot_builder(dev)(theta)
        result = circuit(theta)
        assert jnp.allclose(captured_result, result), [captured_result, result]

    @pytest.mark.parametrize("capture", (True, False))
    @pytest.mark.parametrize("theta", (jnp.pi, 0.1, 0.0))
    def test_simple_circuit(self, backend, theta, capture):
        """Test the integration for a simple circuit."""

        dev = qp.device(backend, wires=2)

        qp.capture.enable()

        @catalyst.qjit
        @qp.qnode(device=dev)
        def captured_circuit(x):
            qp.Hadamard(wires=0)
            qp.RZ(x, wires=0)
            qp.RZ(x, wires=0)
            qp.CNOT(wires=[1, 0])
            qp.Hadamard(wires=1)
            return qp.expval(qp.PauliY(wires=0))

        capture_result = captured_circuit(theta)

        pass

        if capture:
            qp.capture.enable()

        @qp.qnode(device=dev)
        def circuit(x):
            qp.Hadamard(wires=0)
            qp.RZ(x, wires=0)
            qp.RZ(x, wires=0)
            qp.CNOT(wires=[1, 0])
            qp.Hadamard(wires=1)
            return qp.expval(qp.PauliY(wires=0))

        if capture:
            assert qp.capture.enabled()
        else:
            assert not qp.capture.enabled()

        pass

        assert jnp.allclose(capture_result, circuit(theta))

    @pytest.mark.parametrize("theta", (jnp.pi, 0.1, 0.0))
    def test_simple_workflow(self, backend, theta):
        """Test the integration for a simple workflow."""
        dev = qp.device(backend, wires=2)

        # Capture enabled

        qp.capture.enable()

        @qjit
        @qp.qnode(device=dev)
        def captured_circuit(x):
            qp.Hadamard(wires=0)
            qp.RZ(x, wires=0)
            qp.RZ(x, wires=0)
            qp.CNOT(wires=[1, 0])
            qp.Hadamard(wires=1)
            return qp.expval(qp.PauliY(wires=0))

        capture_result = captured_circuit(theta**2)

        pass

        # Capture disabled

        @qjit
        @qp.qnode(device=dev)
        def circuit(x):
            qp.Hadamard(wires=0)
            qp.RZ(x, wires=0)
            qp.RZ(x, wires=0)
            qp.CNOT(wires=[1, 0])
            qp.Hadamard(wires=1)
            return qp.expval(qp.PauliY(wires=0))

        assert jnp.allclose(capture_result, circuit(theta**2))

    @pytest.mark.parametrize(
        "n_wires, basis_state",
        [
            (1, jnp.array([0])),
            (1, jnp.array([1])),
            (2, jnp.array([0, 0])),
            (2, jnp.array([0, 1])),
            (2, jnp.array([1, 0])),
            (2, jnp.array([1, 1])),
        ],
    )
    def test_basis_state(self, backend, n_wires, basis_state):
        """Test the integration for a circuit with BasisState."""
        dev = qp.device(backend, wires=n_wires)

        # Capture enabled

        qp.capture.enable()

        @qjit
        @qp.qnode(dev)
        def captured_circuit(_basis_state):
            qp.BasisState(_basis_state, wires=list(range(n_wires)))
            return qp.state()

        capture_result = captured_circuit(basis_state)

        pass

        # Capture disabled

        @qjit
        @qp.qnode(dev)
        def circuit(_basis_state):
            qp.BasisState(_basis_state, wires=list(range(n_wires)))
            return qp.state()

        assert jnp.allclose(capture_result, circuit(basis_state))

    @pytest.mark.parametrize(
        "n_wires, init_state",
        [
            (1, jnp.array([1.0, 0.0], dtype=jnp.complex128)),
            (1, jnp.array([0.0, 1.0], dtype=jnp.complex128)),
            (1, jnp.array([1.0, 1.0], dtype=jnp.complex128) / jnp.sqrt(2.0)),
            (1, jnp.array([0.0, 1.0], dtype=jnp.float64)),
            (1, jnp.array([0, 1], dtype=jnp.int64)),
            (2, jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.complex128)),
            (2, jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.complex128)),
        ],
    )
    def test_state_prep(self, backend, n_wires, init_state):
        """Test the integration for a circuit with StatePrep."""
        dev = qp.device(backend, wires=n_wires)

        # Capture enabled

        qp.capture.enable()

        @qjit
        @qp.qnode(dev)
        def captured_circuit(init_state):
            qp.StatePrep(init_state, wires=list(range(n_wires)))
            return qp.state()

        capture_result = captured_circuit(init_state)

        pass

        # Capture disabled

        @qjit
        @qp.qnode(dev)
        def circuit(init_state):
            qp.StatePrep(init_state, wires=list(range(n_wires)))
            return qp.state()

        assert jnp.allclose(capture_result, circuit(init_state))

    @pytest.mark.parametrize("theta, val", [(jnp.pi, 0), (-100.0, 1)])
    def test_adjoint(self, backend, theta, val):
        """Test the integration for a circuit with adjoint."""
        device = qp.device(backend, wires=2)

        # Capture enabled

        qp.capture.enable()

        @qjit
        @qp.qnode(device)
        def captured_circuit(theta, val):
            qp.adjoint(qp.RY)(jnp.pi, val)
            qp.adjoint(qp.RZ)(theta, wires=val)
            return qp.state()

        capture_result = captured_circuit(theta, val)

        pass

        # Capture disabled

        @qp.qnode(device)
        def circuit(theta, val):
            qp.adjoint(qp.RY)(jnp.pi, val)
            qp.adjoint(qp.RZ)(theta, wires=val)
            return qp.state()

        assert jnp.allclose(capture_result, circuit(theta, val))

    @pytest.mark.parametrize("theta", (jnp.pi, 0.1, 0.0))
    def test_ctrl(self, backend, theta):
        """Test the integration for a circuit with control."""

        device = qp.device(backend, wires=3)

        # Capture enabled

        qp.capture.enable()

        @qjit
        @qp.qnode(device)
        def captured_circuit(theta):
            qp.ctrl(qp.RX(theta, wires=0), control=[1], control_values=[False])
            qp.ctrl(qp.RX, control=[1], control_values=[False])(theta, wires=[0])
            return qp.state()

        capture_result = captured_circuit(theta)

        pass

        # Capture disabled

        @qp.qnode(device)
        def circuit(theta):
            qp.ctrl(qp.RX(theta, wires=0), control=[1], control_values=[False])
            qp.ctrl(qp.RX, control=[1], control_values=[False])(theta, wires=[0])
            return qp.state()

        assert jnp.allclose(capture_result, circuit(theta))

    @pytest.mark.parametrize("theta", (jnp.pi, 0.1, 0.0))
    def test_ctrl_pcphase(self, backend, theta):
        """Test the integration for a PCPhase circuit with control."""
        if backend == "lightning.kokkos":
            pytest.xfail(reason="Controlled PCPhase not yet implemented on Kokkos.")

        device = qp.device(backend, wires=3)

        # Capture enabled

        qp.capture.enable()

        @qp.qnode(device)
        def circuit(theta):
            qp.ctrl(qp.PCPhase, control=[1], control_values=[False])(theta, 2, wires=[0])
            return qp.state()

        capture_result = qjit(circuit)(theta)

        # Capture disabled

        pass

        assert jnp.allclose(capture_result, circuit(theta))

    @pytest.mark.parametrize(
        "reset, op, expected",
        [(False, qp.I, 1), (False, qp.X, -1), (True, qp.I, 1), (True, qp.X, 1)],
    )
    def test_measure(self, backend, reset, op, expected):
        """Test the integration for a circuit with a mid-circuit measurement.

        We do not currently have full feature parity between PennyLane's `measure` and Catalyst's.
        Moreover, there is limited support for the various MCM methods with QJIT and with program
        capture enabled. Hence, we only test that a simple example with a deterministic outcome
        returns correct results.
        """
        device = qp.device(backend, wires=1)

        qp.capture.enable()

        @qjit
        @qp.qnode(device)
        def captured_circuit():
            op(wires=0)
            qp.measure(wires=0, reset=reset)
            return qp.expval(qp.Z(0))

        capture_result = captured_circuit()

        pass

        assert jnp.allclose(capture_result, expected)

    def test_measure_postselect(self, backend):
        """Test the integration for a circuit with a mid-circuit measurement using postselect.

        See note in test_measure() above for discussion on testing strategy.
        """
        device = qp.device(backend, wires=1)

        qp.capture.enable()

        @qjit
        @qp.qnode(device)
        def captured_circuit():
            qp.H(wires=0)
            qp.measure(wires=0, postselect=1)
            return qp.expval(qp.Z(0))

        capture_result = captured_circuit()

        pass

        expected_result = -1

        assert jnp.allclose(capture_result, expected_result)

    @pytest.mark.parametrize("pred", [False, 0.0, 0])
    def test_measure_as_condition(self, backend, pred):
        """Test the integration for a circuit with a mid-circuit measurement used as a conditional
        predicate.
        """
        device = qp.device(backend, wires=1)

        qp.capture.enable()

        @qjit(autograph=True)
        @qp.qnode(device)
        def captured_circuit():
            m = qp.measure(wires=0)
            if m == pred:
                qp.X(0)
            return qp.expval(qp.Z(0))

        capture_result = captured_circuit()

        pass

        assert jnp.allclose(capture_result, -1)

    @pytest.mark.parametrize("theta", (jnp.pi, 0.1, 0.0))
    def test_forloop(self, backend, theta):
        """Test the integration for a circuit with a for loop."""

        # Capture enabled

        qp.capture.enable()

        @qjit
        @qp.qnode(qp.device(backend, wires=4))
        def captured_circuit(x):

            @qp.for_loop(1, 4, 1)
            def loop(i):
                qp.CNOT(wires=[0, i])
                qp.RX(x, wires=i)

            loop()  # pylint: disable=no-value-for-parameter

            return qp.expval(qp.Z(2))

        capture_result = captured_circuit(theta)

        pass

        # Capture disabled

        @qp.qnode(qp.device(backend, wires=4))
        def circuit(x):

            for i in range(1, 4):
                qp.CNOT(wires=[0, i])
                qp.RX(x, wires=i)

            return qp.expval(qp.Z(2))

        assert jnp.allclose(capture_result, circuit(theta))

    def test_forloop_workflow(self, backend):
        """Test the integration for a circuit with a for loop primitive."""

        # Capture enabled

        qp.capture.enable()

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def captured_circuit(n, x):

            @qp.for_loop(1, n, 1)
            def loop_rx(_, x):
                qp.RX(x, wires=0)
                return jnp.sin(x)

            # apply the for loop
            loop_rx(x)  # pylint: disable=no-value-for-parameter

            return qp.expval(qp.Z(0))

        capture_result = captured_circuit(10, 0.3)

        pass

        # Capture disabled

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def circuit(n, x):

            @qp.for_loop(1, n, 1)
            def loop_rx(_, x):
                qp.RX(x, wires=0)
                return jnp.sin(x)

            # apply the for loop
            loop_rx(x)  # pylint: disable=no-value-for-parameter

            return qp.expval(qp.Z(0))

        assert jnp.allclose(circuit(10, 0.3), capture_result)

    def test_nested_loops(self, backend):
        """Test the integration for a circuit with a nested for loop primitive."""

        # Capture enabled

        qp.capture.enable()

        @qjit
        @qp.qnode(qp.device(backend, wires=4))
        def captured_circuit(n):
            # Input state: equal superposition
            @qp.for_loop(0, n, 1)
            def init(i):
                qp.Hadamard(wires=i)

            # QFT
            @qp.for_loop(0, n, 1)
            def qft(i):
                qp.Hadamard(wires=i)

                @qp.for_loop(i + 1, n, 1)
                def inner(j):
                    qp.ControlledPhaseShift(jnp.pi / 2 ** (n - j + 1), [i, j])

                inner()  # pylint: disable=no-value-for-parameter

            init()  # pylint: disable=no-value-for-parameter
            qft()  # pylint: disable=no-value-for-parameter

            # Expected output: |100...>
            return qp.state()

        capture_result = captured_circuit(4)

        pass

        # Capture disabled

        @qjit
        @qp.qnode(qp.device(backend, wires=4))
        def circuit(n):
            # Input state: equal superposition
            @qp.for_loop(0, n, 1)
            def init(i):
                qp.Hadamard(wires=i)

            # QFT
            @qp.for_loop(0, n, 1)
            def qft(i):
                qp.Hadamard(wires=i)

                @qp.for_loop(i + 1, n, 1)
                def inner(j):
                    qp.ControlledPhaseShift(jnp.pi / 2 ** (n - j + 1), [i, j])

                inner()  # pylint: disable=no-value-for-parameter

            init()  # pylint: disable=no-value-for-parameter
            qft()  # pylint: disable=no-value-for-parameter

            # Expected output: |100...>
            return qp.state()

        result = circuit(4)

        assert jnp.allclose(result, jnp.eye(2**4)[0])
        assert jnp.allclose(capture_result, result)

    def test_while_loop_workflow(self, backend):
        """Test the integration for a circuit with a while_loop primitive."""

        # Capture enabled

        qp.capture.enable()

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def capturted_circuit(x: float):

            def while_cond(i):
                return i < 10

            @qp.while_loop(while_cond)
            def loop_rx(a):
                # perform some work and update (some of) the arguments
                qp.RX(a, wires=0)
                return a + 1

            # apply the while loop
            loop_rx(x)

            return qp.expval(qp.Z(0))

        capture_result_10_iterations = capturted_circuit(0)
        capture_result_1_iteration = capturted_circuit(9)
        capture_result_0_iterations = capturted_circuit(11)

        pass

        # Capture disabled

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def circuit(x: float):

            def while_cond(i):
                return i < 10

            @qp.while_loop(while_cond)
            def loop_rx(a):
                # perform some work and update (some of) the arguments
                qp.RX(a, wires=0)
                return a + 1

            # apply the while loop
            loop_rx(x)

            return qp.expval(qp.Z(0))

        assert jnp.allclose(circuit(0), capture_result_10_iterations)
        assert jnp.allclose(circuit(9), capture_result_1_iteration)
        assert jnp.allclose(circuit(11), capture_result_0_iterations)

    def test_while_loop_workflow_closure(self, backend):
        """Test the integration for a circuit with a while_loop primitive using
        a closure variable."""

        # Capture enabled

        qp.capture.enable()

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def captured_circuit(x: float, step: float):

            def while_cond(i):
                return i < 10

            @qp.while_loop(while_cond)
            def loop_rx(a):
                # perform some work and update (some of) the arguments
                qp.RX(a, wires=0)
                return a + step

            # apply the while loop
            loop_rx(x)

            return qp.expval(qp.Z(0))

        capture_result = captured_circuit(0, 2)

        pass

        # Capture disabled

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def circuit(x: float, step: float):

            def while_cond(i):
                return i < 10

            @qp.while_loop(while_cond)
            def loop_rx(a):
                # perform some work and update (some of) the arguments
                qp.RX(a, wires=0)
                return a + step

            # apply the while loop
            loop_rx(x)

            return qp.expval(qp.Z(0))

        assert jnp.allclose(circuit(0, 2), capture_result)

    def test_while_loop_workflow_nested(self, backend):
        """Test the integration for a circuit with a nested while_loop primitive."""

        # Capture enabled

        qp.capture.enable()

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def captured_circuit(x: float, y: float):

            def while_cond(i):
                return i < 10

            @qp.while_loop(while_cond)
            def outer_loop(a):

                @qp.while_loop(while_cond)
                def inner_loop(b):
                    qp.RX(b, wires=0)
                    return b + 1

                # apply the inner loop
                inner_loop(y)
                qp.RX(a, wires=0)
                return a + 1

            # apply the outer loop
            outer_loop(x)

            return qp.expval(qp.Z(0))

        capture_result = captured_circuit(0, 0)

        pass

        # Capture disabled

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def circuit(x: float, y: float):

            def while_cond(i):
                return i < 10

            @qp.while_loop(while_cond)
            def outer_loop(a):

                @qp.while_loop(while_cond)
                def inner_loop(b):
                    qp.RX(b, wires=0)
                    return b + 1

                # apply the inner loop
                inner_loop(y)
                qp.RX(a, wires=0)
                return a + 1

            # apply the outer loop
            outer_loop(x)

            return qp.expval(qp.Z(0))

        assert jnp.allclose(circuit(0, 0), capture_result)

    def test_cond_workflow_if_else(self, backend):
        """Test the integration for a circuit with a cond primitive with true and false branches."""

        # Capture enabled

        qp.capture.enable()

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def captured_circuit(x: float):

            def ansatz_true():
                qp.RX(x, wires=0)
                qp.Hadamard(wires=0)

            def ansatz_false():
                qp.RY(x, wires=0)

            qp.cond(x > 1.4, ansatz_true, ansatz_false)()

            return qp.expval(qp.Z(0))

        capture_result = captured_circuit(0.1)

        pass

        # Capture disabled

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def circuit(x: float):

            def ansatz_true():
                qp.RX(x, wires=0)
                qp.Hadamard(wires=0)

            def ansatz_false():
                qp.RY(x, wires=0)

            qp.cond(x > 1.4, ansatz_true, ansatz_false)()

            return qp.expval(qp.Z(0))

        assert jnp.allclose(circuit(0.1), capture_result)

    def test_cond_workflow_if(self, backend):
        """Test the integration for a circuit with a cond primitive with a true branch only."""

        # Capture enabled

        qp.capture.enable()

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def captured_circuit(x: float):

            def ansatz_true():
                qp.RX(x, wires=0)
                qp.Hadamard(wires=0)

            qp.cond(x > 1.4, ansatz_true)()

            return qp.expval(qp.Z(0))

        capture_result = captured_circuit(1.5)

        pass

        # Capture disabled

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def circuit(x: float):

            def ansatz_true():
                qp.RX(x, wires=0)
                qp.Hadamard(wires=0)

            qp.cond(x > 1.4, ansatz_true)()

            return qp.expval(qp.Z(0))

        assert jnp.allclose(circuit(1.5), capture_result)

    def test_cond_workflow_with_custom_primitive(self, backend):
        """Test the integration for a circuit with a cond primitive containing a custom
        primitive."""

        # Capture enabled

        qp.capture.enable()

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def captured_circuit(x: float):

            def ansatz_true():
                qp.RX(x, wires=0)
                qp.Hadamard(wires=0)
                qp.GlobalPhase(jnp.pi / 4)  # Custom primitive

            def ansatz_false():
                qp.RY(x, wires=0)
                qp.GlobalPhase(jnp.pi / 2)  # Custom primitive

            qp.cond(x > 1.4, ansatz_true, ansatz_false)()

            return qp.expval(qp.Z(0))

        capture_result = captured_circuit(0.1)

        pass

        # Capture disabled

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def circuit(x: float):

            def ansatz_true():
                qp.RX(x, wires=0)
                qp.Hadamard(wires=0)
                qp.GlobalPhase(jnp.pi / 4)  # Custom primitive

            def ansatz_false():
                qp.RY(x, wires=0)
                qp.GlobalPhase(jnp.pi / 2)  # Custom primitive

            qp.cond(x > 1.4, ansatz_true, ansatz_false)()

            return qp.expval(qp.Z(0))

        assert jnp.allclose(circuit(0.1), capture_result)

    def test_cond_workflow_with_abstract_measurement(self, backend):
        """Test the integration for a circuit with a cond primitive containing an
        abstract measurement."""

        # Capture enabled

        qp.capture.enable()

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def captured_circuit(x: float):

            def ansatz_true():
                qp.RX(x, wires=0)
                qp.Hadamard(wires=0)
                qp.state()  # Abstract measurement

            def ansatz_false():
                qp.RY(x, wires=0)
                qp.state()  # Abstract measurement

            qp.cond(x > 1.4, ansatz_true, ansatz_false)()

            return qp.expval(qp.Z(0))

        capture_result = captured_circuit(0.1)

        pass

        # Capture disabled

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def circuit(x: float):

            def ansatz_true():
                qp.RX(x, wires=0)
                qp.Hadamard(wires=0)
                qp.state()  # Abstract measurement

            def ansatz_false():
                qp.RY(x, wires=0)
                qp.state()  # Abstract measurement

            qp.cond(x > 1.4, ansatz_true, ansatz_false)()

            return qp.expval(qp.Z(0))

        assert jnp.allclose(circuit(0.1), capture_result)

    def test_cond_workflow_with_simple_primitive(self, backend):
        """Test the integration for a circuit with a cond primitive containing an
        simple primitive."""

        # Capture enabled

        qp.capture.enable()

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def captured_circuit(x: float):

            def ansatz_true():
                qp.RX(x, wires=0)
                qp.Hadamard(wires=0)
                return x + 1  # simple primitive

            def ansatz_false():
                qp.RY(x, wires=0)
                return x + 1  # simple primitive

            qp.cond(x > 1.4, ansatz_true, ansatz_false)()

            return qp.expval(qp.Z(0))

        capture_result = captured_circuit(0.1)

        pass

        # Capture disabled

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def circuit(x: float):

            def ansatz_true():
                qp.RX(x, wires=0)
                qp.Hadamard(wires=0)
                return x + 1  # simple primitive

            def ansatz_false():
                qp.RY(x, wires=0)
                return x + 1  # simple primitive

            qp.cond(x > 1.4, ansatz_true, ansatz_false)()

            return qp.expval(qp.Z(0))

        assert jnp.allclose(circuit(0.1), capture_result)

    def test_cond_workflow_nested(self, backend):
        """Test the integration for a circuit with a nested cond primitive."""

        # Capture enabled

        qp.capture.enable()

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def captured_circuit(x: float, y: float):

            def ansatz_true():
                qp.RX(x, wires=0)
                qp.Hadamard(wires=0)

            def ansatz_false():

                def branch_true():
                    qp.RY(y, wires=0)

                def branch_false():
                    qp.RZ(y, wires=0)

                qp.cond(y > 1.4, branch_true, branch_false)()

            qp.cond(x > 1.4, ansatz_true, ansatz_false)()

            return qp.expval(qp.Z(0))

        capture_result = captured_circuit(0.1, 1.5)

        pass

        # Capture disabled

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def circuit(x: float, y: float):

            def ansatz_true():
                qp.RX(x, wires=0)
                qp.Hadamard(wires=0)

            def ansatz_false():

                def branch_true():
                    qp.RY(y, wires=0)

                def branch_false():
                    qp.RZ(y, wires=0)

                qp.cond(y > 1.4, branch_true, branch_false)()

            qp.cond(x > 1.4, ansatz_true, ansatz_false)()

            return qp.expval(qp.Z(0))

        assert jnp.allclose(circuit(0.1, 1.5), capture_result)

    def test_cond_workflow_operator(self, backend):
        """Test the integration for a circuit with a cond primitive returning
        an Operator."""

        # Capture enabled

        qp.capture.enable()

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def captured_circuit(x: float):

            qp.cond(x > 1.4, qp.RX, qp.RY)(x, wires=0)

            return qp.expval(qp.Z(0))

        capture_result = captured_circuit(0.1)

        pass

        # Capture disabled

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def circuit(x: float):

            qp.cond(x > 1.4, qp.RX, qp.RY)(x, wires=0)

            return qp.expval(qp.Z(0))

        assert jnp.allclose(circuit(0.1), capture_result)

    def test_pass_with_setup_input_options(self, backend):
        """Test the integration for a circuit with a pass that takes in options."""

        def my_pass_setup_inputs(my_option=None, my_other_option=None):
            return (), {"my_option": my_option, "my_other_option": my_other_option}

        my_pass = qp.transform(pass_name="my-pass", setup_inputs=my_pass_setup_inputs)

        @qjit(target="mlir", capture=True)
        @partial(my_pass, my_option="my_option_value", my_other_option=False)
        @qp.qnode(qp.device(backend, wires=1))
        def captured_circuit():
            return qp.expval(qp.PauliZ(0))

        capture_mlir = captured_circuit.mlir
        assert 'transform.apply_registered_pass "my-pass"' in capture_mlir
        assert (
            'with options = {"my-option" = "my_option_value", "my-other-option" = false}'
            in capture_mlir
        )

    def test_pass_with_options(self, backend):
        """Test the integration for a circuit with a pass that takes in options."""

        my_pass = qp.transform(pass_name="my-pass")

        @qjit(target="mlir", capture=True)
        @partial(my_pass, my_option="my_option_value", my_other_option=False)
        @qp.qnode(qp.device(backend, wires=1))
        def captured_circuit():
            return qp.expval(qp.PauliZ(0))

        capture_mlir = captured_circuit.mlir
        assert 'transform.apply_registered_pass "my-pass"' in capture_mlir
        assert (
            'with options = {"my-option" = "my_option_value", "my-other-option" = false}'
            in capture_mlir
        )

    def test_transform_cancel_inverses_workflow(self, backend):
        """Test the integration for a circuit with a 'cancel_inverses' transform."""

        # Capture enabled

        qp.capture.enable()

        @qjit
        @qp.transforms.cancel_inverses
        @qp.qnode(qp.device(backend, wires=1))
        def captured_circuit(x: float):
            qp.RX(x, wires=0)
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=0)
            return qp.expval(qp.PauliZ(0))

        capture_result = captured_circuit(0.1)
        assert 'transform.apply_registered_pass "cancel-inverses"' in captured_circuit.mlir

        pass

        # Capture disabled

        @qjit
        @qp.transforms.cancel_inverses
        @qp.qnode(qp.device(backend, wires=1))
        def circuit(x: float):
            qp.RX(x, wires=0)
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=0)
            return qp.expval(qp.PauliZ(0))

        assert jnp.allclose(circuit(0.1), capture_result)

    def test_transform_merge_rotations_workflow(self, backend):
        """Test the integration for a circuit with a 'merge_rotations' transform."""

        # Capture enabled

        qp.capture.enable()

        @qjit
        @qp.transforms.merge_rotations
        @qp.qnode(qp.device(backend, wires=1))
        def captured_circuit(x: float):
            qp.RX(x, wires=0)
            qp.RX(x, wires=0)
            qp.Hadamard(wires=0)
            return qp.expval(qp.PauliZ(0))

        capture_result = captured_circuit(0.1)
        assert 'transform.apply_registered_pass "merge-rotations"' in captured_circuit.mlir

        pass

        # Capture disabled

        @qjit
        @qp.transforms.merge_rotations
        @qp.qnode(qp.device(backend, wires=1))
        def circuit(x: float):
            qp.RX(x, wires=0)
            qp.RX(x, wires=0)
            qp.Hadamard(wires=0)
            return qp.expval(qp.PauliZ(0))

        assert jnp.allclose(circuit(0.1), capture_result)

    def test_chained_catalyst_transforms_workflow(self, backend):
        """Test the integration for a circuit with a combination of 'merge_rotations'
        and 'cancel_inverses' transforms."""

        # Capture enabled

        qp.capture.enable()

        @qp.qnode(qp.device(backend, wires=1))
        def captured_circuit(x: float):
            qp.RX(x, wires=0)
            qp.RX(x, wires=0)
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=0)
            return qp.expval(qp.PauliZ(0))

        captured_inverses_rotations = qjit(
            qp.transforms.cancel_inverses(qp.transforms.merge_rotations(captured_circuit))
        )
        captured_inverses_rotations_result = captured_inverses_rotations(0.1)
        assert has_catalyst_transforms(captured_inverses_rotations.mlir)

        captured_rotations_inverses = qjit(
            qp.transforms.merge_rotations(qp.transforms.cancel_inverses(captured_circuit)),
        )
        captured_rotations_inverses_result = captured_rotations_inverses(0.1)
        assert has_catalyst_transforms(captured_rotations_inverses.mlir)

        pass

        # Capture disabled

        @qp.qnode(qp.device(backend, wires=1))
        def circuit(x: float):
            qp.RX(x, wires=0)
            qp.RX(x, wires=0)
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=0)
            return qp.expval(qp.PauliZ(0))

        inverses_rotations_result = qjit(
            qp.transforms.cancel_inverses(qp.transforms.merge_rotations(circuit))
        )(0.1)
        rotations_inverses_result = qjit(
            qp.transforms.merge_rotations(qp.transforms.cancel_inverses(circuit))
        )(0.1)

        assert (
            inverses_rotations_result
            == rotations_inverses_result
            == captured_inverses_rotations_result
            == captured_rotations_inverses_result
        )

    def test_transform_unitary_to_rot_workflow(self, backend):
        """Test the integration for a circuit with a 'unitary_to_rot' transform."""

        U = qp.Rot(1.0, 2.0, 3.0, wires=0)

        # Capture enabled

        qp.capture.enable()

        @qjit
        @qp.transforms.unitary_to_rot
        @qp.qnode(qp.device(backend, wires=1))
        def captured_circuit(U: ShapedArray([2, 2], complex)):
            qp.QubitUnitary(U, 0)
            return qp.expval(qp.Z(0))

        capture_result = captured_circuit(U.matrix())
        assert is_unitary_rotated(captured_circuit.mlir)

        pass

        # Capture disabled

        @qjit
        @qp.transforms.unitary_to_rot
        @qp.qnode(qp.device(backend, wires=1))
        def circuit(U: ShapedArray([2, 2], complex)):
            qp.QubitUnitary(U, 0)
            return qp.expval(qp.Z(0))

        assert jnp.allclose(circuit(U.matrix()), capture_result)

    def test_mixed_transforms_workflow(self, backend):
        """Test the integration for a circuit with a combination of 'unitary_to_rot'
        and 'cancel_inverses' transforms."""

        U = qp.Rot(1.0, 2.0, 3.0, wires=0)

        # Capture enabled

        qp.capture.enable()

        @qp.qnode(qp.device(backend, wires=1))
        def captured_circuit(U: ShapedArray([2, 2], complex)):
            qp.QubitUnitary(U, 0)
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=0)
            return qp.expval(qp.PauliZ(0))

        # Case 1: During plxpr interpretation, first comes the PL transform
        # with Catalyst counterpart, second comes the PL transform without it

        captured_inverses_unitary = qjit(
            qp.transforms.cancel_inverses(qp.transforms.unitary_to_rot(captured_circuit)),
        )
        captured_inverses_unitary_result = captured_inverses_unitary(U.matrix())

        # Catalyst 'cancel_inverses' should have been scheduled as a pass
        # whereas PL 'unitary_to_rot' should have been expanded
        capture_mlir = captured_inverses_unitary.mlir
        assert 'transform.apply_registered_pass "cancel-inverses"' in capture_mlir
        assert is_unitary_rotated(capture_mlir)

        # Case 2: During plxpr interpretation, first comes the PL transform
        # without Catalyst counterpart, second comes the PL transform with it

        captured_unitary_inverses = qjit(
            qp.transforms.unitary_to_rot(qp.transforms.cancel_inverses(captured_circuit)),
        )
        captured_unitary_inverses_result = captured_unitary_inverses(U.matrix())

        # Both PL transforms should have been expaned and no Catalyst pass should have been
        # scheduled
        capture_mlir = captured_unitary_inverses.mlir
        assert 'transform.apply_registered_pass "cancel-inverses"' not in capture_mlir
        assert 'quantum.custom "Hadamard"' not in capture_mlir
        assert is_unitary_rotated(capture_mlir)

        pass

        # Capture disabled

        @qp.qnode(qp.device(backend, wires=1))
        def circuit(U: ShapedArray([2, 2], complex)):
            qp.QubitUnitary(U, 0)
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=0)
            return qp.expval(qp.PauliZ(0))

        inverses_unitary_result = qjit(
            qp.transforms.cancel_inverses(qp.transforms.unitary_to_rot(circuit))
        )(U.matrix())
        unitary_inverses_result = qjit(
            qp.transforms.unitary_to_rot(qp.transforms.cancel_inverses(circuit))
        )(U.matrix())

        assert (
            inverses_unitary_result
            == unitary_inverses_result
            == captured_inverses_unitary_result
            == captured_unitary_inverses_result
        )

    def test_transform_decompose_workflow(self, backend):
        """Test the integration for a circuit with a 'decompose' transform."""

        # Capture enabled

        qp.capture.enable()

        @qjit
        @partial(qp.transforms.decompose, gate_set=[qp.RX, qp.RY, qp.RZ])
        @qp.qnode(qp.device(backend, wires=2))
        def captured_circuit(x: float, y: float, z: float):
            qp.Rot(x, y, z, 0)
            return qp.expval(qp.PauliZ(0))

        capture_result = captured_circuit(1.5, 2.5, 3.5)
        assert is_rot_decomposed(captured_circuit.mlir)

        pass

        # Capture disabled

        @qjit
        @partial(qp.transforms.decompose, gate_set=[qp.RX, qp.RY, qp.RZ])
        @qp.qnode(qp.device(backend, wires=2))
        def circuit(x: float, y: float, z: float):
            qp.Rot(x, y, z, 0)
            return qp.expval(qp.PauliZ(0))

        assert jnp.allclose(circuit(1.5, 2.5, 3.5), capture_result)

    def test_transform_graph_decompose_workflow(self, backend):
        """Test the integration for a circuit with a 'decompose' graph transform."""

        # Capture enabled

        qp.capture.enable()
        qp.decomposition.enable_graph()

        @qjit
        @partial(qp.transforms.decompose, gate_set=[qp.RX, qp.RY, qp.RZ])
        @qp.qnode(qp.device(backend, wires=2))
        def captured_circuit(x: float, y: float, z: float):
            m = qp.measure(0)

            @qp.cond(m)
            def cond_fn():
                qp.Rot(x, y, z, 0)

            cond_fn()
            return qp.expval(qp.PauliZ(0))

        capture_result = captured_circuit(1.5, 2.5, 3.5)

        qp.decomposition.disable_graph()
        pass

        # Capture disabled
        @partial(qp.transforms.decompose, gate_set=[qp.RX, qp.RY, qp.RZ])
        @qp.qnode(qp.device(backend, wires=2))
        def circuit(x: float, y: float, z: float):
            m = catalyst.measure(0)

            @catalyst.cond(m)
            def cond_fn():
                qp.Rot(x, y, z, 0)

            cond_fn()
            return qp.expval(qp.PauliZ(0))

        # non-capture pathway is not actively developed and raises unnecessary warnings (wontfix)
        with pytest.warns(UserWarning, match="MidCircuitMeasure does not define a decomposition"):
            with pytest.warns(UserWarning, match="Cond does not define a decomposition"):
                non_capture_result = qjit(circuit)(1.5, 2.5, 3.5)

        assert jnp.allclose(non_capture_result, capture_result)

    def test_transform_single_qubit_fusion_workflow(self, backend):
        """Test the integration for a circuit with a 'single_qubit_fusion' transform."""

        # Capture enabled

        qp.capture.enable()

        @qjit
        @qp.transforms.single_qubit_fusion
        @qp.qnode(qp.device(backend, wires=1))
        def captured_circuit():
            qp.Hadamard(wires=0)
            qp.Rot(0.1, 0.2, 0.3, wires=0)
            qp.Rot(0.4, 0.5, 0.6, wires=0)
            qp.RZ(0.1, wires=0)
            qp.RZ(0.4, wires=0)
            return qp.expval(qp.PauliZ(0))

        capture_result = captured_circuit()

        assert is_single_qubit_fusion_applied(captured_circuit.mlir)

        pass

        # Capture disabled

        @qjit
        @qp.transforms.single_qubit_fusion
        @qp.qnode(qp.device(backend, wires=1))
        def circuit():
            qp.Hadamard(wires=0)
            qp.Rot(0.1, 0.2, 0.3, wires=0)
            qp.Rot(0.4, 0.5, 0.6, wires=0)
            qp.RZ(0.1, wires=0)
            qp.RZ(0.4, wires=0)
            return qp.expval(qp.PauliZ(0))

        assert jnp.allclose(circuit(), capture_result)

    def test_transform_commute_controlled_workflow(self, backend):
        """Test the integration for a circuit with a 'commute_controlled' transform."""

        # Capture enabled

        qp.capture.enable()

        @qjit
        @partial(qp.transforms.commute_controlled, direction="left")
        @qp.qnode(qp.device(backend, wires=3))
        def captured_circuit():
            qp.CNOT(wires=[0, 2])
            qp.PauliX(wires=2)
            qp.RX(0.2, wires=2)
            qp.Toffoli(wires=[0, 1, 2])
            qp.CRX(0.1, wires=[0, 1])
            qp.PauliX(wires=1)
            return qp.expval(qp.PauliZ(0))

        capture_result = captured_circuit()

        capture_mlir = captured_circuit.mlir
        assert is_controlled_pushed_back(
            capture_mlir, 'quantum.custom "RX"', 'quantum.custom "CNOT"'
        )
        assert is_controlled_pushed_back(
            capture_mlir, 'quantum.custom "PauliX"', 'quantum.custom "CRX"'
        )

        pass

        # Capture disabled

        @qjit
        @partial(qp.transforms.commute_controlled, direction="left")
        @qp.qnode(qp.device(backend, wires=3))
        def circuit():
            qp.CNOT(wires=[0, 2])
            qp.PauliX(wires=2)
            qp.RX(0.2, wires=2)
            qp.Toffoli(wires=[0, 1, 2])
            qp.CRX(0.1, wires=[0, 1])
            qp.PauliX(wires=1)
            return qp.expval(qp.PauliZ(0))

        assert jnp.allclose(circuit(), capture_result)

    def test_transform_merge_amplitude_embedding_workflow(self, backend):
        """Test the integration for a circuit with a 'merge_amplitude_embedding' transform."""

        # Capture enabled

        qp.capture.enable()

        @qjit
        @qp.transforms.merge_amplitude_embedding
        @qp.qnode(qp.device(backend, wires=2))
        def captured_circuit():
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=0)
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=1)
            return qp.expval(qp.PauliZ(0))

        capture_result = captured_circuit()
        assert is_amplitude_embedding_merged_and_decomposed(captured_circuit.mlir)

        pass

        # Capture disabled

        @qjit
        @qp.transforms.merge_amplitude_embedding
        @qp.qnode(qp.device(backend, wires=2))
        def circuit():
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=0)
            qp.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=1)
            return qp.expval(qp.PauliZ(0))

        assert jnp.allclose(circuit(), capture_result)

    def test_shots_usage(self, backend):
        """Test the integration for a circuit using shots explicitly."""

        # Capture enabled

        qp.capture.enable()

        @qjit
        @qp.set_shots(10)
        @qp.qnode(qp.device(backend, wires=2))
        def captured_circuit():
            @qp.for_loop(0, 2, 1)
            def loop_0(i):
                qp.RX(0, wires=i)

            loop_0()  # pylint: disable=no-value-for-parameter

            qp.RX(0, wires=0)
            return qp.sample()

        capture_result = captured_circuit()
        assert "shots(%" in captured_circuit.mlir

        pass

        @qjit
        @qp.set_shots(10)
        @qp.qnode(qp.device(backend, wires=2))
        def circuit():
            @qp.for_loop(0, 2, 1)
            def loop_0(i):
                qp.RX(0, wires=i)

            loop_0()  # pylint: disable=no-value-for-parameter

            qp.RX(0, wires=0)
            return qp.sample()

        assert jnp.allclose(circuit(), capture_result)

    def test_static_variable_qnode(self, backend):
        """Test the integration for a circuit with a static variable."""

        qp.capture.enable()

        # Basic test
        @qjit(static_argnums=(0,))
        @qp.qnode(qp.device(backend, wires=1))
        def captured_circuit_1(x, y):
            qp.RX(x, wires=0)
            qp.RY(y, wires=0)
            return qp.expval(qp.PauliZ(0))

        result_1 = captured_circuit_1(1.5, 2.0)
        captured_circuit_1_mlir = captured_circuit_1.mlir
        assert "%cst = arith.constant 1.5" in captured_circuit_1_mlir
        assert 'quantum.custom "RX"(%cst)' in captured_circuit_1_mlir
        assert "%cst = arith.constant 2.0" not in captured_circuit_1_mlir

        # Test that qjit static_argnums takes precedence over the one on the qnode
        @qjit(static_argnums=1)
        @qp.qnode(qp.device(backend, wires=1), static_argnums=0)  # should be ignored
        def captured_circuit_2(x, y):
            qp.RX(x, wires=0)
            qp.RY(y, wires=0)
            return qp.expval(qp.PauliZ(0))

        result_2 = captured_circuit_2(1.5, 2.0)
        captured_circuit_2_mlir = captured_circuit_2.mlir
        assert "%cst = arith.constant 2.0" in captured_circuit_2_mlir
        assert 'quantum.custom "RY"(%cst)' in captured_circuit_2_mlir
        assert "%cst = arith.constant 1.5" not in captured_circuit_2_mlir

        assert jnp.allclose(result_1, result_2)

        # Test under a non qnode workflow function
        @qjit(static_argnums=(0,))
        def workflow(x, y):
            @qp.qnode(qp.device(backend, wires=1))
            def c():
                qp.RX(x, wires=0)
                qp.RY(y, wires=0)
                return qp.expval(qp.PauliZ(0))

            return c()

        _ = workflow(1.5, 2.0)
        captured_circuit_3_mlir = workflow.mlir
        assert "%cst = arith.constant 1.5" in captured_circuit_3_mlir
        assert 'quantum.custom "RX"(%cst)' in captured_circuit_3_mlir

        pass


class TestControlFlow:
    """Integration tests for control flow."""

    @pytest.mark.parametrize("reverse", (True, False))
    def test_for_loop_outside_qnode(self, reverse):
        """Test that a for loop outside qnode can be executed."""

        qp.capture.enable()

        if reverse:
            start, stop, step = 6, 0, -2  # 6, 4, 2
        else:
            start, stop, step = 2, 7, 2  # 2, 4, 6

        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def c(x):
            qp.RX(x, 0)
            return qp.expval(qp.Z(0))

        @qp.qjit
        def f(i0):
            @qp.for_loop(start, stop, step)
            def g(i, x):
                return c(i) + x

            return g(i0)  # pylint: disable=no-value-for-parameter

        out = f(3.0)
        assert qp.math.allclose(out, 3 + jnp.cos(2) + jnp.cos(4) + jnp.cos(6))

    def test_while_loop(self):
        """Test that a outside a qnode can be executed."""
        qp.capture.enable()

        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit(x):
            qp.RX(x, 0)
            return qp.expval(qp.Z(0))

        @qp.qjit
        def f(x):

            const = jnp.array([0, 1, 2])

            @qp.while_loop(lambda i, y: i < jnp.sum(const))
            def g(i, y):
                return i + 1, y + circuit(i)

            return g(0, x)

        ind, res = f(1.0)
        assert qp.math.allclose(ind, 3)
        expected = 1.0 + jnp.cos(0) + jnp.cos(1) + jnp.cos(2)
        assert qp.math.allclose(res, expected)

    # pylint: disable=unused-argument
    def test_for_loop_consts(self):
        """This tests for kinda a weird edge case bug where the consts where getting
        reordered when translating the inner jaxpr."""

        qp.capture.enable()

        @qp.qjit
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def circuit(x, n):
            @qp.for_loop(3)
            def outer(i):

                @qp.for_loop(n)
                def inner(j):
                    qp.RY(x, wires=j)

                inner()  # pylint: disable=no-value-for-parameter

            outer()  # pylint: disable=no-value-for-parameter

            # Expected output: |100...>
            return [qp.expval(qp.PauliZ(i)) for i in range(3)]

        res1, res2, res3 = circuit(0.2, 2)

        assert qp.math.allclose(res1, jnp.cos(0.2 * 3))
        assert qp.math.allclose(res2, jnp.cos(0.2 * 3))
        assert qp.math.allclose(res3, 1)

    # pylint: disable=unused-argument
    def test_for_loop_consts_outside_qnode(self):
        """Similar test as above for weird edge case, but not using a qnode."""

        qp.capture.enable()

        @qp.qjit
        def f(x, n):
            @qp.for_loop(3)
            def outer(i, a):

                @qp.for_loop(n)
                def inner(j, b):
                    return b + x

                return inner(a)  # pylint: disable=no-value-for-parameter

            return outer(0.0)  # pylint: disable=no-value-for-parameter

        res = f(0.2, 2)
        assert qp.math.allclose(res, 0.2 * 2 * 3)


def test_adjoint_transform_integration():
    """Test that adjoint transforms can be used with capture enabled."""

    qp.capture.enable()

    def f(x):
        qp.IsingXX(2 * x, wires=(0, 1))
        qp.H(0)

    @qp.qjit
    @qp.qnode(qp.device("lightning.qubit", wires=3))
    def c(x):
        qp.adjoint(f)(x)
        return qp.expval(qp.Z(1))

    x = jnp.array(0.7)
    res = c(x)
    expected = jnp.cos(-2 * x)
    assert qp.math.allclose(res, expected)


@pytest.mark.parametrize("separate_funcs", (True, False))
def test_ctrl_transform_integration(separate_funcs):
    """Test that the ctrl transform can be applied."""

    qp.capture.enable()

    def f(x, y):
        qp.RY(3 * y, wires=3)
        qp.RX(2 * x, wires=3)

    @qp.qjit
    @qp.qnode(qp.device("lightning.qubit", wires=4))
    def c(x, y):
        qp.X(1)
        if separate_funcs:
            qp.ctrl(qp.ctrl(f, 0, [False]), 1, [True])(x, y)
        else:
            qp.ctrl(f, (0, 1), [False, True])(x, y)
        return qp.expval(qp.Z(3))

    x = jnp.array(0.5)
    y = jnp.array(0.9)
    res = c(x, y)
    expected = jnp.cos(2 * x) * jnp.cos(3 * y)
    assert qp.math.allclose(res, expected)


def test_different_static_argnums():
    """Test that the same qnode can be called different times with different static argnums."""

    qp.capture.enable()

    @qp.qnode(qp.device("lightning.qubit", wires=1), static_argnums=1)
    def c(x, pauli):
        if pauli == "X":
            qp.RX(x, 0)
        elif pauli == "Y":
            qp.RY(x, 0)
        else:
            qp.RZ(x, 0)
        return qp.state()

    @qp.qjit
    def w(x):
        return c(x, "X"), c(x, "Y"), c(x, "Z")

    resx, resy, resz = w(0.5)

    a = jnp.cos(0.5 / 2)
    b = jnp.sin(0.5 / 2)
    assert qp.math.allclose(resx, jnp.array([a, -b * 1j]))
    assert qp.math.allclose(resy, jnp.array([a, b]))
    assert qp.math.allclose(resz, jnp.array([a - b * 1j, 0]))
