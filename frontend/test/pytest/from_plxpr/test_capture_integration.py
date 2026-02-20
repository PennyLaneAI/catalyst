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
import pennylane as qml
import pytest
from jax.core import ShapedArray

import catalyst
from catalyst import qjit
from catalyst.from_plxpr import register_transform

pytestmark = pytest.mark.usefixtures("disable_capture")


def circuit_aot_builder(dev):
    """Test AOT builder."""

    qml.capture.enable()

    @catalyst.qjit
    @qml.qnode(device=dev)
    def catalyst_circuit_aot(x: float):
        qml.Hadamard(wires=0)
        qml.RZ(x, wires=0)
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[1, 0])
        qml.Hadamard(wires=1)
        return qml.expval(qml.PauliY(wires=0))

    qml.capture.disable()

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
        dev = qml.device(backend, wires=2)

        @qml.qnode(device=dev)
        def circuit(x):
            qml.Hadamard(wires=0)
            qml.RZ(x, wires=0)
            qml.RZ(x, wires=0)
            qml.CNOT(wires=[1, 0])
            qml.Hadamard(wires=1)
            qml.PCPhase(x, 2, wires=[0])
            return qml.expval(qml.PauliY(wires=0))

        captured_result = circuit_aot_builder(dev)(theta)
        result = circuit(theta)
        assert jnp.allclose(captured_result, result), [captured_result, result]

    @pytest.mark.parametrize("capture", (True, False))
    @pytest.mark.parametrize("theta", (jnp.pi, 0.1, 0.0))
    def test_simple_circuit(self, backend, theta, capture):
        """Test the integration for a simple circuit."""

        dev = qml.device(backend, wires=2)

        qml.capture.enable()

        @catalyst.qjit
        @qml.qnode(device=dev)
        def captured_circuit(x):
            qml.Hadamard(wires=0)
            qml.RZ(x, wires=0)
            qml.RZ(x, wires=0)
            qml.CNOT(wires=[1, 0])
            qml.Hadamard(wires=1)
            return qml.expval(qml.PauliY(wires=0))

        capture_result = captured_circuit(theta)

        qml.capture.disable()

        if capture:
            qml.capture.enable()

        @qml.qnode(device=dev)
        def circuit(x):
            qml.Hadamard(wires=0)
            qml.RZ(x, wires=0)
            qml.RZ(x, wires=0)
            qml.CNOT(wires=[1, 0])
            qml.Hadamard(wires=1)
            return qml.expval(qml.PauliY(wires=0))

        if capture:
            assert qml.capture.enabled()
        else:
            assert not qml.capture.enabled()

        qml.capture.disable()

        assert jnp.allclose(capture_result, circuit(theta))

    @pytest.mark.parametrize("theta", (jnp.pi, 0.1, 0.0))
    def test_simple_workflow(self, backend, theta):
        """Test the integration for a simple workflow."""
        dev = qml.device(backend, wires=2)

        # Capture enabled

        qml.capture.enable()

        @qjit
        @qml.qnode(device=dev)
        def captured_circuit(x):
            qml.Hadamard(wires=0)
            qml.RZ(x, wires=0)
            qml.RZ(x, wires=0)
            qml.CNOT(wires=[1, 0])
            qml.Hadamard(wires=1)
            return qml.expval(qml.PauliY(wires=0))

        capture_result = captured_circuit(theta**2)

        qml.capture.disable()

        # Capture disabled

        @qjit
        @qml.qnode(device=dev)
        def circuit(x):
            qml.Hadamard(wires=0)
            qml.RZ(x, wires=0)
            qml.RZ(x, wires=0)
            qml.CNOT(wires=[1, 0])
            qml.Hadamard(wires=1)
            return qml.expval(qml.PauliY(wires=0))

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
        dev = qml.device(backend, wires=n_wires)

        # Capture enabled

        qml.capture.enable()

        @qjit
        @qml.qnode(dev)
        def captured_circuit(_basis_state):
            qml.BasisState(_basis_state, wires=list(range(n_wires)))
            return qml.state()

        capture_result = captured_circuit(basis_state)

        qml.capture.disable()

        # Capture disabled

        @qjit
        @qml.qnode(dev)
        def circuit(_basis_state):
            qml.BasisState(_basis_state, wires=list(range(n_wires)))
            return qml.state()

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
        dev = qml.device(backend, wires=n_wires)

        # Capture enabled

        qml.capture.enable()

        @qjit
        @qml.qnode(dev)
        def captured_circuit(init_state):
            qml.StatePrep(init_state, wires=list(range(n_wires)))
            return qml.state()

        capture_result = captured_circuit(init_state)

        qml.capture.disable()

        # Capture disabled

        @qjit
        @qml.qnode(dev)
        def circuit(init_state):
            qml.StatePrep(init_state, wires=list(range(n_wires)))
            return qml.state()

        assert jnp.allclose(capture_result, circuit(init_state))

    @pytest.mark.parametrize("theta, val", [(jnp.pi, 0), (-100.0, 1)])
    def test_adjoint(self, backend, theta, val):
        """Test the integration for a circuit with adjoint."""
        device = qml.device(backend, wires=2)

        # Capture enabled

        qml.capture.enable()

        @qjit
        @qml.qnode(device)
        def captured_circuit(theta, val):
            qml.adjoint(qml.RY)(jnp.pi, val)
            qml.adjoint(qml.RZ)(theta, wires=val)
            return qml.state()

        capture_result = captured_circuit(theta, val)

        qml.capture.disable()

        # Capture disabled

        @qml.qnode(device)
        def circuit(theta, val):
            qml.adjoint(qml.RY)(jnp.pi, val)
            qml.adjoint(qml.RZ)(theta, wires=val)
            return qml.state()

        assert jnp.allclose(capture_result, circuit(theta, val))

    @pytest.mark.parametrize("theta", (jnp.pi, 0.1, 0.0))
    def test_ctrl(self, backend, theta):
        """Test the integration for a circuit with control."""

        device = qml.device(backend, wires=3)

        # Capture enabled

        qml.capture.enable()

        @qjit
        @qml.qnode(device)
        def captured_circuit(theta):
            qml.ctrl(qml.RX(theta, wires=0), control=[1], control_values=[False])
            qml.ctrl(qml.RX, control=[1], control_values=[False])(theta, wires=[0])
            return qml.state()

        capture_result = captured_circuit(theta)

        qml.capture.disable()

        # Capture disabled

        @qml.qnode(device)
        def circuit(theta):
            qml.ctrl(qml.RX(theta, wires=0), control=[1], control_values=[False])
            qml.ctrl(qml.RX, control=[1], control_values=[False])(theta, wires=[0])
            return qml.state()

        assert jnp.allclose(capture_result, circuit(theta))

    @pytest.mark.parametrize("theta", (jnp.pi, 0.1, 0.0))
    def test_ctrl_pcphase(self, backend, theta):
        """Test the integration for a PCPhase circuit with control."""
        if backend == "lightning.kokkos":
            pytest.xfail(reason="PCPhase not yet implemented on Kokkos.")

        device = qml.device(backend, wires=3)

        # Capture enabled

        qml.capture.enable()

        @qml.qnode(device)
        def circuit(theta):
            qml.ctrl(qml.PCPhase, control=[1], control_values=[False])(theta, 2, wires=[0])
            return qml.state()

        capture_result = qjit(circuit)(theta)

        # Capture disabled

        qml.capture.disable()

        assert jnp.allclose(capture_result, circuit(theta))

    @pytest.mark.parametrize(
        "reset, op, expected",
        [(False, qml.I, 1), (False, qml.X, -1), (True, qml.I, 1), (True, qml.X, 1)],
    )
    def test_measure(self, backend, reset, op, expected):
        """Test the integration for a circuit with a mid-circuit measurement.

        We do not currently have full feature parity between PennyLane's `measure` and Catalyst's.
        Moreover, there is limited support for the various MCM methods with QJIT and with program
        capture enabled. Hence, we only test that a simple example with a deterministic outcome
        returns correct results.
        """
        device = qml.device(backend, wires=1)

        qml.capture.enable()

        @qjit
        @qml.qnode(device)
        def captured_circuit():
            op(wires=0)
            qml.measure(wires=0, reset=reset)
            return qml.expval(qml.Z(0))

        capture_result = captured_circuit()

        qml.capture.disable()

        assert jnp.allclose(capture_result, expected)

    def test_measure_postselect(self, backend):
        """Test the integration for a circuit with a mid-circuit measurement using postselect.

        See note in test_measure() above for discussion on testing strategy.
        """
        device = qml.device(backend, wires=1)

        qml.capture.enable()

        @qjit
        @qml.qnode(device)
        def captured_circuit():
            qml.H(wires=0)
            qml.measure(wires=0, postselect=1)
            return qml.expval(qml.Z(0))

        capture_result = captured_circuit()

        qml.capture.disable()

        expected_result = -1

        assert jnp.allclose(capture_result, expected_result)

    @pytest.mark.parametrize("theta", (jnp.pi, 0.1, 0.0))
    def test_forloop(self, backend, theta):
        """Test the integration for a circuit with a for loop."""

        # Capture enabled

        qml.capture.enable()

        @qjit
        @qml.qnode(qml.device(backend, wires=4))
        def captured_circuit(x):

            @qml.for_loop(1, 4, 1)
            def loop(i):
                qml.CNOT(wires=[0, i])
                qml.RX(x, wires=i)

            loop()  # pylint: disable=no-value-for-parameter

            return qml.expval(qml.Z(2))

        capture_result = captured_circuit(theta)

        qml.capture.disable()

        # Capture disabled

        @qml.qnode(qml.device(backend, wires=4))
        def circuit(x):

            for i in range(1, 4):
                qml.CNOT(wires=[0, i])
                qml.RX(x, wires=i)

            return qml.expval(qml.Z(2))

        assert jnp.allclose(capture_result, circuit(theta))

    def test_forloop_workflow(self, backend):
        """Test the integration for a circuit with a for loop primitive."""

        # Capture enabled

        qml.capture.enable()

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def captured_circuit(n, x):

            @qml.for_loop(1, n, 1)
            def loop_rx(_, x):
                qml.RX(x, wires=0)
                return jnp.sin(x)

            # apply the for loop
            loop_rx(x)  # pylint: disable=no-value-for-parameter

            return qml.expval(qml.Z(0))

        capture_result = captured_circuit(10, 0.3)

        qml.capture.disable()

        # Capture disabled

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(n, x):

            @qml.for_loop(1, n, 1)
            def loop_rx(_, x):
                qml.RX(x, wires=0)
                return jnp.sin(x)

            # apply the for loop
            loop_rx(x)  # pylint: disable=no-value-for-parameter

            return qml.expval(qml.Z(0))

        assert jnp.allclose(circuit(10, 0.3), capture_result)

    def test_nested_loops(self, backend):
        """Test the integration for a circuit with a nested for loop primitive."""

        # Capture enabled

        qml.capture.enable()

        @qjit
        @qml.qnode(qml.device(backend, wires=4))
        def captured_circuit(n):
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

                inner()  # pylint: disable=no-value-for-parameter

            init()  # pylint: disable=no-value-for-parameter
            qft()  # pylint: disable=no-value-for-parameter

            # Expected output: |100...>
            return qml.state()

        capture_result = captured_circuit(4)

        qml.capture.disable()

        # Capture disabled

        @qjit
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

                inner()  # pylint: disable=no-value-for-parameter

            init()  # pylint: disable=no-value-for-parameter
            qft()  # pylint: disable=no-value-for-parameter

            # Expected output: |100...>
            return qml.state()

        result = circuit(4)

        assert jnp.allclose(result, jnp.eye(2**4)[0])
        assert jnp.allclose(capture_result, result)

    def test_while_loop_workflow(self, backend):
        """Test the integration for a circuit with a while_loop primitive."""

        # Capture enabled

        qml.capture.enable()

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def capturted_circuit(x: float):

            def while_cond(i):
                return i < 10

            @qml.while_loop(while_cond)
            def loop_rx(a):
                # perform some work and update (some of) the arguments
                qml.RX(a, wires=0)
                return a + 1

            # apply the while loop
            loop_rx(x)

            return qml.expval(qml.Z(0))

        capture_result_10_iterations = capturted_circuit(0)
        capture_result_1_iteration = capturted_circuit(9)
        capture_result_0_iterations = capturted_circuit(11)

        qml.capture.disable()

        # Capture disabled

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float):

            def while_cond(i):
                return i < 10

            @qml.while_loop(while_cond)
            def loop_rx(a):
                # perform some work and update (some of) the arguments
                qml.RX(a, wires=0)
                return a + 1

            # apply the while loop
            loop_rx(x)

            return qml.expval(qml.Z(0))

        assert jnp.allclose(circuit(0), capture_result_10_iterations)
        assert jnp.allclose(circuit(9), capture_result_1_iteration)
        assert jnp.allclose(circuit(11), capture_result_0_iterations)

    def test_while_loop_workflow_closure(self, backend):
        """Test the integration for a circuit with a while_loop primitive using
        a closure variable."""

        # Capture enabled

        qml.capture.enable()

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def captured_circuit(x: float, step: float):

            def while_cond(i):
                return i < 10

            @qml.while_loop(while_cond)
            def loop_rx(a):
                # perform some work and update (some of) the arguments
                qml.RX(a, wires=0)
                return a + step

            # apply the while loop
            loop_rx(x)

            return qml.expval(qml.Z(0))

        capture_result = captured_circuit(0, 2)

        qml.capture.disable()

        # Capture disabled

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float, step: float):

            def while_cond(i):
                return i < 10

            @qml.while_loop(while_cond)
            def loop_rx(a):
                # perform some work and update (some of) the arguments
                qml.RX(a, wires=0)
                return a + step

            # apply the while loop
            loop_rx(x)

            return qml.expval(qml.Z(0))

        assert jnp.allclose(circuit(0, 2), capture_result)

    def test_while_loop_workflow_nested(self, backend):
        """Test the integration for a circuit with a nested while_loop primitive."""

        # Capture enabled

        qml.capture.enable()

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def captured_circuit(x: float, y: float):

            def while_cond(i):
                return i < 10

            @qml.while_loop(while_cond)
            def outer_loop(a):

                @qml.while_loop(while_cond)
                def inner_loop(b):
                    qml.RX(b, wires=0)
                    return b + 1

                # apply the inner loop
                inner_loop(y)
                qml.RX(a, wires=0)
                return a + 1

            # apply the outer loop
            outer_loop(x)

            return qml.expval(qml.Z(0))

        capture_result = captured_circuit(0, 0)

        qml.capture.disable()

        # Capture disabled

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float, y: float):

            def while_cond(i):
                return i < 10

            @qml.while_loop(while_cond)
            def outer_loop(a):

                @qml.while_loop(while_cond)
                def inner_loop(b):
                    qml.RX(b, wires=0)
                    return b + 1

                # apply the inner loop
                inner_loop(y)
                qml.RX(a, wires=0)
                return a + 1

            # apply the outer loop
            outer_loop(x)

            return qml.expval(qml.Z(0))

        assert jnp.allclose(circuit(0, 0), capture_result)

    def test_cond_workflow_if_else(self, backend):
        """Test the integration for a circuit with a cond primitive with true and false branches."""

        # Capture enabled

        qml.capture.enable()

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def captured_circuit(x: float):

            def ansatz_true():
                qml.RX(x, wires=0)
                qml.Hadamard(wires=0)

            def ansatz_false():
                qml.RY(x, wires=0)

            qml.cond(x > 1.4, ansatz_true, ansatz_false)()

            return qml.expval(qml.Z(0))

        capture_result = captured_circuit(0.1)

        qml.capture.disable()

        # Capture disabled

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float):

            def ansatz_true():
                qml.RX(x, wires=0)
                qml.Hadamard(wires=0)

            def ansatz_false():
                qml.RY(x, wires=0)

            qml.cond(x > 1.4, ansatz_true, ansatz_false)()

            return qml.expval(qml.Z(0))

        assert jnp.allclose(circuit(0.1), capture_result)

    def test_cond_workflow_if(self, backend):
        """Test the integration for a circuit with a cond primitive with a true branch only."""

        # Capture enabled

        qml.capture.enable()

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def captured_circuit(x: float):

            def ansatz_true():
                qml.RX(x, wires=0)
                qml.Hadamard(wires=0)

            qml.cond(x > 1.4, ansatz_true)()

            return qml.expval(qml.Z(0))

        capture_result = captured_circuit(1.5)

        qml.capture.disable()

        # Capture disabled

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float):

            def ansatz_true():
                qml.RX(x, wires=0)
                qml.Hadamard(wires=0)

            qml.cond(x > 1.4, ansatz_true)()

            return qml.expval(qml.Z(0))

        assert jnp.allclose(circuit(1.5), capture_result)

    def test_cond_workflow_with_custom_primitive(self, backend):
        """Test the integration for a circuit with a cond primitive containing a custom
        primitive."""

        # Capture enabled

        qml.capture.enable()

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def captured_circuit(x: float):

            def ansatz_true():
                qml.RX(x, wires=0)
                qml.Hadamard(wires=0)
                qml.GlobalPhase(jnp.pi / 4)  # Custom primitive

            def ansatz_false():
                qml.RY(x, wires=0)
                qml.GlobalPhase(jnp.pi / 2)  # Custom primitive

            qml.cond(x > 1.4, ansatz_true, ansatz_false)()

            return qml.expval(qml.Z(0))

        capture_result = captured_circuit(0.1)

        qml.capture.disable()

        # Capture disabled

        @qjit
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

        assert jnp.allclose(circuit(0.1), capture_result)

    def test_cond_workflow_with_abstract_measurement(self, backend):
        """Test the integration for a circuit with a cond primitive containing an
        abstract measurement."""

        # Capture enabled

        qml.capture.enable()

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def captured_circuit(x: float):

            def ansatz_true():
                qml.RX(x, wires=0)
                qml.Hadamard(wires=0)
                qml.state()  # Abstract measurement

            def ansatz_false():
                qml.RY(x, wires=0)
                qml.state()  # Abstract measurement

            qml.cond(x > 1.4, ansatz_true, ansatz_false)()

            return qml.expval(qml.Z(0))

        capture_result = captured_circuit(0.1)

        qml.capture.disable()

        # Capture disabled

        @qjit
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

        assert jnp.allclose(circuit(0.1), capture_result)

    def test_cond_workflow_with_simple_primitive(self, backend):
        """Test the integration for a circuit with a cond primitive containing an
        simple primitive."""

        # Capture enabled

        qml.capture.enable()

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def captured_circuit(x: float):

            def ansatz_true():
                qml.RX(x, wires=0)
                qml.Hadamard(wires=0)
                return x + 1  # simple primitive

            def ansatz_false():
                qml.RY(x, wires=0)
                return x + 1  # simple primitive

            qml.cond(x > 1.4, ansatz_true, ansatz_false)()

            return qml.expval(qml.Z(0))

        capture_result = captured_circuit(0.1)

        qml.capture.disable()

        # Capture disabled

        @qjit
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

        assert jnp.allclose(circuit(0.1), capture_result)

    def test_cond_workflow_nested(self, backend):
        """Test the integration for a circuit with a nested cond primitive."""

        # Capture enabled

        qml.capture.enable()

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def captured_circuit(x: float, y: float):

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

        capture_result = captured_circuit(0.1, 1.5)

        qml.capture.disable()

        # Capture disabled

        @qjit
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

        assert jnp.allclose(circuit(0.1, 1.5), capture_result)

    def test_cond_workflow_operator(self, backend):
        """Test the integration for a circuit with a cond primitive returning
        an Operator."""

        # Capture enabled

        qml.capture.enable()

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def captured_circuit(x: float):

            qml.cond(x > 1.4, qml.RX, qml.RY)(x, wires=0)

            return qml.expval(qml.Z(0))

        capture_result = captured_circuit(0.1)

        qml.capture.disable()

        # Capture disabled

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float):

            qml.cond(x > 1.4, qml.RX, qml.RY)(x, wires=0)

            return qml.expval(qml.Z(0))

        assert jnp.allclose(circuit(0.1), capture_result)

    @pytest.mark.usefixtures("use_capture")
    def test_pass_with_options_patch(self, backend):
        """Test the integration for a circuit with a pass that takes in options."""

        @qml.transform
        def my_pass(_tape, my_option=None, my_other_option=None):  # pylint: disable=unused-argument
            """A dummy qml.transform."""
            return

        register_transform(my_pass, "my-pass", False)

        @qjit(target="mlir")
        @partial(my_pass, my_option="my_option_value", my_other_option=False)
        @qml.qnode(qml.device(backend, wires=1))
        def captured_circuit():
            return qml.expval(qml.PauliZ(0))

        capture_mlir = captured_circuit.mlir
        assert 'transform.apply_registered_pass "my-pass"' in capture_mlir
        assert (
            'with options = {"my-option" = "my_option_value", "my-other-option" = false}'
            in capture_mlir
        )

    @pytest.mark.usefixtures("use_capture")
    def test_pass_with_options(self, backend):
        """Test the integration for a circuit with a pass that takes in options."""

        my_pass = qml.transform(pass_name="my-pass")

        @qjit(target="mlir")
        @partial(my_pass, my_option="my_option_value", my_other_option=False)
        @qml.qnode(qml.device(backend, wires=1))
        def captured_circuit():
            return qml.expval(qml.PauliZ(0))

        capture_mlir = captured_circuit.mlir
        assert 'transform.apply_registered_pass "my-pass"' in capture_mlir
        assert (
            'with options = {"my-option" = "my_option_value", "my-other-option" = false}'
            in capture_mlir
        )

    def test_transform_cancel_inverses_workflow(self, backend):
        """Test the integration for a circuit with a 'cancel_inverses' transform."""

        # Capture enabled

        qml.capture.enable()

        @qjit
        @qml.transforms.cancel_inverses
        @qml.qnode(qml.device(backend, wires=1))
        def captured_circuit(x: float):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        capture_result = captured_circuit(0.1)
        assert 'transform.apply_registered_pass "cancel-inverses"' in captured_circuit.mlir

        qml.capture.disable()

        # Capture disabled

        @qjit
        @qml.transforms.cancel_inverses
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        assert jnp.allclose(circuit(0.1), capture_result)

    def test_transform_merge_rotations_workflow(self, backend):
        """Test the integration for a circuit with a 'merge_rotations' transform."""

        # Capture enabled

        qml.capture.enable()

        @qjit
        @qml.transforms.merge_rotations
        @qml.qnode(qml.device(backend, wires=1))
        def captured_circuit(x: float):
            qml.RX(x, wires=0)
            qml.RX(x, wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        capture_result = captured_circuit(0.1)
        assert 'transform.apply_registered_pass "merge-rotations"' in captured_circuit.mlir

        qml.capture.disable()

        # Capture disabled

        @qjit
        @qml.transforms.merge_rotations
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float):
            qml.RX(x, wires=0)
            qml.RX(x, wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        assert jnp.allclose(circuit(0.1), capture_result)

    def test_chained_catalyst_transforms_workflow(self, backend):
        """Test the integration for a circuit with a combination of 'merge_rotations'
        and 'cancel_inverses' transforms."""

        # Capture enabled

        qml.capture.enable()

        @qml.qnode(qml.device(backend, wires=1))
        def captured_circuit(x: float):
            qml.RX(x, wires=0)
            qml.RX(x, wires=0)
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        captured_inverses_rotations = qjit(
            qml.transforms.cancel_inverses(qml.transforms.merge_rotations(captured_circuit))
        )
        captured_inverses_rotations_result = captured_inverses_rotations(0.1)
        assert has_catalyst_transforms(captured_inverses_rotations.mlir)

        captured_rotations_inverses = qjit(
            qml.transforms.merge_rotations(qml.transforms.cancel_inverses(captured_circuit)),
        )
        captured_rotations_inverses_result = captured_rotations_inverses(0.1)
        assert has_catalyst_transforms(captured_rotations_inverses.mlir)

        qml.capture.disable()

        # Capture disabled

        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float):
            qml.RX(x, wires=0)
            qml.RX(x, wires=0)
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        inverses_rotations_result = qjit(
            qml.transforms.cancel_inverses(qml.transforms.merge_rotations(circuit))
        )(0.1)
        rotations_inverses_result = qjit(
            qml.transforms.merge_rotations(qml.transforms.cancel_inverses(circuit))
        )(0.1)

        assert (
            inverses_rotations_result
            == rotations_inverses_result
            == captured_inverses_rotations_result
            == captured_rotations_inverses_result
        )

    def test_transform_unitary_to_rot_workflow(self, backend):
        """Test the integration for a circuit with a 'unitary_to_rot' transform."""

        U = qml.Rot(1.0, 2.0, 3.0, wires=0)

        # Capture enabled

        qml.capture.enable()

        @qjit
        @qml.transforms.unitary_to_rot
        @qml.qnode(qml.device(backend, wires=1))
        def captured_circuit(U: ShapedArray([2, 2], complex)):
            qml.QubitUnitary(U, 0)
            return qml.expval(qml.Z(0))

        capture_result = captured_circuit(U.matrix())
        assert is_unitary_rotated(captured_circuit.mlir)

        qml.capture.disable()

        # Capture disabled

        @qjit
        @qml.transforms.unitary_to_rot
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(U: ShapedArray([2, 2], complex)):
            qml.QubitUnitary(U, 0)
            return qml.expval(qml.Z(0))

        assert jnp.allclose(circuit(U.matrix()), capture_result)

    def test_mixed_transforms_workflow(self, backend):
        """Test the integration for a circuit with a combination of 'unitary_to_rot'
        and 'cancel_inverses' transforms."""

        U = qml.Rot(1.0, 2.0, 3.0, wires=0)

        # Capture enabled

        qml.capture.enable()

        @qml.qnode(qml.device(backend, wires=1))
        def captured_circuit(U: ShapedArray([2, 2], complex)):
            qml.QubitUnitary(U, 0)
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        # Case 1: During plxpr interpretation, first comes the PL transform
        # with Catalyst counterpart, second comes the PL transform without it

        captured_inverses_unitary = qjit(
            qml.transforms.cancel_inverses(qml.transforms.unitary_to_rot(captured_circuit)),
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
            qml.transforms.unitary_to_rot(qml.transforms.cancel_inverses(captured_circuit)),
        )
        captured_unitary_inverses_result = captured_unitary_inverses(U.matrix())

        # Both PL transforms should have been expaned and no Catalyst pass should have been
        # scheduled
        capture_mlir = captured_unitary_inverses.mlir
        assert 'transform.apply_registered_pass "cancel-inverses"' not in capture_mlir
        assert 'quantum.custom "Hadamard"' not in capture_mlir
        assert is_unitary_rotated(capture_mlir)

        qml.capture.disable()

        # Capture disabled

        @qml.qnode(qml.device(backend, wires=1))
        def circuit(U: ShapedArray([2, 2], complex)):
            qml.QubitUnitary(U, 0)
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        inverses_unitary_result = qjit(
            qml.transforms.cancel_inverses(qml.transforms.unitary_to_rot(circuit))
        )(U.matrix())
        unitary_inverses_result = qjit(
            qml.transforms.unitary_to_rot(qml.transforms.cancel_inverses(circuit))
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

        qml.capture.enable()

        @qjit
        @partial(qml.transforms.decompose, gate_set=[qml.RX, qml.RY, qml.RZ])
        @qml.qnode(qml.device(backend, wires=2))
        def captured_circuit(x: float, y: float, z: float):
            qml.Rot(x, y, z, 0)
            return qml.expval(qml.PauliZ(0))

        capture_result = captured_circuit(1.5, 2.5, 3.5)
        assert is_rot_decomposed(captured_circuit.mlir)

        qml.capture.disable()

        # Capture disabled

        @qjit
        @partial(qml.transforms.decompose, gate_set=[qml.RX, qml.RY, qml.RZ])
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(x: float, y: float, z: float):
            qml.Rot(x, y, z, 0)
            return qml.expval(qml.PauliZ(0))

        assert jnp.allclose(circuit(1.5, 2.5, 3.5), capture_result)

    def test_transform_graph_decompose_workflow(self, backend):
        """Test the integration for a circuit with a 'decompose' graph transform."""

        # Capture enabled

        qml.capture.enable()
        qml.decomposition.enable_graph()

        @qjit
        @partial(qml.transforms.decompose, gate_set=[qml.RX, qml.RY, qml.RZ])
        @qml.qnode(qml.device(backend, wires=2))
        def captured_circuit(x: float, y: float, z: float):
            m = qml.measure(0)

            @qml.cond(m)
            def cond_fn():
                qml.Rot(x, y, z, 0)

            cond_fn()
            return qml.expval(qml.PauliZ(0))

        capture_result = captured_circuit(1.5, 2.5, 3.5)

        qml.decomposition.disable_graph()
        qml.capture.disable()

        # Capture disabled
        @partial(qml.transforms.decompose, gate_set=[qml.RX, qml.RY, qml.RZ])
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(x: float, y: float, z: float):
            m = catalyst.measure(0)

            @catalyst.cond(m)
            def cond_fn():
                qml.Rot(x, y, z, 0)

            cond_fn()
            return qml.expval(qml.PauliZ(0))

        # non-capture pathway is not actively developed and raises unnecessary warnings (wontfix)
        with pytest.warns(UserWarning, match="MidCircuitMeasure does not define a decomposition"):
            with pytest.warns(UserWarning, match="Cond does not define a decomposition"):
                non_capture_result = qjit(circuit)(1.5, 2.5, 3.5)

        assert jnp.allclose(non_capture_result, capture_result)

    def test_transform_single_qubit_fusion_workflow(self, backend):
        """Test the integration for a circuit with a 'single_qubit_fusion' transform."""

        # Capture enabled

        qml.capture.enable()

        @qjit
        @qml.transforms.single_qubit_fusion
        @qml.qnode(qml.device(backend, wires=1))
        def captured_circuit():
            qml.Hadamard(wires=0)
            qml.Rot(0.1, 0.2, 0.3, wires=0)
            qml.Rot(0.4, 0.5, 0.6, wires=0)
            qml.RZ(0.1, wires=0)
            qml.RZ(0.4, wires=0)
            return qml.expval(qml.PauliZ(0))

        capture_result = captured_circuit()

        assert is_single_qubit_fusion_applied(captured_circuit.mlir)

        qml.capture.disable()

        # Capture disabled

        @qjit
        @qml.transforms.single_qubit_fusion
        @qml.qnode(qml.device(backend, wires=1))
        def circuit():
            qml.Hadamard(wires=0)
            qml.Rot(0.1, 0.2, 0.3, wires=0)
            qml.Rot(0.4, 0.5, 0.6, wires=0)
            qml.RZ(0.1, wires=0)
            qml.RZ(0.4, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert jnp.allclose(circuit(), capture_result)

    def test_transform_commute_controlled_workflow(self, backend):
        """Test the integration for a circuit with a 'commute_controlled' transform."""

        # Capture enabled

        qml.capture.enable()

        @qjit
        @partial(qml.transforms.commute_controlled, direction="left")
        @qml.qnode(qml.device(backend, wires=3))
        def captured_circuit():
            qml.CNOT(wires=[0, 2])
            qml.PauliX(wires=2)
            qml.RX(0.2, wires=2)
            qml.Toffoli(wires=[0, 1, 2])
            qml.CRX(0.1, wires=[0, 1])
            qml.PauliX(wires=1)
            return qml.expval(qml.PauliZ(0))

        capture_result = captured_circuit()

        capture_mlir = captured_circuit.mlir
        assert is_controlled_pushed_back(
            capture_mlir, 'quantum.custom "RX"', 'quantum.custom "CNOT"'
        )
        assert is_controlled_pushed_back(
            capture_mlir, 'quantum.custom "PauliX"', 'quantum.custom "CRX"'
        )

        qml.capture.disable()

        # Capture disabled

        @qjit
        @partial(qml.transforms.commute_controlled, direction="left")
        @qml.qnode(qml.device(backend, wires=3))
        def circuit():
            qml.CNOT(wires=[0, 2])
            qml.PauliX(wires=2)
            qml.RX(0.2, wires=2)
            qml.Toffoli(wires=[0, 1, 2])
            qml.CRX(0.1, wires=[0, 1])
            qml.PauliX(wires=1)
            return qml.expval(qml.PauliZ(0))

        assert jnp.allclose(circuit(), capture_result)

    def test_transform_merge_amplitude_embedding_workflow(self, backend):
        """Test the integration for a circuit with a 'merge_amplitude_embedding' transform."""

        # Capture enabled

        qml.capture.enable()

        @qjit
        @qml.transforms.merge_amplitude_embedding
        @qml.qnode(qml.device(backend, wires=2))
        def captured_circuit():
            qml.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=0)
            qml.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=1)
            return qml.expval(qml.PauliZ(0))

        capture_result = captured_circuit()
        assert is_amplitude_embedding_merged_and_decomposed(captured_circuit.mlir)

        qml.capture.disable()

        # Capture disabled

        @qjit
        @qml.transforms.merge_amplitude_embedding
        @qml.qnode(qml.device(backend, wires=2))
        def circuit():
            qml.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=0)
            qml.AmplitudeEmbedding(jnp.array([0.0, 1.0]), wires=1)
            return qml.expval(qml.PauliZ(0))

        assert jnp.allclose(circuit(), capture_result)

    def test_shots_usage(self, backend):
        """Test the integration for a circuit using shots explicitly."""

        # Capture enabled

        qml.capture.enable()

        @qjit
        @qml.set_shots(10)
        @qml.qnode(qml.device(backend, wires=2))
        def captured_circuit():
            @qml.for_loop(0, 2, 1)
            def loop_0(i):
                qml.RX(0, wires=i)

            loop_0()  # pylint: disable=no-value-for-parameter

            qml.RX(0, wires=0)
            return qml.sample()

        capture_result = captured_circuit()
        assert "shots(%" in captured_circuit.mlir

        qml.capture.disable()

        @qjit
        @qml.set_shots(10)
        @qml.qnode(qml.device(backend, wires=2))
        def circuit():
            @qml.for_loop(0, 2, 1)
            def loop_0(i):
                qml.RX(0, wires=i)

            loop_0()  # pylint: disable=no-value-for-parameter

            qml.RX(0, wires=0)
            return qml.sample()

        assert jnp.allclose(circuit(), capture_result)

    def test_static_variable_qnode(self, backend):
        """Test the integration for a circuit with a static variable."""

        qml.capture.enable()

        # Basic test
        @qjit(static_argnums=(0,))
        @qml.qnode(qml.device(backend, wires=1))
        def captured_circuit_1(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            return qml.expval(qml.PauliZ(0))

        result_1 = captured_circuit_1(1.5, 2.0)
        captured_circuit_1_mlir = captured_circuit_1.mlir
        assert "%cst = arith.constant 1.5" in captured_circuit_1_mlir
        assert 'quantum.custom "RX"(%cst)' in captured_circuit_1_mlir
        assert "%cst = arith.constant 2.0" not in captured_circuit_1_mlir

        # Test that qjit static_argnums takes precedence over the one on the qnode
        @qjit(static_argnums=1)
        @qml.qnode(qml.device(backend, wires=1), static_argnums=0)  # should be ignored
        def captured_circuit_2(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            return qml.expval(qml.PauliZ(0))

        result_2 = captured_circuit_2(1.5, 2.0)
        captured_circuit_2_mlir = captured_circuit_2.mlir
        assert "%cst = arith.constant 2.0" in captured_circuit_2_mlir
        assert 'quantum.custom "RY"(%cst)' in captured_circuit_2_mlir
        assert "%cst = arith.constant 1.5" not in captured_circuit_2_mlir

        assert jnp.allclose(result_1, result_2)

        # Test under a non qnode workflow function
        @qjit(static_argnums=(0,))
        def workflow(x, y):
            @qml.qnode(qml.device(backend, wires=1))
            def c():
                qml.RX(x, wires=0)
                qml.RY(y, wires=0)
                return qml.expval(qml.PauliZ(0))

            return c()

        _ = workflow(1.5, 2.0)
        captured_circuit_3_mlir = workflow.mlir
        assert "%cst = arith.constant 1.5" in captured_circuit_3_mlir
        assert 'quantum.custom "RX"(%cst)' in captured_circuit_3_mlir

        qml.capture.disable()


class TestControlFlow:
    """Integration tests for control flow."""

    @pytest.mark.parametrize("reverse", (True, False))
    def test_for_loop_outside_qnode(self, reverse):
        """Test that a for loop outside qnode can be executed."""

        qml.capture.enable()

        if reverse:
            start, stop, step = 6, 0, -2  # 6, 4, 2
        else:
            start, stop, step = 2, 7, 2  # 2, 4, 6

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def c(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        @qml.qjit
        def f(i0):
            @qml.for_loop(start, stop, step)
            def g(i, x):
                return c(i) + x

            return g(i0)  # pylint: disable=no-value-for-parameter

        out = f(3.0)
        assert qml.math.allclose(out, 3 + jnp.cos(2) + jnp.cos(4) + jnp.cos(6))

    def test_while_loop(self):
        """Test that a outside a qnode can be executed."""
        qml.capture.enable()

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        @qml.qjit
        def f(x):

            const = jnp.array([0, 1, 2])

            @qml.while_loop(lambda i, y: i < jnp.sum(const))
            def g(i, y):
                return i + 1, y + circuit(i)

            return g(0, x)

        ind, res = f(1.0)
        assert qml.math.allclose(ind, 3)
        expected = 1.0 + jnp.cos(0) + jnp.cos(1) + jnp.cos(2)
        assert qml.math.allclose(res, expected)

    # pylint: disable=unused-argument
    def test_for_loop_consts(self):
        """This tests for kinda a weird edge case bug where the consts where getting
        reordered when translating the inner jaxpr."""

        qml.capture.enable()

        @qml.qjit
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circuit(x, n):
            @qml.for_loop(3)
            def outer(i):

                @qml.for_loop(n)
                def inner(j):
                    qml.RY(x, wires=j)

                inner()  # pylint: disable=no-value-for-parameter

            outer()  # pylint: disable=no-value-for-parameter

            # Expected output: |100...>
            return [qml.expval(qml.PauliZ(i)) for i in range(3)]

        res1, res2, res3 = circuit(0.2, 2)

        assert qml.math.allclose(res1, jnp.cos(0.2 * 3))
        assert qml.math.allclose(res2, jnp.cos(0.2 * 3))
        assert qml.math.allclose(res3, 1)

    # pylint: disable=unused-argument
    def test_for_loop_consts_outside_qnode(self):
        """Similar test as above for weird edge case, but not using a qnode."""

        qml.capture.enable()

        @qml.qjit
        def f(x, n):
            @qml.for_loop(3)
            def outer(i, a):

                @qml.for_loop(n)
                def inner(j, b):
                    return b + x

                return inner(a)  # pylint: disable=no-value-for-parameter

            return outer(0.0)  # pylint: disable=no-value-for-parameter

        res = f(0.2, 2)
        assert qml.math.allclose(res, 0.2 * 2 * 3)


def test_adjoint_transform_integration():
    """Test that adjoint transforms can be used with capture enabled."""

    qml.capture.enable()

    def f(x):
        qml.IsingXX(2 * x, wires=(0, 1))
        qml.H(0)

    @qml.qjit
    @qml.qnode(qml.device("lightning.qubit", wires=3))
    def c(x):
        qml.adjoint(f)(x)
        return qml.expval(qml.Z(1))

    x = jnp.array(0.7)
    res = c(x)
    expected = jnp.cos(-2 * x)
    assert qml.math.allclose(res, expected)


@pytest.mark.parametrize("separate_funcs", (True, False))
def test_ctrl_transform_integration(separate_funcs):
    """Test that the ctrl transform can be applied."""

    qml.capture.enable()

    def f(x, y):
        qml.RY(3 * y, wires=3)
        qml.RX(2 * x, wires=3)

    @qml.qjit
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def c(x, y):
        qml.X(1)
        if separate_funcs:
            qml.ctrl(qml.ctrl(f, 0, [False]), 1, [True])(x, y)
        else:
            qml.ctrl(f, (0, 1), [False, True])(x, y)
        return qml.expval(qml.Z(3))

    x = jnp.array(0.5)
    y = jnp.array(0.9)
    res = c(x, y)
    expected = jnp.cos(2 * x) * jnp.cos(3 * y)
    assert qml.math.allclose(res, expected)


def test_different_static_argnums():
    """Test that the same qnode can be called different times with different static argnums."""

    qml.capture.enable()

    @qml.qnode(qml.device("lightning.qubit", wires=1), static_argnums=1)
    def c(x, pauli):
        if pauli == "X":
            qml.RX(x, 0)
        elif pauli == "Y":
            qml.RY(x, 0)
        else:
            qml.RZ(x, 0)
        return qml.state()

    @qml.qjit
    def w(x):
        return c(x, "X"), c(x, "Y"), c(x, "Z")

    resx, resy, resz = w(0.5)

    a = jnp.cos(0.5 / 2)
    b = jnp.sin(0.5 / 2)
    assert qml.math.allclose(resx, jnp.array([a, -b * 1j]))
    assert qml.math.allclose(resy, jnp.array([a, b]))
    assert qml.math.allclose(resz, jnp.array([a - b * 1j, 0]))
