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
from functools import partial

import jax.numpy as jnp
import pennylane as qml
import pytest
from jax.core import ShapedArray

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


# pylint: disable=too-many-public-methods
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

        default_capture_result_10_iterations = qml.qjit(circuit)(0)
        experimental_capture_result_10_iterations = qml.qjit(circuit, experimental_capture=True)(0)
        assert default_capture_result_10_iterations == experimental_capture_result_10_iterations

        default_capture_result_1_iteration = qml.qjit(circuit)(9)
        experimental_capture_result_1_iteration = qml.qjit(circuit, experimental_capture=True)(9)
        assert default_capture_result_1_iteration == experimental_capture_result_1_iteration

        default_capture_result_0_iterations = qml.qjit(circuit)(11)
        experimental_capture_result_0_iterations = qml.qjit(circuit, experimental_capture=True)(11)
        assert default_capture_result_0_iterations == experimental_capture_result_0_iterations

    def test_while_loop_workflow_closure(self, backend):
        """Test the integration for a circuit with a while_loop primitive using
        a closure variable."""

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

        default_capture_result = qml.qjit(circuit)(0, 2)
        experimental_capture_result = qml.qjit(circuit, experimental_capture=True)(0, 2)
        assert default_capture_result == experimental_capture_result

    def test_while_loop_workflow_nested(self, backend):
        """Test the integration for a circuit with a nested while_loop primitive."""

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

        default_capture_result = qml.qjit(circuit)(0, 0)
        experimental_capture_result = qml.qjit(circuit, experimental_capture=True)(0, 0)
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

    def test_transform_cancel_inverses_workflow(self, backend):
        """Test the integration for a circuit with a 'cancel_inverses' transform."""

        def func(x: float):
            @qml.transforms.cancel_inverses
            @qml.qnode(qml.device(backend, wires=1))
            def circuit(x: float):
                qml.RX(x, wires=0)
                qml.Hadamard(wires=0)
                qml.Hadamard(wires=0)
                return qml.expval(qml.PauliZ(0))

            return circuit(x)

        captured_func = qml.qjit(func, experimental_capture=True, target="mlir")
        assert 'transform.apply_registered_pass "remove-chained-self-inverse"' in captured_func.mlir

        no_capture_result = qml.qjit(func)(0.1)
        experimental_capture_result = captured_func(0.1)
        assert no_capture_result == experimental_capture_result

    def test_transform_merge_rotations_workflow(self, backend):
        """Test the integration for a circuit with a 'merge_rotations' transform."""

        def func(x: float):
            @qml.transforms.merge_rotations
            @qml.qnode(qml.device(backend, wires=1))
            def circuit(x: float):
                qml.RX(x, wires=0)
                qml.RX(x, wires=0)
                qml.Hadamard(wires=0)
                return qml.expval(qml.PauliZ(0))

            return circuit(x)

        captured_func = qml.qjit(func, experimental_capture=True, target="mlir")
        assert 'transform.apply_registered_pass "merge-rotations"' in captured_func.mlir

        no_capture_result = qml.qjit(func)(0.1)
        experimental_capture_result = captured_func(0.1)
        assert no_capture_result == experimental_capture_result

    def test_chained_catalyst_transforms_workflow(self, backend):
        """Test the integration for a circuit with a combination of 'merge_rotations'
        and 'cancel_inverses' transforms."""

        def has_catalyst_transforms(mlir):
            return (
                'transform.apply_registered_pass "remove-chained-self-inverse"' in mlir
                and 'transform.apply_registered_pass "merge-rotations"' in mlir
            )

        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float):
            qml.RX(x, wires=0)
            qml.RX(x, wires=0)
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        def inverses_rotations(x: float):
            return qml.transforms.cancel_inverses(qml.transforms.merge_rotations(circuit))(x)

        inverses_rotations_func = qml.qjit(
            inverses_rotations, experimental_capture=True, target="mlir"
        )
        assert has_catalyst_transforms(inverses_rotations_func.mlir)

        def rotations_inverses(x: float):
            return qml.transforms.merge_rotations(qml.transforms.cancel_inverses(circuit))(x)

        rotations_inverses_func = qml.qjit(
            rotations_inverses, experimental_capture=True, target="mlir"
        )
        assert has_catalyst_transforms(rotations_inverses_func.mlir)

        no_capture_inverses_rotations_result = qml.qjit(inverses_rotations)(0.1)
        no_capture_rotations_inverses_result = qml.qjit(rotations_inverses)(0.1)
        inverses_rotations_result = inverses_rotations_func(0.1)
        rotations_inverses_result = rotations_inverses_func(0.1)
        assert (
            no_capture_inverses_rotations_result
            == no_capture_rotations_inverses_result
            == inverses_rotations_result
            == rotations_inverses_result
        )

    def test_transform_unitary_to_rot_workflow(self, backend):
        """Test the integration for a circuit with a 'unitary_to_rot' transform."""

        def func(U: ShapedArray([2, 2], float)):
            @qml.transforms.unitary_to_rot
            @qml.qnode(qml.device(backend, wires=1))
            def circuit(U: ShapedArray([2, 2], float)):
                qml.QubitUnitary(U, 0)
                return qml.expval(qml.Z(0))

            return circuit(U)

        captured_func = qml.qjit(func, experimental_capture=True, target="mlir")
        assert is_unitary_rotated(captured_func.mlir)

        U = qml.Rot(1.0, 2.0, 3.0, wires=0)

        no_capture_result = qml.qjit(func)(U.matrix())
        experimental_capture_result = captured_func(U.matrix())
        assert no_capture_result == experimental_capture_result

    def test_mixed_transforms_workflow(self, backend):
        """Test the integration for a circuit with a combination of 'unitary_to_rot'
        and 'cancel_inverses' transforms."""

        @qml.qnode(qml.device(backend, wires=1))
        def circuit(U: ShapedArray([2, 2], float)):
            qml.QubitUnitary(U, 0)
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        # Case 1: During plxpr interpretation, first comes the PL transform
        # with Catalyst counterpart, second comes the PL transform without it

        def inverses_unitary(U: ShapedArray([2, 2], float)):
            return qml.transforms.cancel_inverses(qml.transforms.unitary_to_rot(circuit))(U)

        inverses_unitary_func = qml.qjit(inverses_unitary, experimental_capture=True, target="mlir")

        # Catalyst 'cancel_inverses' should have been scheduled as a pass
        # whereas PL 'unitary_to_rot' should have been expanded
        assert (
            'transform.apply_registered_pass "remove-chained-self-inverse"'
            in inverses_unitary_func.mlir
        )
        assert is_unitary_rotated(inverses_unitary_func.mlir)

        # Case 2: During plxpr interpretation, first comes the PL transform
        # without Catalyst counterpart, second comes the PL transform with it

        def unitary_inverses(U: ShapedArray([2, 2], float)):
            return qml.transforms.unitary_to_rot(qml.transforms.cancel_inverses(circuit))(U)

        unitary_inverses_func = qml.qjit(unitary_inverses, experimental_capture=True, target="mlir")

        # Both PL transforms should have been expaned and no Catalyst pass should have been
        # scheduled
        assert (
            'transform.apply_registered_pass "remove-chained-self-inverse"'
            not in unitary_inverses_func.mlir
        )
        assert 'quantum.custom "Hadamard"' not in unitary_inverses_func.mlir
        assert is_unitary_rotated(unitary_inverses_func.mlir)

        # Correctness assertions

        U = qml.Rot(1.0, 2.0, 3.0, wires=0)

        no_capture_inverses_unitary_result = qml.qjit(inverses_unitary)(U.matrix())
        no_capture_unitary_inverses_result = qml.qjit(unitary_inverses)(U.matrix())
        inverses_unitary_result = inverses_unitary_func(U.matrix())
        unitary_inverses_result = unitary_inverses_func(U.matrix())
        assert (
            no_capture_inverses_unitary_result
            == no_capture_unitary_inverses_result
            == inverses_unitary_result
            == unitary_inverses_result
        )

    def test_transform_decompose_workflow(self, backend):
        """Test the integration for a circuit with a 'decompose' transform."""

        def func(x: float, y: float, z: float):
            @qml.qnode(qml.device(backend, wires=2))
            def circuit(x: float, y: float, z: float):
                qml.Rot(x, y, z, 0)
                return qml.expval(qml.PauliZ(0))

            return qml.transforms.decompose(circuit, gate_set=[qml.RX, qml.RY, qml.RZ])(x, y, z)

        captured_func = qml.qjit(func, experimental_capture=True, target="mlir")

        assert is_rot_decomposed(captured_func.mlir)

        no_capture_result = qml.qjit(func)(1.5, 2.5, 3.5)
        experimental_capture_result = captured_func(1.5, 2.5, 3.5)
        assert no_capture_result == experimental_capture_result

    def test_transform_map_wires_workflow(self, backend):
        """Test the integration for a circuit with a 'map_wires' transform."""

        def func(x: float):
            @partial(qml.map_wires, wire_map={0: 1})
            @qml.qnode(qml.device(backend, wires=2))
            def circuit(x):
                qml.RX(x, 0)
                return qml.expval(qml.PauliZ(0))

            return circuit(x)

        captured_func = qml.qjit(func, experimental_capture=True, target="mlir")

        assert is_wire_mapped(captured_func.mlir)

        no_capture_result = qml.qjit(func)(1.5)
        experimental_capture_result = captured_func(1.5)
        assert no_capture_result == experimental_capture_result

    def test_transform_single_qubit_fusion_workflow(self, backend):
        """Test the integration for a circuit with a 'single_qubit_fusion' transform."""

        def func():
            @qml.transforms.single_qubit_fusion
            @qml.qnode(qml.device(backend, wires=1))
            def circuit():
                qml.Hadamard(wires=0)
                qml.Rot(0.1, 0.2, 0.3, wires=0)
                qml.Rot(0.4, 0.5, 0.6, wires=0)
                qml.RZ(0.1, wires=0)
                qml.RZ(0.4, wires=0)
                return qml.expval(qml.PauliZ(0))

            return circuit()

        captured_func = qml.qjit(func, experimental_capture=True, target="mlir")

        assert is_single_qubit_fusion_applied(captured_func.mlir)

        no_capture_result = qml.qjit(func)()
        experimental_capture_result = captured_func()
        assert no_capture_result == experimental_capture_result

    def test_transform_commute_controlled_workflow(self, backend):
        """Test the integration for a circuit with a 'commute_controlled' transform."""

        def func():
            @qml.qnode(qml.device(backend, wires=3))
            def circuit():
                qml.CNOT(wires=[0, 2])
                qml.PauliX(wires=2)
                qml.RX(0.2, wires=2)
                qml.Toffoli(wires=[0, 1, 2])
                qml.CRX(0.1, wires=[0, 1])
                qml.PauliX(wires=1)
                return qml.expval(qml.PauliZ(0))

            return qml.transforms.commute_controlled(circuit, direction="left")()

        captured_func = qml.qjit(func, experimental_capture=True, target="mlir")

        assert is_controlled_pushed_back(
            captured_func.mlir, 'quantum.custom "RX"', 'quantum.custom "CNOT"'
        )
        assert is_controlled_pushed_back(
            captured_func.mlir, 'quantum.custom "PauliX"', 'quantum.custom "CRX"'
        )

        no_capture_result = qml.qjit(func)()
        experimental_capture_result = captured_func()
        assert no_capture_result == experimental_capture_result
