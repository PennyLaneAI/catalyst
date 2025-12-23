# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit test module for the draw function in the unified compiler inspection module."""
# pylint: disable=unnecessary-lambda

import jax
import pytest

pytestmark = pytest.mark.xdsl
xdsl = pytest.importorskip("xdsl")
import matplotlib
import numpy as np
import pennylane as qml

from catalyst.python_interface.inspection import draw, draw_graph
from catalyst.python_interface.transforms import (
    iterative_cancel_inverses_pass,
    merge_rotations_pass,
)


@pytest.mark.usefixtures("use_capture")
class TestDraw:
    """Unit tests for the draw function in the unified compiler inspection module."""

    @pytest.fixture
    def transforms_circuit(self):
        """Fixture for a circuit."""

        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circ():
            qml.RX(1, 0)
            qml.RX(2.0, 0)
            qml.RY(3.0, 1)
            qml.RY(4.0, 1)
            qml.RZ(5.0, 2)
            qml.RZ(6.0, 2)
            qml.Hadamard(0)
            qml.Hadamard(0)
            qml.CNOT([0, 1])
            qml.CNOT([0, 1])
            qml.Hadamard(1)
            qml.Hadamard(1)
            qml.RZ(7.0, 0)
            qml.RZ(8.0, 0)
            qml.CNOT([0, 2])
            qml.CNOT([0, 2])
            return qml.state()

        return circ

    @pytest.mark.parametrize("qjit", [True, False])
    @pytest.mark.parametrize(
        "level, expected",
        [
            (
                0,
                "0: ──RX──RX──H──H─╭●─╭●──RZ──RZ─╭●─╭●─┤  State\n"
                "1: ──RY──RY───────╰X─╰X──H───H──│──│──┤  State\n"
                "2: ──RZ──RZ─────────────────────╰X─╰X─┤  State",
            ),
            (
                1,
                "0: ──RX──H──H─╭●─╭●──RZ────╭●─╭●─┤  State\n"
                "1: ──RY───────╰X─╰X──H───H─│──│──┤  State\n"
                "2: ──RZ────────────────────╰X─╰X─┤  State",
            ),
            (2, "0: ──RX──RZ─┤  State\n1: ──RY─────┤  State\n2: ──RZ─────┤  State"),
            (None, "0: ──RX──RZ─┤  State\n1: ──RY─────┤  State\n2: ──RZ─────┤  State"),
            (50, "0: ──RX──RZ─┤  State\n1: ──RY─────┤  State\n2: ──RZ─────┤  State"),
        ],
    )
    def test_multiple_levels_xdsl(self, transforms_circuit, level, qjit, expected):
        """Test that multiple levels of transformation are applied correctly with xDSL
        compilation passes."""

        transforms_circuit = iterative_cancel_inverses_pass(
            merge_rotations_pass(transforms_circuit)
        )

        if qjit:
            transforms_circuit = qml.qjit(transforms_circuit)

        assert draw(transforms_circuit, level=level)() == expected

    @pytest.mark.parametrize("qjit", [True, False])
    @pytest.mark.parametrize(
        "level, expected",
        [
            (
                0,
                "0: ──RX──RX──H──H─╭●─╭●──RZ──RZ─╭●─╭●─┤  State\n"
                "1: ──RY──RY───────╰X─╰X──H───H──│──│──┤  State\n"
                "2: ──RZ──RZ─────────────────────╰X─╰X─┤  State",
            ),
            (
                1,
                "0: ──RX──H──H─╭●─╭●──RZ────╭●─╭●─┤  State\n"
                "1: ──RY───────╰X─╰X──H───H─│──│──┤  State\n"
                "2: ──RZ────────────────────╰X─╰X─┤  State",
            ),
            (2, "0: ──RX──RZ─┤  State\n1: ──RY─────┤  State\n2: ──RZ─────┤  State"),
            (None, "0: ──RX──RZ─┤  State\n1: ──RY─────┤  State\n2: ──RZ─────┤  State"),
            (50, "0: ──RX──RZ─┤  State\n1: ──RY─────┤  State\n2: ──RZ─────┤  State"),
        ],
    )
    def test_multiple_levels_catalyst(self, transforms_circuit, level, qjit, expected):
        """Test that multiple levels of transformation are applied correctly with Catalyst
        compilation passes."""

        transforms_circuit = qml.transforms.cancel_inverses(
            qml.transforms.merge_rotations(transforms_circuit)
        )

        if qjit:
            transforms_circuit = qml.qjit(transforms_circuit)

        assert draw(transforms_circuit, level=level)() == expected

    @pytest.mark.parametrize("qjit", [True, False])
    @pytest.mark.parametrize(
        "level, expected",
        [
            (
                0,
                "0: ──RX──RX──H──H─╭●─╭●──RZ──RZ─╭●─╭●─┤  State\n"
                "1: ──RY──RY───────╰X─╰X──H───H──│──│──┤  State\n"
                "2: ──RZ──RZ─────────────────────╰X─╰X─┤  State",
            ),
            (
                1,
                "0: ──RX──H──H─╭●─╭●──RZ────╭●─╭●─┤  State\n"
                "1: ──RY───────╰X─╰X──H───H─│──│──┤  State\n"
                "2: ──RZ────────────────────╰X─╰X─┤  State",
            ),
            (2, "0: ──RX──RZ─┤  State\n1: ──RY─────┤  State\n2: ──RZ─────┤  State"),
            (None, "0: ──RX──RZ─┤  State\n1: ──RY─────┤  State\n2: ──RZ─────┤  State"),
            (50, "0: ──RX──RZ─┤  State\n1: ──RY─────┤  State\n2: ──RZ─────┤  State"),
        ],
    )
    def test_multiple_levels_xdsl_catalyst(self, transforms_circuit, level, qjit, expected):
        """Test that multiple levels of transformation are applied correctly with xDSL and
        Catalyst compilation passes."""

        transforms_circuit = iterative_cancel_inverses_pass(
            qml.transforms.merge_rotations(transforms_circuit)
        )
        if qjit:
            transforms_circuit = qml.qjit(transforms_circuit)

        assert draw(transforms_circuit, level=level)() == expected

    @pytest.mark.parametrize("qjit", [True, False])
    @pytest.mark.parametrize(
        "level, expected",
        [
            (
                0,
                "0: ──RX──RX──H──H─╭●─╭●──RZ──RZ─╭●─╭●─┤  State\n"
                "1: ──RY──RY───────╰X─╰X──H───H──│──│──┤  State\n"
                "2: ──RZ──RZ─────────────────────╰X─╰X─┤  State",
            ),
            (
                1,
                "0: ──RX──RX──H──H─╭●─╭●──RZ──RZ─╭●─╭●─┤  State\n"
                "1: ──RY──RY───────╰X─╰X──H───H──│──│──┤  State\n"
                "2: ──RZ──RZ─────────────────────╰X─╰X─┤  State",
            ),
            (
                2,
                "0: ──RX──RX──H──H─╭●─╭●──RZ──RZ─╭●─╭●─┤  State\n"
                "1: ──RY──RY───────╰X─╰X──H───H──│──│──┤  State\n"
                "2: ──RZ──RZ─────────────────────╰X─╰X─┤  State",
            ),
            (
                None,
                "0: ──RX──RX──H──H─╭●─╭●──RZ──RZ─╭●─╭●─┤  State\n"
                "1: ──RY──RY───────╰X─╰X──H───H──│──│──┤  State\n"
                "2: ──RZ──RZ─────────────────────╰X─╰X─┤  State",
            ),
            (
                50,
                "0: ──RX──RX──H──H─╭●─╭●──RZ──RZ─╭●─╭●─┤  State\n"
                "1: ──RY──RY───────╰X─╰X──H───H──│──│──┤  State\n"
                "2: ──RZ──RZ─────────────────────╰X─╰X─┤  State",
            ),
        ],
    )
    def test_no_passes(self, transforms_circuit, level, qjit, expected):
        """Test that if no passes are applied, the circuit is still visualized."""

        if qjit:
            transforms_circuit = qml.qjit(transforms_circuit)

        assert draw(transforms_circuit, level=level)() == expected

    @pytest.mark.parametrize(
        "op, expected",
        [
            (
                lambda: qml.ctrl(qml.RX(0.1, 0), control=(1, 2, 3)),
                "1: ─╭●──┤  State\n2: ─├●──┤  State\n3: ─├●──┤  State\n0: ─╰RX─┤  State",
            ),
            (
                lambda: qml.ctrl(qml.RX(0.1, 0), control=(1, 2, 3), control_values=(0, 1, 0)),
                "1: ─╭○──┤  State\n2: ─├●──┤  State\n3: ─├○──┤  State\n0: ─╰RX─┤  State",
            ),
            (
                lambda: qml.adjoint(qml.ctrl(qml.RX(0.1, 0), (1, 2, 3), control_values=(0, 1, 0))),
                "1: ─╭○───┤  State\n2: ─├●───┤  State\n3: ─├○───┤  State\n0: ─╰RX†─┤  State",
            ),
            (
                lambda: qml.ctrl(qml.adjoint(qml.RX(0.1, 0)), (1, 2, 3), control_values=(0, 1, 0)),
                "1: ─╭○───┤  State\n2: ─├●───┤  State\n3: ─├○───┤  State\n0: ─╰RX†─┤  State",
            ),
        ],
    )
    def test_ctrl_adjoint_variants(self, op, expected):
        """
        Test the visualization of control and adjoint variants.
        """

        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circuit():
            op()
            return qml.state()

        assert draw(circuit)() == expected

    def test_ctrl_before_custom_op(self):
        """
        Test the visualization of control operations before custom ops.
        """

        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circuit():
            qml.ctrl(qml.X(3), control=[0, 1, 2], control_values=[1, 0, 1])
            qml.RX(0.1, 2)
            return qml.state()

        assert (
            draw(circuit)()
            == "0: ─╭●─────┤  State\n1: ─├○─────┤  State\n2: ─├●──RX─┤  State\n3: ─╰X─────┤  State"
        )

    @pytest.mark.parametrize(
        "measurement, expected",
        [
            (
                lambda: (qml.probs(0), qml.probs(1), qml.probs(2)),
                "0: ──RX─┤  Probs\n1: ──RY─┤  Probs\n2: ──RZ─┤  Probs",
            ),
            (
                lambda: qml.probs(),
                "0: ──RX─┤  Probs\n1: ──RY─┤  Probs\n2: ──RZ─┤  Probs",
            ),
            (
                lambda: qml.sample(),
                "0: ──RX─┤  Sample\n1: ──RY─┤  Sample\n2: ──RZ─┤  Sample",
            ),
            (
                lambda: (
                    qml.expval(qml.X(0)),
                    qml.expval(qml.Y(1)),
                    qml.expval(qml.Z(2)),
                ),
                "0: ──RX─┤  <X>\n1: ──RY─┤  <Y>\n2: ──RZ─┤  <Z>",
            ),
            (
                lambda: (
                    qml.expval(qml.X(0) @ qml.Y(1)),
                    qml.expval(qml.Y(1) @ qml.Z(2) @ qml.X(0)),
                    qml.expval(qml.Z(2) @ qml.X(0) @ qml.Y(1)),
                ),
                "0: ──RX─┤ ╭<X@Y> ╭<Y@Z@X> ╭<Z@X@Y>\n"
                "1: ──RY─┤ ╰<X@Y> ├<Y@Z@X> ├<Z@X@Y>\n"
                "2: ──RZ─┤        ╰<Y@Z@X> ╰<Z@X@Y>",
            ),
            (
                lambda: (
                    qml.expval(
                        qml.Hamiltonian([0.2, 0.2], [qml.PauliX(0), qml.Y(1)])
                        @ qml.Hamiltonian([0.1, 0.1], [qml.PauliZ(2), qml.PauliZ(3)])
                    )
                ),
                "0: ──RX─┤ ╭<(𝓗)@(𝓗)>\n"
                "1: ──RY─┤ ├<(𝓗)@(𝓗)>\n"
                "2: ──RZ─┤ ├<(𝓗)@(𝓗)>\n"
                "3: ─────┤ ╰<(𝓗)@(𝓗)>",
            ),
            (
                lambda: (qml.var(qml.X(0)), qml.var(qml.Y(1)), qml.var(qml.Z(2))),
                "0: ──RX─┤  Var[X]\n1: ──RY─┤  Var[Y]\n2: ──RZ─┤  Var[Z]",
            ),
            (
                lambda: (
                    qml.var(qml.X(0) @ qml.Y(1)),
                    qml.var(qml.Y(1) @ qml.Z(2) @ qml.X(0)),
                    qml.var(qml.Z(2) @ qml.X(0) @ qml.Y(1)),
                ),
                "0: ──RX─┤ ╭Var[X@Y] ╭Var[Y@Z@X] ╭Var[Z@X@Y]\n"
                "1: ──RY─┤ ╰Var[X@Y] ├Var[Y@Z@X] ├Var[Z@X@Y]\n"
                "2: ──RZ─┤           ╰Var[Y@Z@X] ╰Var[Z@X@Y]",
            ),
        ],
    )
    def test_measurements(self, measurement, expected):
        """
        Test the visualization of measurements.
        """

        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circuit():
            qml.RX(0.1, 0)
            qml.RY(0.2, 1)
            qml.RZ(0.3, 2)
            return measurement()

        if isinstance(measurement(), qml.measurements.SampleMP):
            circuit = qml.set_shots(10)(circuit)

        assert draw(circuit)() == expected

    def test_global_phase(self):
        """Test the visualization of global phase shifts."""

        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circuit():
            qml.H(0)
            qml.H(1)
            qml.H(2)
            qml.GlobalPhase(0.5)
            return qml.state()

        assert draw(circuit)() == (
            "0: ──H─╭GlobalPhase─┤  State\n"
            "1: ──H─├GlobalPhase─┤  State\n"
            "2: ──H─╰GlobalPhase─┤  State"
        )

    @pytest.mark.parametrize(
        "postselect, mid_measure_label",
        [
            (None, "┤↗├"),
            (0, "┤↗₀├"),
            (1, "┤↗₁├"),
        ],
    )
    def test_draw_mid_circuit_measurement_postselect(self, postselect, mid_measure_label):
        """Test that mid-circuit measurements are drawn correctly."""

        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit():
            qml.Hadamard(0)
            qml.measure(0, postselect=postselect)
            qml.PauliX(0)
            return qml.expval(qml.PauliZ(0))

        drawing = draw(circuit)()
        expected_drawing = "0: ──H──" + mid_measure_label + "──X─┤  <Z>"

        assert drawing == expected_drawing

    @pytest.mark.parametrize(
        "ops, expected",
        [
            (
                [
                    (qml.QubitUnitary, jax.numpy.array([[0, 1], [1, 0]]), [0]),
                    (
                        qml.QubitUnitary,
                        jax.numpy.array([[0, 1, 0, 1], [1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]]),
                        [0, 1],
                    ),
                    (qml.QubitUnitary, jax.numpy.zeros((8, 8)), [0, 1, 2]),
                    (
                        qml.QubitUnitary,
                        jax.numpy.array([[0, 1, 0, 1], [1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]]),
                        [0, 1],
                    ),
                    (qml.QubitUnitary, jax.numpy.array([[0, 1], [1, 0]]), [0]),
                ],
                "0: ──U(M0)─╭U(M1)─╭U(M2)─╭U(M1)──U(M0)─┤  State\n"
                "1: ────────╰U(M1)─├U(M2)─╰U(M1)────────┤  State\n"
                "2: ───────────────╰U(M2)───────────────┤  State",
            ),
            (
                [
                    (qml.StatePrep, jax.numpy.array([1, 0]), [0]),
                    (qml.StatePrep, jax.numpy.array([1, 0, 0, 0]), [0, 1]),
                    (
                        qml.StatePrep,
                        jax.numpy.array([1, 0, 0, 0, 1, 0, 0, 0]),
                        [0, 1, 2],
                    ),
                    (qml.StatePrep, jax.numpy.array([1, 0, 0, 0]), [0, 1]),
                    (qml.StatePrep, jax.numpy.array([1, 0]), [0]),
                ],
                "0: ──|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩──|Ψ⟩─┤  State\n"
                "1: ──────╰|Ψ⟩─├|Ψ⟩─╰|Ψ⟩──────┤  State\n"
                "2: ───────────╰|Ψ⟩───────────┤  State",
            ),
            (
                [
                    (qml.MultiRZ, 0.1, [0]),
                    (qml.MultiRZ, 0.1, [0, 1]),
                    (qml.MultiRZ, 0.1, [0, 1, 2]),
                    (qml.MultiRZ, 0.1, [0, 1]),
                    (qml.MultiRZ, 0.1, [0]),
                ],
                "0: ──MultiRZ─╭MultiRZ─╭MultiRZ─╭MultiRZ──MultiRZ─┤  State\n"
                "1: ──────────╰MultiRZ─├MultiRZ─╰MultiRZ──────────┤  State\n"
                "2: ───────────────────╰MultiRZ───────────────────┤  State",
            ),
            (
                [
                    (qml.BasisState, jax.numpy.array([1]), [0]),
                    (qml.BasisState, jax.numpy.array([1, 0]), [0, 1]),
                    (qml.BasisState, jax.numpy.array([1, 0, 0]), [0, 1, 2]),
                    (qml.BasisState, jax.numpy.array([1, 0]), [0, 1]),
                    (qml.BasisState, jax.numpy.array([1]), [0]),
                ],
                "0: ──|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩──|Ψ⟩─┤  State\n"
                "1: ──────╰|Ψ⟩─├|Ψ⟩─╰|Ψ⟩──────┤  State\n"
                "2: ───────────╰|Ψ⟩───────────┤  State",
            ),
        ],
    )
    def test_visualization_cases(self, ops, expected):
        """
        Test the visualization of the quantum operations defined in the unified compiler dialect.
        """

        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circuit():
            for op, param, wires in ops:
                op(param, wires=wires)
            return qml.state()

        assert draw(circuit)() == expected

    def test_reshape(self):
        """Test that the visualization works when the parameters are reshaped."""

        one_dim = jax.numpy.array([1, 0])
        two_dim = jax.numpy.array([[0, 1], [1, 0]])
        eight_dim = jax.numpy.zeros((8, 8))

        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit():
            qml.RX(one_dim[0], wires=0)
            qml.RZ(two_dim[0, 0], wires=0)
            qml.QubitUnitary(eight_dim[:2, :2], wires=0)
            qml.QubitUnitary(eight_dim[0:4, 0:4], wires=[0, 1])
            return qml.state()

        expected = (
            "0: ──RX(M0)──RZ(M0)──U(M1)─╭U(M2)─┤  State\n"
            "1: ────────────────────────╰U(M2)─┤  State"
        )
        assert draw(circuit)() == expected

    def test_args_warning(self):
        """Test that a warning is raised when dynamic arguments are used."""

        # pylint: disable=unused-argument
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circ(arg):
            qml.RX(0.1, wires=0)
            return qml.state()

        with pytest.warns(UserWarning):
            draw(circ)(0.1)

    def adjoint_op_not_implemented(self):
        """Test that NotImplementedError is raised when AdjointOp is used."""

        @qml.qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit():
            qml.adjoint(qml.QubitUnitary)(jax.numpy.array([[0, 1], [1, 0]]), wires=[0])
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(NotImplementedError, match="not yet supported"):
            print(draw(circuit)())

    def test_cond_not_implemented(self):
        """Test that NotImplementedError is raised when cond is used."""

        @qml.qjit
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit():
            m0 = qml.measure(0, reset=False, postselect=0)
            qml.cond(m0, qml.RX, qml.RY)(1.23, 1)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(NotImplementedError, match="not yet supported"):
            print(draw(circuit)())

    def test_for_loop_not_implemented(self):
        """Test that NotImplementedError is raised when for loop is used."""

        @qml.qjit(autograph=True)
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit():
            for _ in range(3):
                qml.RX(0.1, 0)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(NotImplementedError, match="not yet supported"):
            print(draw(circuit)())

    def test_while_loop_not_implemented(self):
        """Test that NotImplementedError is raised when while loop is used."""

        @qml.qjit(autograph=True)
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit():
            i = 0
            while i < 3:
                qml.RX(0.1, 0)
                i += 1
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(NotImplementedError, match="not yet supported"):
            print(draw(circuit)())


@pytest.mark.usefixtures("use_both_frontend")
class TestDrawGraph:
    """Tests the `draw_graph` frontend."""

    @pytest.mark.parametrize(
        "unsupported_level",
        (
            [0],
            [0, 1],
            (0,),
            (0, 1),
            slice(0, 2),
            "cancel-inverses",
        ),
    )
    def test_unsupported_levels(self, unsupported_level):
        """Tests proper handling of the level argument."""

        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(qml.device("null.qubit", wires=2))
        def qjit_qnode():
            qml.H(0)
            return qml.expval(qml.Z(0))

        with pytest.raises(TypeError, match="The `level` argument must be an integer or `None`"):
            _ = draw_graph(qjit_qnode, level=unsupported_level)()

    def test_unsupported_qnode(self):
        """Tests that only qjit'd qnodes are allowed to be visualized."""

        @qml.qnode(qml.device("null.qubit", wires=2))
        def qnode():
            qml.H(0)
            return qml.expval(qml.Z(0))

        with pytest.raises(TypeError, match="The circuit must be a qjit compiled qnode"):
            _ = draw_graph(qnode)()

    def test_return_types(self):
        """Tests the return types of the function without crashing CI."""

        @qml.qjit(autograph=True, target="mlir")
        @qml.qnode(qml.device("null.qubit", wires=2))
        def qjit_qnode():
            qml.H(0)
            return qml.expval(qml.Z(0))

        # from unittest.mock import patch

        ## Mock out the creation of the PNG
        # with patch("pydot.Dot.create_png") as mock_create_png:
        #    # Creates a simple 1x1 pixel transparent image https://en.wikipedia.org/wiki/PNG
        #    mock_create_png.return_value = (
        #        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00"
        #        b"\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9c"
        #        b"c\x00\x01\x00\x00\x05\x00\x01\r\n\x2e\xe4\x00\x00\x00\x00IEND\xaeB`\x82"
        #    )

        fig, axes = draw_graph(qjit_qnode)()

        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(axes, matplotlib.axes._axes.Axes)

    def test_transforms_step_through(self):
        """Tests that the level argument controls transformations step through."""

        @qml.qjit
        @qml.transforms.merge_rotations
        @qml.transforms.cancel_inverses
        @qml.qnode(qml.device("null.qubit", wires=3))
        def circuit():
            qml.H(0)
            qml.T(1)
            qml.H(0)
            qml.RX(0.1, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.X(0))

        drawer = draw_graph(circuit)
        _ = drawer()
        cache = drawer._cache

        no_transforms = cache[0][0]
        assert cache[0][1] == "Before MLIR Passes"
        cancel_inverses = cache[1][0]
        assert cache[1][1] == "cancel-inverses"
        merge_rotations = cache[2][0]
        assert cache[2][1] == "merge-rotations"

        # Check no transforms
        assert no_transforms.count("<name> Hadamard|<wire> [0]") == 2
        assert no_transforms.count("<name> T|<wire> [1]") == 1
        assert no_transforms.count("<name> RX|<wire> [0]") == 2
        assert no_transforms.count("expval(PauliX)") == 1

        # Cancel inverses
        assert cancel_inverses.count("<name> Hadamard|<wire> [0]") == 0
        assert cancel_inverses.count("<name> T|<wire> [1]") == 1
        assert cancel_inverses.count("<name> RX|<wire> [0]") == 2
        assert cancel_inverses.count("expval(PauliX)") == 1

        # Merge rotations
        assert merge_rotations.count("<name> Hadamard|<wire> [0]") == 0
        assert merge_rotations.count("<name> T|<wire> [1]") == 1
        assert merge_rotations.count("<name> RX|<wire> [0]") == 1
        assert merge_rotations.count("expval(PauliX)") == 1

    def test_empty_passpipeline(self):
        """Tests that it works with an empty pass pipeline."""

        @qml.qjit
        @qml.qnode(qml.device("null.qubit", wires=3))
        def circuit():
            qml.H(0)
            qml.T(1)
            qml.H(0)
            qml.RX(0.1, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.X(0))

        drawer = draw_graph(circuit)
        _ = drawer()
        cache = drawer._cache

        assert len(cache) == 1
        graph = cache[0][0]
        assert cache[0][1] == "Before MLIR Passes"

        assert graph.count("<name> Hadamard|<wire> [0]") == 2
        assert graph.count("<name> T|<wire> [1]") == 1
        assert graph.count("<name> RX|<wire> [0]") == 2
        assert graph.count("expval(PauliX)") == 1

    def test_early_callback_exit(self):
        """Tests that unnecessary callbacks aren't performend."""

        @qml.qjit
        @qml.transforms.merge_rotations
        @qml.transforms.cancel_inverses
        @qml.qnode(qml.device("null.qubit", wires=3))
        def circuit():
            qml.H(0)
            qml.T(1)
            qml.H(0)
            qml.RX(0.1, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.X(0))

        # Show circuit after cancel_inverses transform
        _ = draw_graph(circuit, level=1)()

        drawer = draw_graph(circuit)
        _ = drawer()
        cache = drawer._cache
        assert len(cache) == 2

        no_transforms = cache[0][0]
        assert cache[0][1] == "Before MLIR Passes"
        cancel_inverses = cache[1][0]
        assert cache[1][1] == "cancel-inverses"

        # Check no transforms
        assert no_transforms.count("<name> Hadamard|<wire> [0]") == 2
        assert no_transforms.count("<name> T|<wire> [1]") == 1
        assert no_transforms.count("<name> RX|<wire> [0]") == 2
        assert no_transforms.count("expval(PauliX)") == 1

        # Cancel inverses
        assert cancel_inverses.count("<name> Hadamard|<wire> [0]") == 0
        assert cancel_inverses.count("<name> T|<wire> [1]") == 1
        assert cancel_inverses.count("<name> RX|<wire> [0]") == 2
        assert cancel_inverses.count("expval(PauliX)") == 1


if __name__ == "__main__":
    pytest.main(["-x", __file__])
