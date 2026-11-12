# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for ``qp.flatten``: a compile-time, one-way exit from ``qjit`` programs to tapes."""

# pylint: disable=unnecessary-lambda,expression-not-assigned

import numpy as np
import pennylane as qp
import pytest
from pennylane.core.operator import Operator2
from pennylane.ops import MidMeasure
from pennylane.ops.mid_measure import MeasurementValue
from pennylane.tape import QuantumScript

from catalyst.python_interface.tape_extract import FlattenError

pytestmark = pytest.mark.capture

# Compiling the decomposition rules of every gate is slow and not needed by these tests.
qjit = qp.qjit(capture=True, collect_decomp_rules=False)


def assert_ops_equal(actual, expected):
    """Compare operator lists, allowing for floating point differences in the parameters."""
    assert len(actual) == len(expected), f"{actual} != {expected}"
    for a, e in zip(actual, expected):
        assert qp.equal(a, e, atol=1e-12), f"{a} != {e}"


def _requirements_circuit(call_loop=True):
    # pylint: disable=unused-variable
    def f():
        qp.H(0)
        qp.H(1)
        qp.H(0)
        qp.H(2)
        qp.H(3)

        @qp.for_loop(0, 3)
        def loop(i):
            qp.CNOT((i, i + 1))

        if call_loop:
            loop()

        qp.RX(0.1, 0)
        qp.RX(0.2, 0)
        qp.RZ(0.3, 2)
        qp.RZ(0.4, 2)

        return qp.state()

    return qp.transforms.merge_rotations(
        qp.transforms.cancel_inverses(qp.qnode(qp.device("lightning.qubit", wires=4))(f))
    )


REQUIREMENTS_OPS = [
    qp.H(1),
    qp.H(2),
    qp.H(3),
    qp.CNOT((0, 1)),
    qp.CNOT((1, 2)),
    qp.CNOT((2, 3)),
    qp.RX(0.3, 0),
    qp.RZ(0.7, 2),
]


class TestRequirementsExamples:
    """The examples of the requirements."""

    def test_requirements_example(self):
        """The compile pipeline (cancel_inverses, merge_rotations) is applied and the loop is
        unrolled."""
        f = qjit(_requirements_circuit())
        tape = qp.flatten(f)()

        assert isinstance(tape, QuantumScript)
        assert_ops_equal(tape.operations, REQUIREMENTS_OPS)
        assert tape.measurements == [qp.state()]
        assert tape.shots == qp.measurements.Shots(None)

    @pytest.mark.slow
    def test_requirements_example_default_options(self):
        """The example with the default qjit options (this compiles all decomposition rules
        ahead of time and is slow)."""
        f = qp.qjit(capture=True)(_requirements_circuit())
        tape = qp.flatten(f)()
        assert_ops_equal(tape.operations, REQUIREMENTS_OPS)
        assert tape.measurements == [qp.state()]
        assert tape.shots.total_shots is None

    def test_uncalled_loop_is_not_applied(self):
        """A ``for_loop`` that is defined but never called applies no gates, like in PennyLane.
        (The requirements example defines ``loop`` without calling it.)"""
        f = qjit(_requirements_circuit(call_loop=False))
        tape = qp.flatten(f)()
        assert_ops_equal(
            tape.operations, [qp.H(1), qp.H(2), qp.H(3), qp.RX(0.3, 0), qp.RZ(0.7, 2)]
        )

    def test_mcm_example_matches_non_qjit_tape(self):
        """Branches conditioned on MCMs become Conditional ops tied to the MidMeasure, exactly as
        in the non-qjit PennyLane pathway."""

        @qp.set_shots(10)
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def f():
            qp.H(0)
            m = qp.measure(0)
            qp.cond(m, qp.X)(0)
            return qp.sample()

        f()
        expected = f._tape  # pylint: disable=protected-access
        tape = qp.flatten(qjit(f))()

        assert len(tape.operations) == len(expected.operations) == 3
        h, mcm, cond = tape.operations
        assert qp.equal(h, qp.H(0))
        assert isinstance(mcm, MidMeasure)
        assert mcm.wires == qp.wires.Wires(0) and mcm.postselect is None and not mcm.reset
        assert isinstance(cond, qp.ops.Conditional)
        assert qp.equal(cond.base, qp.X(0))
        assert isinstance(cond.meas_val, MeasurementValue)
        assert cond.meas_val.measurements == [mcm]
        assert not cond.meas_val.has_processing  # the raw outcome, like qp.cond(m, ...)
        assert tape.measurements == expected.measurements == [qp.sample()]
        assert tape.shots == expected.shots == qp.measurements.Shots(10)


class TestStructure:
    """The content of the tape."""

    def test_operator2_instructions(self):
        """Gates are PennyLane2 (Operator2) instructions."""

        @qjit
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def f():
            qp.RX(0.1, 0)
            qp.CNOT([0, 1])
            qp.measure(1)
            return qp.expval(qp.Z(0))

        tape = qp.flatten(f)()
        assert all(isinstance(op, Operator2) for op in tape.operations)

    def test_execution_config_not_in_tape(self):
        """The MCM method, the differentiation method, etc. are not part of the tape."""

        @qjit
        @qp.set_shots(100)
        @qp.qnode(
            qp.device("lightning.qubit", wires=2), mcm_method="one-shot", diff_method="parameter-shift"
        )
        def f():
            qp.H(0)
            m = qp.measure(0)
            qp.cond(m, qp.X)(1)
            return qp.expval(qp.Z(1))

        tape = qp.flatten(f)()
        # The one-shot transform (a loop over shots) is not applied to the tape
        assert [type(op) for op in tape.operations] == [
            type(qp.H(0)),
            MidMeasure,
            qp.ops.Conditional,
        ]
        assert tape.shots.total_shots == 100
        for attr in ("mcm_method", "diff_method", "gradient_method", "execution_config"):
            assert not hasattr(tape, attr)

    @pytest.mark.parametrize("shots", [None, 7])
    def test_shots_from_device(self, shots):
        """Shots come from the device or set_shots."""

        @qjit
        @qp.qnode(qp.device("lightning.qubit", wires=1), shots=shots)
        def f():
            qp.H(0)
            return qp.probs() if shots is None else qp.sample()

        assert qp.flatten(f)().shots == qp.measurements.Shots(shots)

    def test_measurements(self):
        """Terminal measurements of all kinds are converted."""

        @qjit
        @qp.set_shots(10)
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def f():
            qp.H(0)
            return (
                qp.expval(qp.Z(0) @ qp.X(1)),
                qp.var(qp.Y(2)),
                qp.probs(wires=[0, 1]),
                qp.sample(wires=[2]),
                qp.counts(wires=[0]),
                qp.expval(qp.Hamiltonian([0.5, 0.2], [qp.Z(0), qp.X(1) @ qp.Y(2)])),
                qp.expval(qp.Hermitian(np.eye(2, dtype=complex), wires=1)),
            )

        tape = qp.flatten(f)()
        expected = [
            qp.expval(qp.Z(0) @ qp.X(1)),
            qp.var(qp.Y(2)),
            qp.probs(wires=[0, 1]),
            qp.sample(wires=[2]),
            qp.counts(wires=[0]),
            qp.expval(qp.Hamiltonian([0.5, 0.2], [qp.Z(0), qp.X(1) @ qp.Y(2)])),
            qp.expval(qp.Hermitian(np.eye(2, dtype=complex), wires=1)),
        ]
        for mp, exp in zip(tape.measurements, expected, strict=True):
            assert qp.equal(mp, exp), f"{mp} != {exp}"

    def test_gates_and_modifiers(self):
        """Parametric, controlled, adjoint and matrix gates are converted."""
        u = np.array([[0, 1], [1, 0]], dtype=complex)

        @qjit
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def f():
            qp.Rot(0.1, 0.2, 0.3, 0)
            qp.adjoint(qp.S(0))
            qp.ctrl(qp.RX(0.3, 2), control=[0, 1], control_values=[1, 0])
            qp.ctrl(qp.X(2), control=0)
            qp.QubitUnitary(u, 1)
            qp.MultiRZ(0.4, wires=[0, 1])
            qp.PauliRot(0.5, "XY", wires=[0, 2])
            qp.IsingXX(0.6, wires=[1, 2])
            return qp.state()

        tape = qp.flatten(f)()
        assert_ops_equal(
            tape.operations,
            [
                qp.Rot(0.1, 0.2, 0.3, 0),
                qp.adjoint(qp.S(0)),
                qp.ctrl(qp.RX(0.3, 2), control=[0, 1], control_values=[1, 0]),
                qp.CNOT([0, 2]),
                qp.QubitUnitary(u, 1),
                qp.MultiRZ(0.4, wires=[0, 1]),
                qp.PauliRot(0.5, "XY", wires=[0, 2]),
                qp.IsingXX(0.6, wires=[1, 2]),
            ],
        )

    def test_state_preparation(self):
        """State preparations keep their data."""
        state = np.array([1, 0, 0, 1]) / np.sqrt(2)

        @qjit
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def f():
            qp.StatePrep(state, wires=[0, 1])
            qp.BasisState(np.array([1]), wires=[0])
            return qp.state()

        ops = qp.flatten(f)().operations
        assert isinstance(ops[0], qp.StatePrep)
        assert np.allclose(ops[0].data[0], state)
        assert qp.equal(ops[1], qp.BasisState(np.array([1]), wires=[0]))

    def test_mcm_options_and_branches(self):
        """reset, postselect, else branches and multi-MCM predicates."""

        @qjit
        @qp.set_shots(100)
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def f():
            qp.H(0)
            m0 = qp.measure(0)
            m1 = qp.measure(1, reset=True)
            qp.measure(2, postselect=1)
            qp.cond(m0 & m1, qp.X, qp.Z)(1)
            return qp.expval(qp.Z(0))

        ops = qp.flatten(f)().operations
        assert [type(op).__name__ for op in ops] == [
            "Hadamard",
            "MidMeasure",
            "MidMeasure",
            "MidMeasure",
            "Conditional",
            "Conditional",
        ]
        m0, m1, m2 = ops[1:4]
        assert not m0.reset and m1.reset and m2.postselect == 1
        true_branch, false_branch = ops[4:]
        assert qp.equal(true_branch.base, qp.X(1)) and qp.equal(false_branch.base, qp.Z(1))
        # MeasurementValue orders its measurements by their (random) meas_uid
        measured = sorted(true_branch.meas_val.measurements, key=lambda m: m.wires[0])
        assert measured == [m0, m1]
        for bits, value in true_branch.meas_val.items():
            assert bool(value) == (bits == (1, 1))
        for bits, value in false_branch.meas_val.items():
            assert bool(value) == (bits != (1, 1))


class TestStaticArguments:
    """Arguments are compile-time constants."""

    def test_positional_keyword_and_array_arguments(self):
        """Scalars, arrays and keyword arguments become static; loops depending on them are
        unrolled and branches on them are resolved."""

        @qjit
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def f(x, arr, n, flag=False):
            qp.RX(x, 0)
            qp.RY(arr[1] * 2, 1)

            @qp.for_loop(0, n)
            def loop(i):
                qp.RZ(arr[i], 2)

            loop()

            @qp.while_loop(lambda i: i < 2)
            def wloop(i):
                qp.T(i)
                return i + 1

            wloop(0)
            qp.cond(flag, qp.S)(0)
            qp.cond(x > 0.5, qp.X, qp.Y)(1)
            return qp.expval(qp.Z(0))

        tape = qp.flatten(f)(0.7, np.array([0.1, 0.2, 0.3]), 3, flag=True)
        assert_ops_equal(
            tape.operations,
            [
                qp.RX(0.7, 0),
                qp.RY(0.4, 1),
                qp.RZ(0.1, 2),
                qp.RZ(0.2, 2),
                qp.RZ(0.3, 2),
                qp.T(0),
                qp.T(1),
                qp.S(0),
                qp.X(1),
            ],
        )
        tape = qp.flatten(f)(0.2, np.array([0.1, 0.2, 0.3]), 1)
        assert_ops_equal(
            tape.operations,
            [qp.RX(0.2, 0), qp.RY(0.4, 1), qp.RZ(0.1, 2), qp.T(0), qp.T(1), qp.Y(1)],
        )

    def test_qjit_object_is_not_modified(self):
        """flatten does not change the options of the QJIT object, which can still be called
        with dynamic arguments."""

        @qjit
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def f(x):
            qp.RX(x, 0)
            return qp.expval(qp.Z(0))

        qp.flatten(f)(0.3)
        assert not f.compile_options.static_argnums
        assert np.allclose(f(0.3), np.cos(0.3))
        assert np.allclose(f(0.4), np.cos(0.4))

    def test_classical_processing(self):
        """Classical processing around the QNode is evaluated at compile time."""

        @qjit
        def f(x):
            @qp.qnode(qp.device("lightning.qubit", wires=1))
            def circuit(y):
                qp.RX(y, 0)
                return qp.expval(qp.Z(0))

            return circuit(qp.math.sin(x) * 2) + 1

        assert_ops_equal(qp.flatten(f)(0.5).operations, [qp.RX(np.sin(0.5) * 2, 0)])


class TestErrors:
    """Informative errors."""

    def test_not_qjit(self):
        """Only QJIT objects are supported."""
        with pytest.raises(TypeError, match="decorated with qjit"):
            qp.flatten(lambda: None)

    def test_mcm_dependent_loop_bound(self):
        """Classical dynamism that cannot be resolved at compile time raises an error."""

        @qjit
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def f():
            m = qp.measure(0)

            @qp.for_loop(0, m + 1)
            def loop(i):
                qp.X(1)

            loop()
            return qp.expval(qp.Z(0))

        with pytest.raises(FlattenError, match="for loop bound depends on a mid-circuit"):
            qp.flatten(f)()

    def test_mcm_dependent_parameter(self):
        """Gate parameters computed from MCM outcomes cannot be represented."""

        @qjit
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def f():
            m = qp.measure(0)
            qp.RX(m * 0.1, 1)
            return qp.expval(qp.Z(0))

        with pytest.raises(FlattenError, match="parameter of RX depends on a mid-circuit"):
            qp.flatten(f)()

    def test_dialect_without_pennylane_analogue(self):
        """Compiling to other dialects (e.g. pbc with to_ppr) raises an informative error."""

        @qjit
        @qp.transforms.to_ppr
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def f():
            qp.H(0)
            qp.CNOT([0, 1])
            qp.T(1)
            return qp.expval(qp.Z(0))

        with pytest.raises(FlattenError, match="'pbc.ppr', which has no PennyLane analogue"):
            qp.flatten(f)()


class TestPennyLaneClassicCompatibility:
    """The tape can be compiled, executed and drawn with the tape-based PennyLane pathway."""

    @pytest.fixture
    def tape(self):
        """A compiled tape."""

        @qjit
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def f(x):
            qp.H(0)
            qp.H(0)
            qp.RX(x, 1)
            qp.RX(0.2, 1)
            qp.CNOT([1, 2])
            qp.adjoint(qp.S(2))
            qp.ctrl(qp.RY(0.3, 0), control=1)
            return qp.expval(qp.Z(0) @ qp.Z(2)), qp.probs(wires=[1])

        return qp.flatten(f)(0.1), f(0.1)

    def test_tape_transforms(self, tape):
        """PennyLane tape transforms can be applied."""
        tape, _ = tape
        (t1,), _ = qp.transforms.cancel_inverses(tape)
        (t2,), _ = qp.transforms.merge_rotations(t1)
        assert qp.equal(t2.operations[0], qp.RX(0.3, 1), atol=1e-12)
        (t3,), _ = qp.transforms.decompose(t2, gate_set={"RX", "RY", "RZ", "CNOT", "GlobalPhase"})
        assert {op.name for op in t3.operations} <= {"RX", "RY", "RZ", "CNOT", "GlobalPhase"}

    @pytest.mark.parametrize("device_name", ["default.qubit", "lightning.qubit"])
    def test_execute(self, tape, device_name):
        """The tape can be executed on PennyLane devices, with the same results as qjit."""
        tape, expected = tape
        (res,) = qp.execute([tape], qp.device(device_name, wires=3))
        assert np.allclose(res[0], expected[0])
        assert np.allclose(res[1], expected[1])

    def test_execute_mcm_tape(self):
        """A tape with MCMs can be executed with PennyLane's MCM methods."""

        @qjit
        @qp.set_shots(50)
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def f():
            qp.H(0)
            m = qp.measure(0)
            qp.cond(m, qp.X)(1)
            return qp.sample(wires=[0, 1])

        tape = qp.flatten(f)()
        for mcm_method in ("deferred", "one-shot"):
            (samples,) = qp.execute([tape], qp.device("default.qubit"), mcm_method=mcm_method)
            assert samples.shape == (50, 2)
            assert np.all(samples[:, 0] == samples[:, 1])

    def test_draw(self, tape):
        """The tape can be drawn."""
        tape, _ = tape
        text = qp.drawer.tape_text(tape, decimals=1)
        assert "RX(0.1)" in text and "RX(0.2)" in text and "S†" in text

    def test_draw_mpl(self, tape):
        """The tape can be drawn with matplotlib."""
        pytest.importorskip("matplotlib")
        tape, _ = tape
        fig, _ = qp.drawer.tape_mpl(tape)
        assert fig is not None

    def test_not_compilable_with_qjit(self, tape):
        """The tape is a one-way exit: it cannot be compiled with qjit."""
        tape, _ = tape
        with pytest.raises(TypeError, match="qjit cannot compile a tape"):
            qp.qjit(tape)
