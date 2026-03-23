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
"""Unit and integration tests for the unified compiler `measurements_from_samples` transform."""
# pylint: disable=line-too-long

from functools import partial

import numpy as np
import pennylane as qml
import pytest
from pennylane.exceptions import CompileError

from catalyst.python_interface.transforms import (
    measurements_from_samples_pass,
)

pytestmark = pytest.mark.xdsl


@pytest.mark.usefixtures("use_capture")
class TestIntegrationUsefulErrors:
    """Tests that useful error messages are raised in the frontend for unsupported behaviour"""

    def test_no_shots_raises_error(self):
        """Test that when no shots are provided, the pass raises an error"""

        @qml.qjit
        @measurements_from_samples_pass
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        with pytest.raises(
            ValueError, match="measurements_from_samples pass requires non-zero shots"
        ):
            circuit(1.2)

    def test_dynamic_shots_raises_error(self):
        """Test that when dynamic shots are provided, the pass raises an error"""

        @qml.qjit
        @measurements_from_samples_pass
        def workflow(a, shots):

            @qml.set_shots(shots)
            @qml.qnode(qml.device("lightning.qubit", wires=1))
            def circuit(x):
                qml.RX(x, 0)
                return qml.expval(qml.Z(0))

            circuit(a)

        with pytest.raises(CompileError, match="using a dynamic number of shots is not supported"):
            workflow(1.2, 100)

    def test_counts_raises_not_implemented(self):
        """Test that a circuit with counts causes measurements_from_samples_pass
        to raise a NotImplementedError"""

        dev = qml.device("lightning.qubit", wires=4)

        with pytest.raises(NotImplementedError, match="operations are not supported"):

            @qml.qjit
            @measurements_from_samples_pass
            @qml.set_shots(1000)
            @qml.qnode(dev)
            def circuit(theta: float):
                qml.RX(theta, 0)
                return qml.counts()


    @pytest.mark.parametrize("mp", (qml.expval, qml.var))
    def test_overlapping_tensor(self, mp):
        """Check that an error is raised if the circuit returns a tensor with overlapping wires."""

        # Note: This error is raised by the diagonalize pass that measurements_from_samples
        # calls, not by measurements_from_samples directly. However, the logic in this pass
        # relies on the validation being performed, so its tested here. If this test ever breaks
        # because of changes in diagonalize_measurements, the logic in measurements_from_samples
        # should be re-evaluated.

        dev = qml.device("lightning.qubit", wires=2)

        with pytest.raises(CompileError, match="Observables are not qubit-wise commuting"):

            @qml.qjit
            @measurements_from_samples_pass
            @qml.qnode(dev, shots=1000)
            def circuit():
                return mp(qml.Z(0) @ qml.X(0))

    @pytest.mark.parametrize("mp", (qml.expval, qml.var))
    def test_overlapping_sum(self, mp):
        """Check that an error is raised if the circuit returns a sum with overlapping wires."""

        # Note: This error is raised by the diagonalize pass that measurements_from_samples
        # calls, not by measurements_from_samples directly. However, the logic in this pass
        # relies on the validation being performed, so its tested here. If this test ever breaks
        # because of changes in diagonalize_measurements, the logic in measurements_from_samples
        # should be re-evaluated.

        dev = qml.device("lightning.qubit", wires=2)

        with pytest.raises(CompileError, match="Observables are not qubit-wise commuting"):

            @qml.qjit
            @measurements_from_samples_pass
            @qml.qnode(dev, shots=1000)
            def circuit():
                return mp(2 * qml.Z(0) + qml.X(0))

    @pytest.mark.parametrize("mp", (qml.expval, qml.var))
    def test_overlapping_mps(self, mp):
        """Check that an error is raised if the circuit returns different mps
        containing observables with overlapping wires."""

        # Note: This error is raised by the diagonalize pass that measurements_from_samples
        # calls, not by measurements_from_samples directly. However, the logic in this pass
        # relies on the validation being performed, so its tested here. If this test ever breaks
        # because of changes in diagonalize_measurements, the logic in measurements_from_samples
        # should be re-evaluated.

        dev = qml.device("lightning.qubit", wires=2)

        with pytest.raises(CompileError, match="Observables are not qubit-wise commuting"):

            @qml.qjit
            @measurements_from_samples_pass
            @qml.qnode(dev, shots=1000)
            def circuit():
                return mp(qml.Z(0)), mp(qml.X(0))

    def test_overlapping_obs_and_sample(self):
        """Check that an error is raised if the circuit returns an mp with an observable that
        overlaps with an mp in the computational basis."""

        # Note: This error is raised by the diagonalize pass that measurements_from_samples
        # calls, not by measurements_from_samples directly. However, the logic in this pass
        # relies on the validation being performed, so its tested here. If this test ever breaks
        # because of changes in diagonalize_measurements, the logic in measurements_from_samples
        # should be re-evaluated.

        dev = qml.device("lightning.qubit", wires=2)

        with pytest.raises(CompileError, match="Observables are not qubit-wise commuting"):

            @qml.qjit
            @measurements_from_samples_pass
            @qml.qnode(dev, shots=1000)
            def circuit():
                return qml.sample(wires=[0]), qml.expval(qml.X(0))


@pytest.mark.usefixtures("use_capture")
class TestIntegrationWithOtherPasses:
    """Tests the integration of the xDSL-basd MeasurementsFromSamplesPass with other key passes"""

    @pytest.mark.usefixtures("use_capture")
    def test_integrate_with_decompose(self):
        """Test that the measurements_from_samples pass works correctly when used in combination
        with the decompose pass."""
        dev = qml.device("null.qubit", wires=4)

        @qml.qjit(target="mlir")
        @measurements_from_samples_pass
        @partial(
            qml.transforms.decompose,
            gate_set={"X", "Y", "Z", "S", "H", "CNOT", "RZ", "RY", "GlobalPhase"},
        )
        @qml.qnode(dev, shots=1000)
        def circuit():
            qml.CRX(0.1, wires=[0, 1])
            return qml.expval(qml.Z(0))

        res = circuit()
        assert res == 1.0

    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_ops, expected_res",
        [
            ((qml.I, qml.I), 2.0),
            ((qml.I, qml.X), 2.0),
            ((qml.X, qml.I), -2.0),
            ((qml.X, qml.X), -2.0),
        ],
    )
    def test_expval_sprod_with_split_non_commuting(self, shots, initial_ops, expected_res):
        """Test the measurements_from_samples transform on a device with two wires and terminal
        measurements that require an observable (i.e. expval and var).

        In this test, the terminal measurements are performed on the combination of both wires.
        """

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit
        @qml.transform(pass_name="measurements-from-samples")
        @qml.transform(pass_name="split-non-commuting")
        @qml.qnode(dev, shots=shots)
        def circuit():
            initial_ops[0](wires=0)
            initial_ops[1](wires=1)
            return qml.expval(2 * qml.Z(wires=0))

        assert expected_res == circuit()
    @pytest.mark.usefixtures("use_capture")
    def test_integrate_with_diagonalize(self):
        """Test that the measurements_from_samples pass works correctly when used in combination
        with the diagonalize-measurements pass."""

        dev = qml.device("lightning.qubit", wires=4)

        @qml.qjit
        @measurements_from_samples_pass
        @qml.transform(pass_name="diagonalize-final-measurements")
        @qml.qnode(dev, shots=3000)
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Y(0))

        res = circuit(0.768)
        assert np.isclose(res, -np.sin(0.768), atol=0.05)


@pytest.mark.usefixtures("use_capture")
class TestMeasurementsFromSamplesIntegration:
    """Tests of the execution of simple workloads with the xDSL-based MeasurementsFromSamplesPass
    transform and compare to expected results. The run_filecheck function is used to verify that the
    expected changes to the IR were applied, as a sanity check.
    """

    @pytest.mark.parametrize(
        "transform",
        [measurements_from_samples_pass, qml.transform(pass_name="measurements-from-samples")],
    )
    def test_qjit_filecheck(self, transform, run_filecheck_qjit):
        """Test that the measurements_from_samples_pass works correctly with qjit."""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit(target="mlir")
        @transform
        @qml.qnode(dev, shots=25)
        def circuit():
            # CHECK: [[samples:%.+]] = func.call @circuit.from_samples() : () -> tensor<25x1xf64>
            # CHECK: func.call @expval_from_samples.tensor.25x1xf64([[samples]]) :
            # CHECK-SAME: (tensor<25x1xf64>) -> tensor<f64>
            # CHECK-NOT: quantum.namedobs
            # CHECK: [[obs:%.+]] = quantum.compbasis
            # CHECK: [[samples:%.+]] = quantum.sample [[obs]] : tensor<25x1xf64>
            # CHECK-NOT: quantum.expval
            return qml.expval(qml.Z(wires=0))

        run_filecheck_qjit(circuit)

    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_op, mp, obs, expected_res",
        [
            # PauliZ observables
            (qml.I, qml.expval, qml.Z, 1.0),
            (qml.X, qml.expval, qml.Z, -1.0),
            (qml.I, qml.var, qml.Z, 0.0),
            (qml.X, qml.var, qml.Z, 0.0),
            # PauliX observables
            pytest.param(
                partial(qml.RY, phi=np.pi / 2),
                qml.expval,
                qml.X,
                1.0,
            ),
            pytest.param(
                partial(qml.RY, phi=-np.pi / 2),
                qml.expval,
                qml.X,
                -1.0,
            ),
            pytest.param(
                partial(qml.RY, phi=np.pi / 2),
                qml.var,
                qml.X,
                0.0,
            ),
            pytest.param(
                partial(qml.RY, phi=-np.pi / 2),
                qml.var,
                qml.X,
                0.0,
            ),
            # PauliY observables
            pytest.param(
                partial(qml.RX, phi=-np.pi / 2),
                qml.expval,
                qml.Y,
                1.0,
            ),
            pytest.param(
                partial(qml.RX, phi=np.pi / 2),
                qml.expval,
                qml.Y,
                -1.0,
            ),
            pytest.param(
                partial(qml.RX, phi=-np.pi / 2),
                qml.var,
                qml.Y,
                0.0,
            ),
            pytest.param(
                partial(qml.RX, phi=np.pi / 2),
                qml.var,
                qml.Y,
                0.0,
            ),
        ],
    )
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def test_exec_1_wire_mp_with_obs(self, shots, initial_op, mp, obs, expected_res, run_filecheck_qjit):
        """Test the measurements_from_samples transform on a device with a single wire and terminal
        measurements that require an observable (i.e. expval and var).
        """

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_op(wires=0)
            # CHECK-NOT: quantum.namedobs
            # CHECK-NOT: quantum.var
            # CHECK-NOT: quantum.expval
            # CHECK: quantum.sample
            return mp(obs(wires=0))

        assert expected_res == circuit_ref(), "Sanity check failed, is expected_res correct?"
        circuit_compiled = qml.qjit(
            measurements_from_samples_pass(circuit_ref),
        )

        run_filecheck_qjit(circuit_compiled)

        assert expected_res == circuit_compiled()

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_op, expected_res",
        [
            (qml.I, [1.0, 0.0]),
            (qml.X, [0.0, 1.0]),
        ],
    )
    def test_exec_1_wire_probs(self, shots, initial_op, expected_res, run_filecheck_qjit):
        """Test the measurements_from_samples transform on a device with a single wire and terminal
        probs measurements.
        """

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_op(wires=0)
            # CHECK-NOT: quantum.probs
            # CHECK: quantum.sample
            return qml.probs(wires=0)

        assert np.array_equal(
            expected_res, circuit_ref()
        ), "Sanity check failed, is expected_res correct?"
        circuit_compiled = qml.qjit(
            measurements_from_samples_pass(circuit_ref),
        )

        run_filecheck_qjit(circuit_compiled)

        assert np.array_equal(expected_res, circuit_compiled())

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.xfail(
        reason="Counts not supported in Catalyst with program capture",
        strict=True,
        raises=NotImplementedError,
    )
    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_op, expected_res",
        [
            (qml.I, {"0": 10, "1": 0}),
            (qml.X, {"0": 0, "1": 10}),
        ],
    )
    def test_exec_1_wire_counts(self, shots, initial_op, expected_res):
        """Test the measurements_from_samples transform on a device with a single wire and terminal
        counts measurements.
        """

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_op(wires=0)
            return qml.counts(wires=0, all_outcomes=True)

        assert np.array_equal(
            expected_res, circuit_ref()
        ), "Sanity check failed, is expected_res correct?"

        circuit_compiled = qml.qjit(
            measurements_from_samples_pass(circuit_ref),
        )

        assert np.array_equal(expected_res, _counts_catalyst_to_pl(*circuit_compiled()))

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_op, expected_res_base",
        [
            (qml.I, 0),
            (qml.X, 1),
        ],
    )
    def test_exec_1_wire_sample(self, shots, initial_op, expected_res_base):
        """Test the measurements_from_samples transform on a device with a single wire and terminal
        sample measurements.

        In this case, the measurements_from_samples pass should effectively be a no-op.
        """
        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_op(wires=0)
            return qml.sample(wires=0)

        circuit_compiled = qml.qjit(
            measurements_from_samples_pass(circuit_ref),
        )

        expected_res = expected_res_base * np.ones(shape=(shots, 1), dtype=int)

        assert np.array_equal(expected_res, circuit_compiled())

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_ops, mp, obs, expected_res",
        [
            ((qml.I, qml.I), qml.expval, qml.Z, (1.0, 1.0)),
            ((qml.I, qml.X), qml.expval, qml.Z, (1.0, -1.0)),
            ((qml.X, qml.I), qml.expval, qml.Z, (-1.0, 1.0)),
            ((qml.X, qml.X), qml.expval, qml.Z, (-1.0, -1.0)),
            ((qml.I, qml.I), qml.var, qml.Z, (0.0, 0.0)),
            ((qml.I, qml.X), qml.var, qml.Z, (0.0, 0.0)),
            ((qml.X, qml.I), qml.var, qml.Z, (0.0, 0.0)),
            ((qml.X, qml.X), qml.var, qml.Z, (0.0, 0.0)),
        ],
    )
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def test_exec_2_wire_with_obs_separate(self, shots, initial_ops, mp, obs, expected_res, run_filecheck_qjit):
        """Test the measurements_from_samples transform on a device with two wires and terminal
        measurements that require an observable (i.e. expval and var).

        In this test, the terminal measurements are performed separately per wire.
        """

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_ops[0](wires=0)
            initial_ops[1](wires=1)
            # CHECK-NOT: quantum.namedobs
            # CHECK-NOT: quantum.var
            # CHECK-NOT: quantum.expval
            # CHECK: quantum.sample
            return mp(obs(wires=0)), mp(obs(wires=1))

        assert expected_res == circuit_ref(), "Sanity check failed, is expected_res correct?"
        circuit_compiled = qml.qjit(
            measurements_from_samples_pass(circuit_ref),
        )

        run_filecheck_qjit(circuit_compiled)

        assert expected_res == circuit_compiled()

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.xfail(
        reason="Operator arithmetic not yet supported with capture enabled", strict=True
    )
    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_ops, mp, expected_res",
        [
            ((qml.I, qml.I), qml.expval, 1.0),
            ((qml.I, qml.X), qml.expval, -1.0),
            ((qml.X, qml.I), qml.expval, -1.0),
            ((qml.X, qml.X), qml.expval, 1.0),
            ((qml.I, qml.I), qml.var, 0.0),
            ((qml.I, qml.X), qml.var, 0.0),
            ((qml.X, qml.I), qml.var, 0.0),
            ((qml.X, qml.X), qml.var, 0.0),
        ],
    )
    def test_exec_2_wire_with_obs_combined(self, shots, initial_ops, mp, expected_res):
        """Test the measurements_from_samples transform on a device with two wires and terminal
        measurements that require an observable (i.e. expval and var).

        In this test, the terminal measurements are performed on the combination of both wires.
        """

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_ops[0](wires=0)
            initial_ops[1](wires=1)
            return mp(qml.Z(wires=0) @ qml.Z(wires=1))

        assert expected_res == circuit_ref(), "Sanity check failed, is expected_res correct?"

        circuit_compiled = qml.qjit(measurements_from_samples_pass(circuit_ref))

        assert expected_res == circuit_compiled()

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_ops, expected_res",
        [
            ((qml.I, qml.I), [1.0, 0.0, 0.0, 0.0]),
            ((qml.I, qml.X), [0.0, 1.0, 0.0, 0.0]),
            ((qml.X, qml.I), [0.0, 0.0, 1.0, 0.0]),
            ((qml.X, qml.X), [0.0, 0.0, 0.0, 1.0]),
        ],
    )
    def test_exec_2_wire_probs_global(self, shots, initial_ops, expected_res, run_filecheck_qjit):
        """Test the measurements_from_samples transform on a device with two wires and a terminal,
        "global" probs measurements (one that implicitly acts on all wires).
        """
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_ops[0](wires=0)
            initial_ops[1](wires=1)
            # CHECK-NOT: quantum.probs
            # CHECK: quantum.sample
            return qml.probs()

        assert np.array_equal(
            expected_res, circuit_ref()
        ), "Sanity check failed, is expected_res correct?"
        circuit_compiled = qml.qjit(measurements_from_samples_pass(circuit_ref))

        run_filecheck_qjit(circuit_compiled)

        assert np.array_equal(expected_res, circuit_compiled())

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_ops, expected_res",
        [
            ((qml.I, qml.I), ([1.0, 0.0], [1.0, 0.0])),
            ((qml.I, qml.X), ([1.0, 0.0], [0.0, 1.0])),
            ((qml.X, qml.I), ([0.0, 1.0], [1.0, 0.0])),
            ((qml.X, qml.X), ([0.0, 1.0], [0.0, 1.0])),
        ],
    )
    def test_exec_2_wire_probs_per_wire(self, shots, initial_ops, expected_res, run_filecheck_qjit):
        """Test the measurements_from_samples transform on a device with two wires and a terminal,
        "global" probs measurements (one that implicitly acts on all wires).
        """
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev, shots=shots)
        def circuit_ref():
            initial_ops[0](wires=0)
            initial_ops[1](wires=1)
            # CHECK-NOT: quantum.probs
            # CHECK: quantum.sample
            return qml.probs(wires=0), qml.probs(wires=1)

        assert np.array_equal(
            expected_res, circuit_ref()
        ), "Sanity check failed, is expected_res correct?"
        circuit_compiled = qml.qjit(measurements_from_samples_pass(circuit_ref))

        run_filecheck_qjit(circuit_compiled)

        assert np.array_equal(expected_res, circuit_compiled())

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.xfail(reason="Dynamic shots not supported")
    def test_exec_expval_dynamic_shots(self):
        """Test the measurements_from_samples transform where the number of shots is dynamic.

        This use case is not currently supported.
        """

        @qml.qjit
        def workload(shots):
            dev = qml.device("lightning.qubit", wires=1)

            @measurements_from_samples_pass
            @qml.qnode(dev, shots=shots)
            def circuit():
                return qml.expval(qml.Z(wires=0))

            return circuit()

        result = workload(2)
        assert result == 1.0


def _counts_catalyst_to_pl(basis_states, counts):
    """Helper function to convert counts in the Catalyst format to the PennyLane format.

    Example:

    >>> basis_states, counts = ([0, 1], [6, 4])
    >>> _counts_catalyst_to_pl(basis_states, counts)
    {'0': 6, '1': 4}
    """
    return {format(int(state), "01b"): count for state, count in zip(basis_states, counts)}


if __name__ == "__main__":
    pytest.main(["-x", __file__])
