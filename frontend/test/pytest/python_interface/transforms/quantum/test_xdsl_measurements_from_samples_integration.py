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
import pennylane as qp
import pytest
from pennylane.exceptions import CompileError

from catalyst.python_interface.transforms import (
    measurements_from_samples_pass,
)

pytestmark = pytest.mark.xdsl


@pytest.mark.parametrize("capture", [True, False])
class TestIntegrationUsefulErrors:
    """Tests that useful error messages are raised in the frontend for unsupported behaviour"""

    def test_no_shots_raises_error(self, capture):
        """Test that when no shots are provided, the pass raises an error"""

        @qp.qjit(capture=capture)
        @measurements_from_samples_pass
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit(x):
            qp.RX(x, 0)
            return qp.expval(qp.Z(0))

        with pytest.raises(
            ValueError, match="measurements_from_samples pass requires non-zero shots"
        ):
            circuit(1.2)

    def test_dynamic_shots_raises_error(self, capture):
        """Test that when dynamic shots are provided, the pass raises an error"""

        if capture is False:
            pytest.xfail(
                reason="passes applied to workflows raise an error without program capture"
            )

        @qp.qjit(capture=capture)
        @measurements_from_samples_pass
        def workflow(a, shots):

            @qp.set_shots(shots)
            @qp.qnode(qp.device("lightning.qubit", wires=1))
            def circuit(x):
                qp.RX(x, 0)
                return qp.expval(qp.Z(0))

            circuit(a)

        with pytest.raises(CompileError, match="using a dynamic number of shots is not supported"):
            workflow(1.2, 100)

    def test_counts_raises_not_implemented(self, capture):
        """Test that a circuit with counts causes measurements_from_samples_pass
        to raise a NotImplementedError"""

        dev = qp.device("lightning.qubit", wires=4)

        with pytest.raises(NotImplementedError, match="operations are not supported"):

            @qp.qjit(capture=capture)
            @measurements_from_samples_pass
            @qp.set_shots(1000)
            @qp.qnode(dev)
            def circuit(theta: float):
                qp.RX(theta, 0)
                return qp.counts()

    @pytest.mark.parametrize("mp", (qp.expval, qp.var))
    def test_overlapping_tensor(self, mp, capture):
        """Check that an error is raised if the circuit returns a tensor with overlapping wires."""

        # Note: This error is raised by the diagonalize pass that measurements_from_samples
        # calls, not by measurements_from_samples directly. However, the logic in this pass
        # relies on the validation being performed, so it's tested here. If this test ever breaks
        # because of changes in diagonalize_measurements, the logic in measurements_from_samples
        # should be re-evaluated.

        dev = qp.device("lightning.qubit", wires=2)

        with pytest.raises(CompileError, match="Observables are not qubit-wise commuting"):

            @qp.qjit(capture=capture)
            @measurements_from_samples_pass
            @qp.qnode(dev, shots=1000)
            def circuit():
                return mp(qp.Z(0) @ qp.X(0))

    @pytest.mark.parametrize("mp", (qp.expval, qp.var))
    def test_overlapping_sum(self, mp, capture):
        """Check that an error is raised if the circuit returns a sum with overlapping wires."""

        # Note: This error is raised by the diagonalize pass that measurements_from_samples
        # calls, not by measurements_from_samples directly. However, the logic in this pass
        # relies on the validation being performed, so its tested here. If this test ever breaks
        # because of changes in diagonalize_measurements, the logic in measurements_from_samples
        # should be re-evaluated.

        dev = qp.device("lightning.qubit", wires=2)

        with pytest.raises(CompileError, match="Observables are not qubit-wise commuting"):

            @qp.qjit(capture=capture)
            @measurements_from_samples_pass
            @qp.qnode(dev, shots=1000)
            def circuit():
                return mp(2 * qp.Z(0) + qp.X(0))

    @pytest.mark.parametrize("mp", (qp.expval, qp.var))
    def test_overlapping_mps(self, mp, capture):
        """Check that an error is raised if the circuit returns different mps
        containing observables with overlapping wires."""

        # Note: This error is raised by the diagonalize pass that measurements_from_samples
        # calls, not by measurements_from_samples directly. However, the logic in this pass
        # relies on the validation being performed, so its tested here. If this test ever breaks
        # because of changes in diagonalize_measurements, the logic in measurements_from_samples
        # should be re-evaluated.

        dev = qp.device("lightning.qubit", wires=2)

        with pytest.raises(CompileError, match="Observables are not qubit-wise commuting"):

            @qp.qjit(capture=capture)
            @measurements_from_samples_pass
            @qp.qnode(dev, shots=1000)
            def circuit():
                return mp(qp.Z(0)), mp(qp.X(0))

    def test_overlapping_obs_and_sample(self, capture):
        """Check that an error is raised if the circuit returns an mp with an observable that
        overlaps with an mp in the computational basis."""

        # Note: This error is raised by the diagonalize pass that measurements_from_samples
        # calls, not by measurements_from_samples directly. However, the logic in this pass
        # relies on the validation being performed, so its tested here. If this test ever breaks
        # because of changes in diagonalize_measurements, the logic in measurements_from_samples
        # should be re-evaluated.

        dev = qp.device("lightning.qubit", wires=2)

        with pytest.raises(CompileError, match="Observables are not qubit-wise commuting"):

            @qp.qjit(capture=capture)
            @measurements_from_samples_pass
            @qp.qnode(dev, shots=1000)
            def circuit():
                return qp.sample(wires=[0]), qp.expval(qp.X(0))

    @pytest.mark.parametrize("obs", (2 * qp.X(0), qp.X(1) + qp.X(2)))
    def test_hamiltonianop_raises_error(self, obs, capture):
        """Test that a circuit with a HamiltonianOp observable raises an error message
        instructing the user to apply `split-non-commuting` first"""

        dev = qp.device("lightning.qubit", wires=2)

        with pytest.raises(CompileError, match="Apply `qp.transforms.split_non_commuting`"):

            @qp.qjit(capture=capture)
            @measurements_from_samples_pass
            @qp.qnode(dev, shots=1000)
            def circuit():
                return qp.expval(obs)


@pytest.mark.parametrize("capture", [True, False])
class TestIntegrationWithOtherPasses:
    """Tests the integration of the xDSL-basd MeasurementsFromSamplesPass with other key passes"""

    def test_integrate_with_decompose(self, capture, run_filecheck_qjit):
        """Test that the measurements_from_samples pass works correctly when used in combination
        with the decompose pass."""
        dev = qp.device("lightning.qubit", wires=4)

        @qp.qjit(target="mlir", capture=capture, seed=12)
        @measurements_from_samples_pass
        @partial(
            qp.transforms.decompose,
            gate_set={"X", "Y", "Z", "S", "H", "CNOT", "RZ", "RY", "GlobalPhase"},
        )
        @qp.qnode(dev, shots=5000)
        def circuit(x):
            # CHECK-NOT: quantum.custom "CRX"
            # CHECK-NOT: quantum.expval
            # CHECK: quantum.sample
            qp.X(0)
            qp.CRX(x, wires=[0, 1])
            return qp.expval(qp.Z(1))

        assert np.isclose(circuit(1.234), np.cos(1.234), atol=0.05)
        run_filecheck_qjit(circuit)

    @pytest.mark.parametrize("coeff", [0.5, 2, -1.7])
    @pytest.mark.parametrize("phi", [0, np.pi, 0.1234, -1.25])
    def test_expval_sprod_with_split_non_commuting(self, coeff, phi, capture, run_filecheck_qjit):
        """Test the measurements_from_samples transform on a device with two wires and terminal
        measurements that require an observable (i.e. expval and var).

        In this test, the terminal measurements are performed on the combination of both wires.
        """

        dev = qp.device("lightning.qubit", wires=2)

        @qp.qnode(dev)
        def circuit():
            qp.RX(phi, 0)
            # CHECK-NOT: quantum.expval
            # CHECK: quantum.sample
            return qp.expval(coeff * qp.Z(0))

        expected_res = coeff * np.cos(phi)
        assert np.allclose(expected_res, circuit()), "Sanity check failed, is expected_res correct?"

        pipeline = qp.CompilePipeline(
            qp.transform(pass_name="split-non-commuting"),
            qp.transform(pass_name="measurements-from-samples"),
        )

        circ = qp.set_shots(circuit, 6000)
        circuit_compiled = qp.qjit(
            pipeline(circ),
            capture=capture,
            seed=34,
        )

        assert np.isclose(expected_res, circuit_compiled(), atol=0.05)

        run_filecheck_qjit(circuit_compiled)

    @pytest.mark.xfail(reason="split-non-commuting doesn't support var", strict=True)
    @pytest.mark.parametrize("coeff", [0.5, 2, -1.7])
    @pytest.mark.parametrize("phi", [0, np.pi, 0.1234, -1.25])
    def test_var_sprod_with_split_non_commuting(self, coeff, phi, capture, run_filecheck_qjit):
        """Test the measurements_from_samples transform with a variance of an SProd. This only
        works with split-non-commuting applied, to remove the HamiltonianOps.

        In this test, the terminal measurements are performed on the combination of both wires.
        """

        def circuit():
            qp.RX(phi, 0)
            # CHECK-NOT: quantum.var
            # CHECK: quantum.sample
            return qp.var(coeff * qp.Z(wires=0))

        # var for the observable is (1-cos(phi)**2), and var scales
        # as Var(a*X) = a^2 * Var(X) for constant a
        expected_res = coeff**2 * (1 - np.cos(phi) ** 2)
        assert np.isclose(expected_res, circuit()), "Sanity check failed, is expected_res correct?"

        pipeline = qp.CompilePipeline(
            qp.transform(pass_name="split-non-commuting"),
            qp.transform(pass_name="measurements-from-samples"),
        )

        circ = qp.set_shots(circuit, 5000)
        circuit_compiled = qp.qjit(
            pipeline(circ),
            capture=capture,
            seed=56,
        )

        assert np.isclose(expected_res, circuit_compiled(), atol=0.05)

        run_filecheck_qjit(circuit_compiled)

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    @pytest.mark.parametrize("coeff", [1.3, -4])
    @pytest.mark.parametrize("phi1, phi2", [(0, 0), (-0.57, 0), (0, 2.34), (-0.57, 2.34)])
    def test_expval_sum_with_split_non_commuting(
        self, coeff, phi1, phi2, capture, run_filecheck_qjit
    ):
        """Test the measurements_from_samples transform on a device with two wires and terminal
        measurements that require an observable (i.e. expval and var).

        In this test, the terminal measurements are performed on the combination of both wires.
        """
        dev = qp.device("lightning.qubit", wires=2)

        @qp.qnode(dev)
        def circuit():
            qp.RX(phi1, wires=0)
            qp.RX(phi2, wires=1)
            # CHECK-NOT: quantum.expval
            # CHECK: quantum.sample
            return qp.expval(coeff * qp.Z(wires=0) + qp.Y(1))

        expected_res = coeff * np.cos(phi1) - np.sin(phi2)
        assert np.isclose(expected_res, circuit()), "Sanity check failed, is expected_res correct?"

        pipeline = qp.CompilePipeline(
            qp.transform(pass_name="split-non-commuting"),
            qp.transform(pass_name="measurements-from-samples"),
        )

        circ = qp.set_shots(circuit, 5000)
        circuit_compiled = qp.qjit(pipeline(circ), capture=capture, seed=78)

        assert np.isclose(expected_res, circuit_compiled(), atol=0.1)
        run_filecheck_qjit(circuit_compiled)

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    @pytest.mark.xfail(reason="split-non-commuting doesn't support var")
    @pytest.mark.parametrize("coeff", [1.3, -4])
    @pytest.mark.parametrize("phi1, phi2", [(0, 0), (-0.57, 0), (0, 2.34), (-0.57, 2.34)])
    def test_var_sum_with_split_non_commuting(self, coeff, phi1, phi2, capture, run_filecheck_qjit):
        """Test the measurements_from_samples transform on a device with two wires and terminal
        measurements that require an observable (i.e. expval and var).

        In this test, the terminal measurements are performed on the combination of both wires.
        """
        dev = qp.device("lightning.qubit", wires=2)

        @qp.qnode(dev)
        def circuit():
            qp.RX(phi1, wires=0)
            qp.RX(phi2, wires=1)
            # CHECK-NOT: quantum.expval
            # CHECK: quantum.sample
            return qp.var(coeff * qp.Z(wires=0) + qp.Y(1))

        expected_res = coeff**2 * (1 - np.cos(phi1) ** 2) + (1 - np.sin(phi2) ** 2)
        assert np.isclose(expected_res, circuit()), "Sanity check failed, is expected_res correct?"

        pipeline = qp.CompilePipeline(
            qp.transform(pass_name="split-non-commuting"),
            qp.transform(pass_name="measurements-from-samples"),
        )
        circ = qp.set_shots(circuit, 5000)
        circuit_compiled = qp.qjit(pipeline(circ), capture=capture, seed=91)

        assert np.isclose(expected_res, circuit_compiled(), atol=0.05)
        run_filecheck_qjit(circuit_compiled)

    def test_integrate_with_diagonalize(self, capture):
        """Test that the measurements_from_samples pass works correctly when used in combination
        with the diagonalize-measurements pass."""

        dev = qp.device("lightning.qubit", wires=4)

        @qp.qjit(capture=capture, seed=23)
        @measurements_from_samples_pass
        @qp.transform(pass_name="diagonalize-final-measurements")
        @qp.qnode(dev, shots=3000)
        def circuit(x):
            qp.RX(x, 0)
            return qp.expval(qp.Y(0))

        phi = 0.768
        res = circuit(phi)
        assert np.isclose(res, -np.sin(phi), atol=0.05)


@pytest.mark.parametrize("capture", [True, False])
class TestMeasurementsFromSamplesIntegration:
    """Tests of the execution of simple workloads with the xDSL-based MeasurementsFromSamplesPass
    transform and compare to expected results. The run_filecheck function is used to verify that the
    expected changes to the IR were applied, as a sanity check.
    """

    @pytest.mark.parametrize(
        "transform",
        [measurements_from_samples_pass, qp.transform(pass_name="measurements-from-samples")],
    )
    def test_qjit_filecheck(self, transform, capture, run_filecheck_qjit):
        """Test that the measurements_from_samples_pass works correctly with qjit."""
        dev = qp.device("lightning.qubit", wires=2)

        @qp.qjit(target="mlir", capture=capture)
        @transform
        @qp.qnode(dev, shots=25)
        def circuit():
            # CHECK: [[samples:%.+]] = func.call @circuit.from_samples() : () -> tensor<25x1xf64>
            # CHECK: func.call @expval_from_samples.tensor.25x1xf64([[samples]]) :
            # CHECK-SAME: (tensor<25x1xf64>) -> tensor<f64>
            # CHECK-NOT: quantum.namedobs
            # CHECK: [[obs:%.+]] = quantum.compbasis
            # CHECK: [[samples:%.+]] = quantum.sample [[obs]] : tensor<25x1xf64>
            # CHECK-NOT: quantum.expval
            return qp.expval(qp.Z(wires=0))

        run_filecheck_qjit(circuit)

    # pylint: disable=unnecessary-lambda
    @pytest.mark.parametrize("phi", [0, np.pi / 3, 0.4568])
    @pytest.mark.parametrize(
        "initial_op, mp, obs, expected_res",
        [
            # PauliZ observables
            pytest.param(
                qp.RX,
                qp.expval,
                qp.Z,
                lambda phi: np.cos(phi),
            ),
            pytest.param(
                qp.RX,
                qp.var,
                qp.Z,
                lambda phi: 1 - np.cos(phi) ** 2,
            ),
            # PauliX observables
            pytest.param(
                qp.RY,
                qp.expval,
                qp.X,
                lambda phi: np.sin(phi),
            ),
            pytest.param(
                qp.RY,
                qp.var,
                qp.X,
                lambda phi: 1 - np.sin(phi) ** 2,
            ),
            # PauliY observables
            pytest.param(
                qp.RX,
                qp.expval,
                qp.Y,
                lambda phi: -np.sin(phi),
            ),
            pytest.param(
                qp.RX,
                qp.var,
                qp.Y,
                lambda phi: 1 - np.sin(phi) ** 2,
            ),
        ],
    )
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def test_exec_1_wire_mp_with_obs(
        self, initial_op, mp, obs, phi, expected_res, capture, run_filecheck_qjit
    ):
        """Test the measurements_from_samples transform on a device with a single wire and terminal
        measurements that require an observable (i.e. expval and var).
        """

        dev = qp.device("lightning.qubit", wires=1)

        @qp.qnode(dev)
        def circuit_ref():
            initial_op(phi, wires=0)
            # CHECK-NOT: quantum.namedobs
            # CHECK-NOT: quantum.var
            # CHECK-NOT: quantum.expval
            # CHECK: quantum.sample
            return mp(obs(wires=0))

        assert np.isclose(
            expected_res(phi), circuit_ref()
        ), "Sanity check failed, is expected_res correct?"

        circ = qp.set_shots(circuit_ref, 5000)
        circuit_compiled = qp.qjit(
            measurements_from_samples_pass(circ),
            capture=capture,
            seed=45,
        )

        run_filecheck_qjit(circuit_compiled)
        assert np.isclose(expected_res(phi), circuit_compiled(), atol=0.05)

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.parametrize("phi, omega", [(0, 0), (np.pi, 0), (0.342, 1.08)])
    def test_exec_1_wire_probs(self, phi, omega, capture, run_filecheck_qjit):
        """Test the measurements_from_samples transform on a device with a single wire and terminal
        probs measurements.
        """

        dev = qp.device("lightning.qubit", wires=1)

        @qp.qnode(dev)
        def circuit_ref():
            qp.RY(phi, wires=0)
            qp.RX(omega, wires=0)
            # CHECK-NOT: quantum.probs
            # CHECK: quantum.sample
            return qp.probs(wires=0)

        expected_res = [
            0.5 * (1 + np.cos(omega) * np.cos(phi)),
            0.5 * (1 - np.cos(omega) * np.cos(phi)),
        ]
        assert np.allclose(
            expected_res, circuit_ref(), atol=0.005
        ), "Sanity check failed, is expected_res correct?"

        circ_shots = qp.set_shots(circuit_ref, 5000)
        circuit_compiled = qp.qjit(
            measurements_from_samples_pass(circ_shots),
            capture=capture,
            seed=67,
        )

        run_filecheck_qjit(circuit_compiled)
        assert np.allclose(expected_res, circuit_compiled(), atol=0.05)

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.parametrize(
        "initial_op, expected_res",
        [
            (qp.I, {"0": 10, "1": 0}),
            (qp.X, {"0": 0, "1": 10}),
        ],
    )
    def test_exec_1_wire_counts(self, initial_op, expected_res, capture):
        """Test the measurements_from_samples transform on a device with a single wire and terminal
        counts measurements.
        """

        dev = qp.device("lightning.qubit", wires=1)

        @qp.qnode(dev, shots=10)
        def circuit_ref():
            initial_op(wires=0)
            return qp.counts(wires=0, all_outcomes=True)

        assert np.array_equal(
            expected_res, circuit_ref()
        ), "Sanity check failed, is expected_res correct?"

        with pytest.raises(NotImplementedError, match="operations are not supported"):
            circuit_compiled = qp.qjit(
                measurements_from_samples_pass(circuit_ref),
                capture=capture,
                seed=89,
            )

            assert np.array_equal(expected_res, _counts_catalyst_to_pl(*circuit_compiled()))

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.parametrize("shots", [1, 2])
    @pytest.mark.parametrize(
        "initial_op, expected_res_base",
        [
            (qp.I, 0),
            (qp.X, 1),
        ],
    )
    def test_exec_1_wire_sample(self, shots, initial_op, expected_res_base, capture):
        """Test the measurements_from_samples transform on a device with a single wire and terminal
        sample measurements.

        In this case, the measurements_from_samples pass should effectively be a no-op.
        """
        dev = qp.device("lightning.qubit", wires=1)

        @qp.qnode(dev, shots=shots)
        def circuit_ref():
            initial_op(wires=0)
            return qp.sample(wires=0)

        circuit_compiled = qp.qjit(
            measurements_from_samples_pass(circuit_ref),
            capture=capture,
            seed=123,
        )

        expected_res = expected_res_base * np.ones(shape=(shots, 1), dtype=int)

        assert np.array_equal(expected_res, circuit_compiled())

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.parametrize(
        "angles", [(0, 0), (0, np.pi), (np.pi, 0), (0.74, 0.123), (-1.23, 0.86)]
    )
    @pytest.mark.parametrize(
        "mp, expected_res",
        [
            (qp.expval, lambda angles: (np.cos(angles[0]), -np.sin(angles[1]))),
            (qp.var, lambda angles: (1 - np.cos(angles[0]) ** 2, 1 - np.sin(angles[1]) ** 2)),
        ],
    )
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def test_exec_2_wire_with_obs_separate(
        self, angles, mp, expected_res, capture, run_filecheck_qjit
    ):
        """Test the measurements_from_samples transform on a device with two wires and terminal
        measurements that require an observable (i.e. expval and var).

        In this test, the terminal measurements are performed separately per wire.
        """

        dev = qp.device("lightning.qubit", wires=2)

        @qp.qnode(dev)
        def circuit_ref():
            qp.RX(phi=angles[0], wires=0)
            qp.RX(phi=angles[1], wires=1)
            # CHECK-NOT: quantum.namedobs
            # CHECK-NOT: quantum.var
            # CHECK-NOT: quantum.expval
            # CHECK: quantum.sample
            return mp(qp.Z(wires=0)), mp(qp.Y(wires=1))

        assert np.allclose(
            expected_res(angles), circuit_ref()
        ), "Sanity check failed, is expected_res correct?"
        circ_shots = qp.set_shots(circuit_ref, 5000)
        circuit_compiled = qp.qjit(
            measurements_from_samples_pass(circ_shots),
            capture=capture,
            seed=234,
        )

        run_filecheck_qjit(circuit_compiled)
        assert np.allclose(expected_res(angles), circuit_compiled(), atol=0.05)

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.parametrize(
        "angles", [(0, 0), (0, np.pi), (np.pi, 0), (0.74, 0.123), (-1.23, 0.86)]
    )
    @pytest.mark.parametrize(
        "mp, expected_res",
        [
            (qp.expval, lambda angles: (np.cos(angles[0]) * -np.sin(angles[1]))),
            (qp.var, lambda angles: ((1 - np.cos(angles[0]) ** 2 * np.sin(angles[1]) ** 2))),
        ],
    )
    def test_exec_2_wire_with_tensor_obs(self, angles, mp, expected_res, capture):
        """Test the measurements_from_samples transform on a device with two wires and terminal
        measurements that require an observable (i.e. expval and var).

        In this test, the terminal measurements are performed on the combination of both wires.
        """

        dev = qp.device("lightning.qubit", wires=2)

        @qp.qnode(dev)
        def circuit_ref():
            qp.RX(phi=angles[0], wires=0)
            qp.RX(phi=angles[1], wires=1)
            return mp(qp.Z(wires=0) @ qp.Y(wires=1))

        assert np.allclose(
            expected_res(angles), circuit_ref()
        ), "Sanity check failed, is expected_res correct?"

        circ_shots = qp.set_shots(circuit_ref, 5000)
        circuit_compiled = qp.qjit(
            measurements_from_samples_pass(circ_shots), capture=capture, seed=456
        )

        assert np.allclose(expected_res(angles), circuit_compiled(), atol=0.05)

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.parametrize("phi", [0, 0.123, -0.784, 1.94])
    def test_exec_2_wire_probs_global(self, phi, capture, run_filecheck_qjit):
        """Test the measurements_from_samples transform on a device with two wires and a terminal,
        "global" probs measurements (one that implicitly acts on all wires).
        """
        dev = qp.device("lightning.qubit", wires=2)

        @qp.qnode(dev)
        def circuit_ref():
            qp.H(wires=0)
            qp.IsingXX(phi, wires=(0, 1))
            # CHECK-NOT: quantum.probs
            # CHECK: quantum.sample
            return qp.probs()

        x1 = np.cos(phi / 2) ** 2 / 2
        x2 = np.sin(phi / 2) ** 2 / 2
        expected_res = np.array([x1, x2, x1, x2])
        assert np.allclose(
            expected_res, circuit_ref()
        ), "Sanity check failed, is expected_res correct?"

        circ_shots = qp.set_shots(circuit_ref, 5000)
        circuit_compiled = qp.qjit(
            measurements_from_samples_pass(circ_shots), capture=capture, seed=567
        )

        run_filecheck_qjit(circuit_compiled)

        assert np.allclose(expected_res, circuit_compiled(), atol=0.05)

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.parametrize("phi", [0, 0.123, -0.784, 1.94])
    def test_exec_2_wire_probs_per_wire(self, phi, capture, run_filecheck_qjit):
        """Test the measurements_from_samples transform on a device with two wires and a terminal,
        "global" probs measurements (one that implicitly acts on all wires).
        """
        dev = qp.device("lightning.qubit", wires=2)

        @qp.qnode(dev)
        def circuit_ref():
            qp.RX(phi, wires=0)
            qp.RX(-phi, wires=1)
            # CHECK-NOT: quantum.probs
            # CHECK: quantum.sample
            return qp.probs(wires=0), qp.probs(wires=1)

        probs1 = np.array([np.cos(phi / 2) ** 2, np.sin(phi / 2) ** 2])
        probs2 = np.array([np.cos(-phi / 2) ** 2, np.sin(-phi / 2) ** 2])

        assert np.allclose(
            [probs1, probs2], circuit_ref()
        ), "Sanity check failed, is expected_res correct?"

        circ_shots = qp.set_shots(circuit_ref, 5000)
        circuit_compiled = qp.qjit(
            measurements_from_samples_pass(circ_shots), capture=capture, seed=678
        )

        run_filecheck_qjit(circuit_compiled)

        assert np.allclose([probs1, probs2], circuit_compiled(), atol=0.05)

    # -------------------------------------------------------------------------------------------- #

    @pytest.mark.parametrize("theta", [0, -1.23, 0.765])
    def test_measurements_from_samples_multiple_measurements(
        self, theta, capture, run_filecheck_qjit
    ):
        """Test the transform measurements_from_samples with multiple measurement types
        as part of the Catalyst pipeline."""

        dev = qp.device("lightning.qubit", wires=4)

        @qp.qjit(capture=capture, seed=789)
        @measurements_from_samples_pass
        @qp.set_shots(5000)
        @qp.qnode(dev)
        def circuit(theta: float):
            qp.RY(theta, 0)
            qp.RY(theta / 2, 1)
            qp.RY(2 * theta, 2)
            qp.RY(theta, 3)
            # CHECK-NOT: quantum.namedobs
            # CHECK-NOT: quantum.expval
            # CHECK-NOT: quantum.var
            # CHECK-NOT: quantum.probs
            # CHECK: quantum.sample
            return (
                qp.expval(qp.PauliX(wires=0) @ qp.PauliX(wires=1)),
                qp.var(qp.PauliX(wires=1)),
                qp.probs(wires=[3]),
            )

        run_filecheck_qjit(circuit)

        expval_res, var_res, probs_res = circuit(theta)

        expval_expected = np.sin(theta) * np.sin(theta / 2)
        var_expected = 1 - np.sin(theta / 2) ** 2
        probs_expected = [np.cos(theta / 2) ** 2, np.sin(theta / 2) ** 2]

        assert np.isclose(expval_res, expval_expected, atol=0.05)
        assert np.isclose(var_res, var_expected, atol=0.05)
        assert np.allclose(probs_res, probs_expected, atol=0.05)


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
