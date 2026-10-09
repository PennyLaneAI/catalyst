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

"""Test cases for the gridsynth discretization/decomposition pass."""

import numpy as np
import pennylane as qp
import pytest

from catalyst.passes import gridsynth


@pytest.mark.parametrize(
    "param",
    [-11.1, -7.7, -4.4, -2.2, -1.1, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 1.1, 2.2, 4.4, 7.7, 11.1],
)
@pytest.mark.parametrize("op", [qp.RZ, qp.PhaseShift])
@pytest.mark.parametrize("eps", [1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
def test_PhaseShift_gridsynth(param, op, eps):
    """Test that PhaseShift gates are correctly decomposed using the gridsynth pass."""

    dev = qp.device("lightning.qubit", wires=1)

    @qp.qnode(dev)
    def circuit(x: float):
        qp.Hadamard(0)
        op(x, wires=0)
        return qp.state()

    expected = circuit(param)
    gridsynthed_circuit = qp.transforms.gridsynth(circuit, epsilon=eps)
    qjitted_circuit = qp.qjit(gridsynthed_circuit, capture=True, collect_decomp_rules=False)
    result = qjitted_circuit(param)

    assert qp.math.allclose(result, expected, atol=eps)


@pytest.mark.parametrize(
    "param",
    [-7.7, -2.2, -0.1, 0.0, 0.1, 2.2, 7.7],
)
@pytest.mark.parametrize("eps", [1e-6])
def test_gridsynth_ppr_basis(param, eps):
    """Test that gridsynth with ppr_basis is consistent with the default gridsynth."""

    dev = qp.device("lightning.qubit", wires=1)

    @qp.qnode(dev)
    def circuit(x: float):
        qp.RZ(x, wires=0)
        return qp.state()

    gridsynthed_circuit = qp.transforms.gridsynth(circuit, epsilon=eps)
    gridsynthed_circuit_ppr = qp.transforms.gridsynth(circuit, epsilon=eps, ppr_basis=True)
    qjitted_circuit = qp.qjit(gridsynthed_circuit, capture=True, collect_decomp_rules=False)
    qjitted_circuit_with_ppr = qp.qjit(
        gridsynthed_circuit_ppr, capture=True, collect_decomp_rules=False
    )
    result = qjitted_circuit(param)
    result_with_ppr = qjitted_circuit_with_ppr(param)
    assert np.allclose(result, result_with_ppr, atol=eps)


@pytest.mark.parametrize("ppr_basis", [False, True])
def test_gridsynth_specs(ppr_basis):
    """Test that specs counts the runtime gridsynth decomposition without extra wires."""
    eps = 1e-4

    @qp.qjit(capture=True, target="mlir")
    @qp.transforms.gridsynth(epsilon=eps, ppr_basis=ppr_basis)
    @qp.qnode(qp.device("null.qubit", wires=2))
    def circuit(x: float):
        qp.RZ(x, 0)
        qp.RZ(x, 1)
        return qp.expval(qp.Z(0))

    resources = qp.specs(circuit, level=1)(0.5).resources
    assert resources.num_wires == 2

    ops = resources.quantum_operations
    t_per_rotation = (ops["PPR-pi/8-w1"] if ppr_basis else ops["T"]) / 2
    # Ross-Selinger sequences have about 3 log2(1/eps) T gates.
    assert 2.5 * np.log2(1 / eps) < t_per_rotation < 3.5 * np.log2(1 / eps)


@pytest.mark.parametrize("method", ["deterministic", "mixed"])
@pytest.mark.parametrize("ppr_basis", [False, True])
# Within the epsilon ranges that the resource hints are fitted on.
@pytest.mark.parametrize("eps", [1e-4, 1e-6])
def test_gridsynth_specs_matches_runtime(eps, ppr_basis, method):
    """Test that the specs T-count estimate matches the T-count executed at runtime, on average
    over random angles."""
    angles = np.random.default_rng(42).uniform(0, 4 * np.pi, 50)

    # The seed fixes the sequences sampled by the mixed method.
    @qp.qjit(capture=True, seed=37)
    @gridsynth(epsilon=eps, ppr_basis=ppr_basis, method=method)
    @qp.qnode(qp.device("null.qubit", wires=1))
    def circuit(angles):
        for theta in angles:
            qp.RZ(theta, 0)
        return qp.expval(qp.Z(0))

    estimated = qp.specs(circuit, level=1)(angles).resources
    executed = qp.specs(circuit, level="device")(angles).resources

    estimated_t = estimated.quantum_operations["PPR-pi/8-w1" if ppr_basis else "T"]
    # The device reports the PPR exp(-i pi/8 P) as PauliRot(pi/4).
    executed_t = executed.quantum_operations["PauliRot-pi/4-w1" if ppr_basis else "T"]
    assert estimated_t == pytest.approx(executed_t, rel=0.05)


def _mixed_circuit(op, eps, seed=None):
    dev = qp.device("lightning.qubit", wires=1)

    @qp.qnode(dev)
    def circuit(x: float):
        qp.Hadamard(0)
        op(x, wires=0)
        return qp.state()

    mixed_circuit = gridsynth(circuit, epsilon=eps, method="mixed")
    return circuit, qp.qjit(mixed_circuit, capture=True, collect_decomp_rules=False, seed=seed)


@pytest.mark.parametrize("param", [-7.7, -2.2, -0.1, 0.0, 0.3, 1.1, 4.4])
@pytest.mark.parametrize("op", [qp.RZ, qp.PhaseShift])
@pytest.mark.parametrize("eps", [1e-2, 1e-4, 1e-6])
def test_mixed_gridsynth_samples(param, op, eps):
    """Test that every sampled sequence of the mixed method is close to the target. For the
    diamond-norm error 2 * eps of the mixture, each branch satisfies Re(w) >= sqrt(1 - eps), so its
    infidelity is at most eps."""
    circuit, qjitted_circuit = _mixed_circuit(op, eps)
    expected = circuit(param)

    for _ in range(5):
        result = qjitted_circuit(param)
        fidelity = np.abs(np.vdot(expected, result)) ** 2
        assert 1 - fidelity <= eps


def test_mixed_gridsynth_average_channel():
    """Test that the average over samples is closer to the target than individual samples."""
    eps = 1e-2
    param = 1.1
    circuit, qjitted_circuit = _mixed_circuit(qp.RZ, eps)
    expected = np.asarray(circuit(param))
    expected_dm = np.outer(expected, expected.conj())

    num_samples = 400
    states = [np.asarray(qjitted_circuit(param)) for _ in range(num_samples)]
    average_dm = sum(np.outer(s, s.conj()) for s in states) / num_samples
    sample_errors = [np.linalg.norm(np.outer(s, s.conj()) - expected_dm) for s in states]

    assert len({tuple(np.round(s, 8)) for s in states}) > 1
    assert np.linalg.norm(average_dm - expected_dm) < np.mean(sample_errors)


def test_mixed_gridsynth_seeded():
    """Test that the runtime samples are reproducible with a qjit seed."""
    _, qjitted_circuit = _mixed_circuit(qp.RZ, 1e-2, seed=37)
    _, qjitted_circuit_same_seed = _mixed_circuit(qp.RZ, 1e-2, seed=37)

    assert np.allclose(qjitted_circuit(1.1), qjitted_circuit_same_seed(1.1))


def test_mixed_gridsynth_different_seeds():
    """Test that different qjit seeds lead to different runtime samples."""
    dev = qp.device("lightning.qubit", wires=1)

    @qp.qnode(dev)
    def circuit(x: float):
        qp.Hadamard(0)
        for i in range(5):
            qp.RZ(x * (i + 1), wires=0)
        return qp.state()

    mixed_circuit = gridsynth(circuit, epsilon=1e-2, method="mixed")
    results = [
        qp.qjit(mixed_circuit, capture=True, collect_decomp_rules=False, seed=seed)(1.1)
        for seed in (37, 38)
    ]
    assert not np.allclose(*results)


def test_invalid_method():
    """Test that an unknown synthesis method raises an error."""
    with pytest.raises(ValueError, match="method must be 'deterministic' or 'mixed'"):
        gridsynth(method="rus")


def test_mixed_gridsynth_ppr_basis():
    """Test that the mixed method in the PPR basis matches the target up to its accuracy."""
    eps = 1e-6
    dev = qp.device("lightning.qubit", wires=1)

    @qp.qnode(dev)
    def circuit(x: float):
        qp.Hadamard(0)
        qp.RZ(x, wires=0)
        return qp.state()

    mixed_circuit = gridsynth(circuit, epsilon=eps, ppr_basis=True, method="mixed")
    qjitted_circuit = qp.qjit(mixed_circuit, capture=True, collect_decomp_rules=False)
    expected = circuit(0.7)
    result = qjitted_circuit(0.7)
    assert 1 - np.abs(np.vdot(expected, result)) ** 2 <= eps


_EPSILON_WARN_FRAGMENT = "For epsilon smaller than 1e-6"


def test_epsilon_warning_emitted_for_small_epsilon(capfd):
    """Test that a runtime warning is written to stderr when epsilon < 1e-6."""
    dev = qp.device("lightning.qubit", wires=1)

    @qp.qnode(dev)
    def circuit(x: float):
        qp.RZ(x, wires=0)
        return qp.state()

    qp.qjit(
        qp.transforms.gridsynth(circuit, epsilon=1e-8), capture=True, collect_decomp_rules=False
    )(0.5)

    captured = capfd.readouterr()
    assert _EPSILON_WARN_FRAGMENT in captured.err


def test_epsilon_warning_not_emitted_for_safe_epsilon(capfd):
    """Test that no runtime warning is written to stderr when epsilon >= 1e-6."""
    dev = qp.device("lightning.qubit", wires=1)

    @qp.qnode(dev)
    def circuit(x: float):
        qp.RZ(x, wires=0)
        return qp.state()

    qp.qjit(
        qp.transforms.gridsynth(circuit, epsilon=1e-4), capture=True, collect_decomp_rules=False
    )(0.5)

    captured = capfd.readouterr()
    assert _EPSILON_WARN_FRAGMENT not in captured.err
