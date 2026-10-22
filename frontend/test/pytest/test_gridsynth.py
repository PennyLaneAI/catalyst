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
from catalyst.utils.exceptions import CompileError


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

    assert qp.math.allclose(result, expected, atol=eps, rtol=0)


@pytest.mark.parametrize("param", [-2.2, 0.3, 1.1])
@pytest.mark.parametrize("op", [qp.RZ, qp.PhaseShift])
def test_adjoint_gridsynth(param, op):
    """Test that adjoint rotations are discretized as the inverse rotation."""
    eps = 1e-6
    dev = qp.device("lightning.qubit", wires=1)

    @qp.qnode(dev)
    def circuit(x: float):
        qp.Hadamard(0)
        qp.adjoint(op(x, wires=0))
        return qp.state()

    expected = circuit(param)
    gridsynthed_circuit = qp.transforms.gridsynth(circuit, epsilon=eps)
    qjitted_circuit = qp.qjit(gridsynthed_circuit, capture=True, collect_decomp_rules=False)
    assert qp.math.allclose(qjitted_circuit(param), expected, atol=eps)


@pytest.mark.parametrize("param", [-2.2, 0.3, 1.1])
@pytest.mark.parametrize("op", [qp.RZ, qp.PhaseShift])
@pytest.mark.parametrize("ppr_basis", [False, True])
@pytest.mark.parametrize("region", [qp.adjoint, lambda fn: qp.adjoint(qp.adjoint(fn))])
def test_adjoint_region_gridsynth(param, op, ppr_basis, region):
    """Test that rotations inside adjoint regions are discretized and inverted."""
    eps = 1e-6
    dev = qp.device("lightning.qubit", wires=1)

    def rotations(x):
        op(x, wires=0)
        qp.Hadamard(0)
        op(2 * x, wires=0)

    @qp.qnode(dev)
    def circuit(x: float):
        qp.Hadamard(0)
        region(rotations)(x)
        return qp.state()

    expected = circuit(param)
    gridsynthed_circuit = qp.transforms.gridsynth(circuit, epsilon=eps, ppr_basis=ppr_basis)
    qjitted_circuit = qp.qjit(gridsynthed_circuit, capture=True, collect_decomp_rules=False)
    assert qp.math.allclose(qjitted_circuit(param), expected, atol=2 * eps)


def test_gridsynth_ctrl_region_error():
    """Test that the control lowering raises an error for rotations discretized inside control
    regions, instead of dropping the control."""
    dev = qp.device("lightning.qubit", wires=2)

    def rotation(x):
        qp.RZ(x, 0)

    @qp.qnode(dev)
    def circuit(x: float):
        qp.ctrl(rotation, control=1)(x)
        return qp.state()

    with pytest.raises(CompileError, match="cannot control a function call"):
        qp.qjit(
            qp.transforms.gridsynth(circuit, epsilon=1e-6), capture=True, collect_decomp_rules=False
        )(0.7)


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
    assert np.allclose(result, result_with_ppr, atol=eps, rtol=0)


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


@pytest.mark.parametrize("method", ["deterministic", "mixed", "mixed-fallback"])
@pytest.mark.parametrize("ppr_basis", [False, True])
# Within the epsilon ranges that the resource hints are fitted on.
@pytest.mark.parametrize("eps", [1e-4, 1e-6])
def test_gridsynth_specs_matches_runtime(eps, ppr_basis, method):
    """Test that the specs T-count estimate matches the T-count executed at runtime, on average
    over random angles."""
    angles = np.random.default_rng(42).uniform(0, 4 * np.pi, 50)
    # The estimate cannot know which angles are within 2 * eps of a multiple of pi/4, which need at
    # most one T gate.
    distance_to_pi_4 = np.abs(np.remainder(angles + np.pi / 8, np.pi / 4) - np.pi / 8)
    angles = angles[distance_to_pi_4 > 2 * eps]

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
    # Device-level measurements return 0, so mixed-fallback's fallbacks, which add about 3% to the
    # estimate, never execute.
    rel = 0.08 if method == "mixed-fallback" else 0.05
    assert estimated_t == pytest.approx(executed_t, rel=rel)


@pytest.mark.parametrize("ppr_basis", [False, True])
def test_mixed_fallback_gridsynth_specs_wires(ppr_basis):
    """Test the known limitation that specs counts every ancilla allocation of the mixed fallback
    method as a wire, although each ancilla is released before the next one is allocated."""

    @qp.qjit(capture=True, target="mlir")
    @gridsynth(epsilon=1e-6, ppr_basis=ppr_basis, method="mixed-fallback")
    @qp.qnode(qp.device("null.qubit", wires=2))
    def circuit(x: float):
        qp.RZ(x, 0)
        qp.RZ(x, 1)
        return qp.expval(qp.Z(0))

    assert qp.specs(circuit, level=1)(0.5).resources.num_wires == 4


@pytest.mark.parametrize("param", [-7.7, -2.2, -0.1, 0.0, 0.3, 1.1, 4.4])
@pytest.mark.parametrize("op", [qp.RZ, qp.PhaseShift])
@pytest.mark.parametrize("eps", [1e-2, 1e-4, 1e-6])
@pytest.mark.parametrize("ppr_basis", [False, True])
def test_mixed_gridsynth_samples(param, op, eps, ppr_basis):
    """Test that every sampled sequence of the mixed method is close to the target. For the
    diamond-norm error 2 * eps of the mixture, each branch satisfies Re(w) >= sqrt(1 - eps), so its
    infidelity is at most eps. Each wire samples its own sequence."""
    num_wires = 5
    dev = qp.device("lightning.qubit", wires=num_wires)

    @qp.qnode(dev)
    def circuit(x: float):
        for wire in range(num_wires):
            qp.Hadamard(wire)
            op(x, wires=wire)
        return qp.state()

    mixed_circuit = gridsynth(circuit, epsilon=eps, ppr_basis=ppr_basis, method="mixed")
    qjitted_circuit = qp.qjit(mixed_circuit, capture=True, collect_decomp_rules=False, seed=37)
    expected = qp.matrix(op(param, wires=0)) @ np.array([1, 1]) / np.sqrt(2)

    result = np.reshape(qjitted_circuit(param), [2] * num_wires)
    for wire in range(num_wires):
        wire_state = np.moveaxis(result, wire, 0).reshape(2, -1)
        density_matrix = wire_state @ wire_state.conj().T
        fidelity = np.real(np.vdot(expected, density_matrix @ expected))
        assert 1 - fidelity <= eps


@pytest.mark.parametrize("op", [qp.RZ, qp.PhaseShift])
@pytest.mark.parametrize("eps", [1e-2, 1e-4])
@pytest.mark.parametrize("ppr_basis", [False, True])
def test_mixed_fallback_gridsynth_samples(op, eps, ppr_basis):
    """Test the sampled states of the mixed fallback method."""
    dev = qp.device("lightning.qubit", wires=1)

    @qp.qnode(dev)
    def circuit(x: float):
        qp.Hadamard(0)
        op(x, wires=0)
        return qp.state()

    fallback_circuit = gridsynth(circuit, epsilon=eps, ppr_basis=ppr_basis, method="mixed-fallback")
    qjitted_circuit = qp.qjit(fallback_circuit, capture=True, collect_decomp_rules=False)

    infidelities = []
    for param in [-2.2, 0.3, 1.1, 4.4]:
        expected = circuit(param)
        for _ in range(30):
            result = qjitted_circuit(param)
            infidelities.append(1 - np.abs(np.vdot(expected, result)) ** 2)

    infidelities = np.array(infidelities)
    # Each circuit execution samples one projective rotation. Its ancilla measurement succeeds with
    # probability at least 0.95, leaving infidelity at most eps. Otherwise a fallback only loosely
    # corrects the rotation, with infidelity at most MAX_FALLBACK_EPSILON / 2 = 0.125. The
    # threshold is 85% rather than 95% to allow for sampling noise: at a 95% success rate, 120
    # executions fall below it with probability <1e-5.
    assert np.all(infidelities <= 0.125)
    assert np.mean(infidelities <= eps) >= 0.85


@pytest.mark.parametrize("method", ["mixed", "mixed-fallback"])
def test_mixed_gridsynth_seed(method):
    """Test that the runtime samples are reproducible with a qjit seed and differ across seeds."""
    dev = qp.device("lightning.qubit", wires=1)

    @qp.qnode(dev)
    def circuit(x: float):
        qp.Hadamard(0)
        for i in range(5):
            qp.RZ(x * (i + 1), wires=0)
        return qp.state()

    mixed_circuit = gridsynth(circuit, epsilon=1e-2, method=method)
    results = [
        qp.qjit(mixed_circuit, capture=True, collect_decomp_rules=False, seed=seed)(1.1)
        for seed in (37, 37, 38)
    ]
    assert np.allclose(results[0], results[1])
    assert not np.allclose(results[0], results[2])


@pytest.mark.parametrize("method", ["mixed", "mixed-fallback"])
@pytest.mark.parametrize("mcm_method", ["single-branch-statistics", "one-shot"])
def test_mixed_gridsynth_samples_per_shot(method, mcm_method):
    """Test that the shots of a call share one sample with single-branch statistics, and that
    every shot draws its own with one-shot. With a coarse epsilon, samples differ visibly, so the
    means of calls spread far more than shot noise if their shots share a sample."""
    x = 1.1
    shots = 1000
    dev = qp.device("lightning.qubit", wires=1)

    @qp.qnode(dev, shots=shots, mcm_method=mcm_method)
    def circuit(x: float):
        qp.Hadamard(0)
        qp.RZ(x, wires=0)
        return qp.expval(qp.X(0))

    qjitted_circuit = qp.qjit(
        gridsynth(circuit, epsilon=0.05, method=method), capture=True, collect_decomp_rules=False
    )
    spread = np.std([qjitted_circuit(x) for _ in range(20)])
    shot_noise = np.sqrt((1 - np.cos(x) ** 2) / shots)

    if mcm_method == "one-shot":
        assert spread < 2 * shot_noise
    else:
        assert spread > 4 * shot_noise


def test_invalid_method():
    """Test that an unknown synthesis method raises an error."""
    with pytest.raises(ValueError, match="method must be 'deterministic', 'mixed' or"):
        gridsynth(method="rus")


@pytest.mark.parametrize("eps", [1e-10, 1e-12, 1e-15])
def test_gridsynth_high_precision(eps):
    """Test that gridsynth reaches precisions beyond double-precision geometry."""
    dev = qp.device("lightning.qubit", wires=1)

    @qp.qnode(dev)
    def circuit(x: float):
        qp.Hadamard(0)
        qp.RZ(x, wires=0)
        return qp.state()

    gridsynthed_circuit = qp.transforms.gridsynth(circuit, epsilon=eps)
    qjitted_circuit = qp.qjit(gridsynthed_circuit, capture=True, collect_decomp_rules=False)

    for param in [-4.4, -0.3, 0.1, 2.2]:
        # Simulating the ~150 gates at eps=1e-15 in double precision adds an error of about 1e-14.
        assert qp.math.allclose(
            qjitted_circuit(param), circuit(param), atol=max(eps, 1e-13), rtol=0
        )


@pytest.mark.parametrize("method", ["mixed", "mixed-fallback"])
@pytest.mark.parametrize("eps", [1e-10, 1e-15])
def test_mixed_gridsynth_high_precision(eps, method):
    """Test that the mixed methods reach the precisions of the deterministic method."""
    dev = qp.device("lightning.qubit", wires=1)

    @qp.qnode(dev)
    def circuit(x: float):
        qp.Hadamard(0)
        qp.RZ(x, wires=0)
        return qp.state()

    mixed_circuit = gridsynth(circuit, epsilon=eps, method=method)
    qjitted_circuit = qp.qjit(mixed_circuit, capture=True, collect_decomp_rules=False, seed=37)

    infidelities = []
    for param in [-4.4, -0.3, 0.1, 2.2]:
        for _ in range(5):
            fidelity = np.abs(np.vdot(circuit(param), qjitted_circuit(param))) ** 2
            infidelities.append(1 - fidelity)

    infidelities = np.array(infidelities)
    # Simulating the gates in double precision adds an infidelity of about 1e-14.
    accurate = infidelities <= max(eps, 1e-13)
    if method == "mixed":
        assert np.all(accurate)
    else:
        # The ancilla measurement fails with probability at most 5%, and is followed by a fallback
        # with infidelity at most MAX_FALLBACK_EPSILON / 2 = 0.125.
        assert np.all(infidelities <= 0.125)
        assert np.mean(accurate) >= 0.75


@pytest.mark.parametrize("method", ["deterministic", "mixed", "mixed-fallback"])
def test_error_for_unsupported_epsilon(method):
    """Test that an epsilon below the supported precision raises an error at runtime."""
    dev = qp.device("lightning.qubit", wires=1)

    @qp.qnode(dev)
    def circuit(x: float):
        qp.RZ(x, wires=0)
        return qp.state()

    qjitted_circuit = qp.qjit(
        gridsynth(circuit, epsilon=1e-16, method=method), capture=True, collect_decomp_rules=False
    )
    with pytest.raises(RuntimeError, match="the smallest supported epsilon is 1e-15"):
        qjitted_circuit(0.5)
