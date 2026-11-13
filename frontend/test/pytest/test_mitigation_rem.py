# Copyright 2026 Haiqu, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test integration for catalyst.mitigate_with_rem."""

from functools import reduce

import jax.numpy as jnp
import numpy as np
import pennylane as qp
import pytest

import catalyst
from catalyst.api_extensions.error_mitigation import mitigate_with_rem
from catalyst.api_extensions.rem_postprocessing import (
    _rem_apply_to_counts,
    _rem_apply_to_probs,
    _rem_apply_to_samples,
    _rem_calibrate_counts,
    _rem_calibrate_probs,
    _rem_calibrate_samples,
    _stretch_confusion_matrix,
    rem_apply,
    rem_calibrate,
)


def _ghz_qnode(n_qubits, shots, return_kind):
    """Return a noiseless GHZ-state QNode whose measurement is selected by string."""
    dev = qp.device("lightning.qubit", wires=n_qubits)

    def circuit():
        qp.Hadamard(wires=0)
        for k in range(1, n_qubits):
            qp.CNOT(wires=[0, k])
        if return_kind == "sample":
            return qp.sample()
        if return_kind == "counts":
            return qp.counts()
        if return_kind == "probs":
            return qp.probs(wires=range(n_qubits))
        raise ValueError(return_kind)

    return qp.set_shots(qp.QNode(circuit, dev), shots=shots)


@pytest.mark.parametrize("n_qubits", [1, 2, 11])
def test_probs(n_qubits):
    """With no readout noise the mitigated probs must equal the analytic GHZ probs."""
    shots = 2000
    circuit = _ghz_qnode(n_qubits, shots, "probs")

    @catalyst.qjit
    def mitigated():
        return mitigate_with_rem(circuit)()

    out = np.asarray(mitigated())
    expected = np.zeros(2**n_qubits)
    expected[0] = 0.5
    expected[-1] = 0.5
    assert np.allclose(out, expected, atol=5e-2)


@pytest.mark.parametrize("n_qubits", [1, 2, 11])
def test_counts(n_qubits):
    """Mitigated counts on a noiseless GHZ state must concentrate on |0...0> and |1...1>."""
    shots = 2000
    circuit = _ghz_qnode(n_qubits, shots, "counts")

    @catalyst.qjit
    def mitigated():
        return mitigate_with_rem(circuit)()

    eigvals, counts = mitigated()
    counts = np.asarray(counts)
    assert counts.shape == (2**n_qubits,)
    # The two GHZ peaks should hold essentially all of the shot mass.
    peak_mass = counts[0] + counts[-1]
    assert peak_mass > 0.9 * shots
    # The eigenvalue axis is the basis-state index, MSB-first.
    assert np.allclose(np.asarray(eigvals), np.arange(2**n_qubits))


@pytest.mark.parametrize("n_qubits", [1, 2, 11])
def test_sample(n_qubits):
    """Mitigated sample histogram on a noiseless GHZ state must match the analytic distribution.

    Note that for `n_qubits == 11`, 2^n_qubits > shots (2000), which covers the shot-count bounded
    codepath for transition matrix generation.
    """
    shots = 2000
    circuit = _ghz_qnode(n_qubits, shots, "sample")

    @catalyst.qjit
    def mitigated():
        return mitigate_with_rem(circuit)()

    bitstrings, counts = mitigated()
    bitstrings = np.asarray(bitstrings)
    counts = np.asarray(counts)

    # Codes index basis states MSB-first.
    powers = 1 << np.arange(n_qubits - 1, -1, -1)
    codes = (bitstrings * powers[None, :]).sum(axis=1)

    histogram = np.zeros(2**n_qubits)
    for code, c in zip(codes, counts):
        histogram[code] += c

    assert histogram[0] + histogram[-1] > 0.9 * shots


def test_return_calibration_matrices():
    """return_calibration_matrices=True wraps the mitigated probs in a (result, cm) tuple."""
    n_qubits, shots = 2, 2000
    circuit = _ghz_qnode(n_qubits, shots, "probs")

    @catalyst.qjit
    def mitigated():
        return mitigate_with_rem(circuit, return_calibration_matrices=True)()

    result, cm = mitigated()
    cm = np.asarray(cm)
    assert cm.shape == (n_qubits, 2, 2)
    # Each 2x2 confusion matrix is column-stochastic.
    assert np.allclose(cm.sum(axis=-2), 1.0, atol=1e-6)

    expected = np.zeros(2**n_qubits)
    expected[0] = 0.5
    expected[-1] = 0.5
    assert np.allclose(np.asarray(result), expected, atol=5e-2)


def test_caching_probs_roundtrip():
    """Compute calibration once, reuse it on a second qjit call, get matching GHZ probs."""
    n_qubits, shots = 2, 2000
    circuit = _ghz_qnode(n_qubits, shots, "probs")

    @catalyst.qjit
    def fresh():
        return mitigate_with_rem(circuit, return_calibration_matrices=True)()

    _, cm = fresh()

    @catalyst.qjit
    def cached(cm_in):
        return mitigate_with_rem(circuit, calibration_matrices=cm_in)()

    out = np.asarray(cached(cm))
    expected = np.zeros(2**n_qubits)
    expected[0] = 0.5
    expected[-1] = 0.5
    assert np.allclose(out, expected, atol=5e-2)


def test_caching_counts_roundtrip():
    """Counts MP: cached calibration matrices yield concentrated GHZ peaks on a noiseless device."""
    n_qubits, shots = 2, 2000
    circuit = _ghz_qnode(n_qubits, shots, "counts")

    @catalyst.qjit
    def fresh():
        return mitigate_with_rem(circuit, return_calibration_matrices=True)()

    _, cm = fresh()

    @catalyst.qjit
    def cached(cm_in):
        return mitigate_with_rem(circuit, calibration_matrices=cm_in)()

    eigvals, counts = cached(cm)
    counts = np.asarray(counts)
    assert counts.shape == (2**n_qubits,)
    assert counts[0] + counts[-1] > 0.9 * shots
    assert np.allclose(np.asarray(eigvals), np.arange(2**n_qubits))


def test_caching_sample_roundtrip():
    """Sample MP histogram-path cache reuse on a noiseless GHZ device."""
    n_qubits, shots = 2, 2000
    circuit = _ghz_qnode(n_qubits, shots, "sample")

    @catalyst.qjit
    def fresh():
        return mitigate_with_rem(circuit, return_calibration_matrices=True)()

    _, cm = fresh()

    @catalyst.qjit
    def cached(cm_in):
        return mitigate_with_rem(circuit, calibration_matrices=cm_in)()

    bitstrings, counts = cached(cm)
    bitstrings = np.asarray(bitstrings)
    counts = np.asarray(counts)
    powers = 1 << np.arange(n_qubits - 1, -1, -1)
    codes = (bitstrings * powers[None, :]).sum(axis=1)
    histogram = np.zeros(2**n_qubits)
    for code, c in zip(codes, counts):
        histogram[code] += c
    assert histogram[0] + histogram[-1] > 0.9 * shots


def test_caching_skips_rem_calibrate_in_ir():
    """Cached path must not emit _rem_calibrate_* helpers in the traced MLIR."""
    n_qubits, shots = 2, 200
    circuit = _ghz_qnode(n_qubits, shots, "probs")
    cm = jnp.broadcast_to(jnp.eye(2, dtype=jnp.float64), (n_qubits, 2, 2))

    @catalyst.qjit(target="mlir")
    def cached():
        return mitigate_with_rem(circuit, calibration_matrices=cm)()

    ir = cached.mlir
    assert "runCalibration(false)" in ir
    assert "_rem_apply_to_probs" in ir
    assert "_rem_calibrate_probs" not in ir


def _per_qubit_symmetric_confusion(n_qubits, p):
    """Stack of ``(n_qubits, 2, 2)`` symmetric bit-flip channels with flip probability ``p``."""
    one = np.array([[1.0 - p, p], [p, 1.0 - p]])
    return jnp.asarray(np.broadcast_to(one, (n_qubits, 2, 2)))


@pytest.mark.parametrize(
    "n_qubits,n_shots,expected_k",
    [
        # 2**n <= n_shots: histogram path, output K = 2**n_qubits.
        (2, 8, 4),
        (3, 16, 8),
        # 2**n > n_shots: sort-RLE path, output K = n_shots.
        (4, 8, 8),
        (5, 16, 16),
    ],
)
def test_sample_path_dispatch(n_qubits, n_shots, expected_k):
    """Cover both sample post-processing paths: histogram (K=2**n) and sort-RLE (K=n_shots).

    With an identity noise channel, the linear solve reduces to the user
    histogram itself, so total mass must equal n_shots in either path and
    the output K must follow the trace-time branch in _rem_apply_to_samples.
    """
    cm = _per_qubit_symmetric_confusion(n_qubits, 0.0)

    rng = np.random.default_rng(0)
    samples = rng.integers(0, 2, size=(n_shots, n_qubits)).astype(np.int32)

    bitstrings, counts = _rem_apply_to_samples(
        jnp.asarray(samples),
        cm,
        n_qubits,
    )

    assert counts.shape == (expected_k,)
    assert bitstrings.shape == (expected_k, n_qubits)
    assert np.isclose(float(jnp.sum(counts)), float(n_shots))


def test_sample_sort_rle_path():
    """Sort-RLE path (2**n > n_shots) must preserve total shot mass through the linear solve."""
    n_qubits, n_shots, p = 4, 8, 0.10
    cm = _per_qubit_symmetric_confusion(n_qubits, p)

    rng = np.random.default_rng(7)
    codes = rng.choice(2**n_qubits, size=n_shots, replace=False)
    bits_msb_first = ((codes[:, None] >> np.arange(n_qubits - 1, -1, -1)) & 1).astype(np.int32)

    _, counts = _rem_apply_to_samples(
        jnp.asarray(bits_msb_first),
        cm,
        n_qubits,
    )
    assert counts.shape == (n_shots,)
    assert np.isclose(float(jnp.sum(counts)), float(n_shots), atol=1e-6)


def test_inverse_recovers_probs():
    """Round-trip: apply known confusion to a clean prob vector, invert with REM, recover input."""
    n_qubits, p = 2, 0.10
    clean = jnp.asarray([0.5, 0.0, 0.0, 0.5])  # 2-qubit GHZ analytic probs
    cm = _per_qubit_symmetric_confusion(n_qubits, p)

    flip = np.array([[1.0 - p, p], [p, 1.0 - p]])
    forward = reduce(np.kron, [flip] * n_qubits)
    noisy = jnp.asarray(forward @ np.asarray(clean))

    recovered = _rem_apply_to_probs(noisy, cm, n_qubits)
    assert np.allclose(np.asarray(recovered), np.asarray(clean), atol=1e-8)


def test_inverse_recovers_counts():
    """Same round-trip for counts: float linear solve must invert the analytic channel exactly."""
    n_qubits, p = 2, 0.15
    clean = jnp.asarray([500.0, 0.0, 0.0, 500.0])
    cm = _per_qubit_symmetric_confusion(n_qubits, p)

    flip = np.array([[1.0 - p, p], [p, 1.0 - p]])
    forward = reduce(np.kron, [flip] * n_qubits)
    noisy = jnp.asarray(forward @ np.asarray(clean))

    recovered = _rem_apply_to_counts(noisy, cm, n_qubits)
    assert np.allclose(np.asarray(recovered), np.asarray(clean), atol=1e-6)


def test_calibrate_paths_agree():
    """_rem_calibrate_{samples,counts,probs} on the same channel must produce matching matrices."""
    n_qubits, n_shots = 2, 4000
    p = 0.10
    rng = np.random.default_rng(0)

    flips_zero = rng.random((n_shots, n_qubits)) < p
    zeros_samples = flips_zero.astype(np.int32)
    flips_one = rng.random((n_shots, n_qubits)) < p
    ones_samples = (1 - flips_one).astype(np.int32)

    cm_from_samples = _rem_calibrate_samples(jnp.asarray(zeros_samples), jnp.asarray(ones_samples))

    # Build counts/probs from the same samples so the three paths consume the same channel.
    # Column k of the samples is wire k, and wire 0 is the most significant bit of the index.
    powers = (1 << np.arange(n_qubits - 1, -1, -1)).astype(np.int32)
    z_codes = (zeros_samples * powers[None, :]).sum(axis=1)
    o_codes = (ones_samples * powers[None, :]).sum(axis=1)
    zeros_counts = np.bincount(z_codes, minlength=2**n_qubits).astype(np.int64)
    ones_counts = np.bincount(o_codes, minlength=2**n_qubits).astype(np.int64)

    cm_from_counts = _rem_calibrate_counts(jnp.asarray(zeros_counts), jnp.asarray(ones_counts))
    cm_from_probs = _rem_calibrate_probs(
        jnp.asarray(zeros_counts / zeros_counts.sum()),
        jnp.asarray(ones_counts / ones_counts.sum()),
    )

    expected = np.array([[1.0 - p, p], [p, 1.0 - p]])
    for k in range(n_qubits):
        for cm in (cm_from_samples, cm_from_counts, cm_from_probs):
            assert np.allclose(np.asarray(cm[k]), expected, atol=3e-2)


@pytest.mark.parametrize("n_qubits", [1, 2, 3])
def test_stretch_confusion_matrix(n_qubits):
    """_stretch_confusion_matrix on a bitstring row must agree with an explicit lookup."""
    np_rng = np.random.default_rng(42)
    cm = np_rng.random((2, 2))
    bitstring = np_rng.integers(0, 2, size=n_qubits).astype(np.int32)

    expected = np.empty((n_qubits, n_qubits))
    for i in range(n_qubits):
        for j in range(n_qubits):
            expected[i, j] = cm[bitstring[i], bitstring[j]]

    got = _stretch_confusion_matrix(jnp.asarray(bitstring), jnp.asarray(cm))
    assert np.allclose(np.asarray(got), expected)


# Per-wire, asymmetric readout errors: P(measure 1 | prepared 0) and P(measure 0 | prepared 1)
# differ between wires and between directions, so a reversed wire order or a transposed
# confusion matrix changes the results. A symmetric channel shared by all wires hides both.
_P_1_GIVEN_0 = (0.02, 0.10, 0.25)
_P_0_GIVEN_1 = (0.15, 0.05, 0.30)


def _per_qubit_asymmetric_confusion(n_qubits):
    """Stack of ``(n_qubits, 2, 2)`` confusion matrices indexed as ``[wire, measured, prepared]``."""
    return np.stack(
        [
            np.array(
                [
                    [1.0 - _P_1_GIVEN_0[k], _P_0_GIVEN_1[k]],
                    [_P_1_GIVEN_0[k], 1.0 - _P_0_GIVEN_1[k]],
                ]
            )
            for k in range(n_qubits)
        ]
    )


def _transition_matrix(cm):
    """Full ``(2**n, 2**n)`` matrix ``A[measured, prepared]``; wire 0 is the most significant bit."""
    return reduce(np.kron, list(cm))


def test_calibrate_samples_wire_order_and_orientation():
    """_rem_calibrate_samples returns confusion_matrices[k][measured, prepared] for wire k."""
    n_qubits, n_shots = 3, 1000
    cm = _per_qubit_asymmetric_confusion(n_qubits)

    # Column k flips exactly round(p * n_shots) shots, so the estimate is exact.
    zeros_samples = np.zeros((n_shots, n_qubits), dtype=np.int32)
    ones_samples = np.ones((n_shots, n_qubits), dtype=np.int32)
    for k in range(n_qubits):
        zeros_samples[: round(_P_1_GIVEN_0[k] * n_shots), k] = 1
        ones_samples[: round(_P_0_GIVEN_1[k] * n_shots), k] = 0

    got = _rem_calibrate_samples(jnp.asarray(zeros_samples), jnp.asarray(ones_samples))
    assert np.allclose(np.asarray(got), cm)


@pytest.mark.parametrize("kind", ["probs", "counts"])
def test_calibrate_probs_counts_wire_order_and_orientation(kind):
    """_rem_calibrate_probs/counts return confusion_matrices[k][measured, prepared] for wire k."""
    n_qubits = 3
    cm = _per_qubit_asymmetric_confusion(n_qubits)
    transition = _transition_matrix(cm)
    # Columns 0 and -1 are the noisy distributions of the all-zeros and all-ones states.
    zeros_probs, ones_probs = transition[:, 0], transition[:, -1]

    if kind == "probs":
        got = _rem_calibrate_probs(jnp.asarray(zeros_probs), jnp.asarray(ones_probs))
    else:
        n_shots = 10**6
        got = _rem_calibrate_counts(
            jnp.asarray(np.round(zeros_probs * n_shots).astype(np.int64)),
            jnp.asarray(np.round(ones_probs * n_shots).astype(np.int64)),
        )
    assert np.allclose(np.asarray(got), cm, atol=1e-5)


@pytest.mark.parametrize("kind", ["probs", "counts"])
def test_apply_probs_counts_per_wire_asymmetric(kind):
    """_rem_apply_to_probs/counts invert a per-wire, asymmetric readout channel exactly."""
    n_qubits = 3
    cm = _per_qubit_asymmetric_confusion(n_qubits)
    # Unequal weights on |100> and |011>: not invariant under reversing the wire order.
    clean = np.zeros(2**n_qubits)
    clean[0b100], clean[0b011] = 0.7, 0.3
    scale = 1.0 if kind == "probs" else 1000.0
    noisy = _transition_matrix(cm) @ clean * scale

    apply = _rem_apply_to_probs if kind == "probs" else _rem_apply_to_counts
    recovered = apply(jnp.asarray(noisy), jnp.asarray(cm), n_qubits)
    assert np.allclose(np.asarray(recovered), clean * scale, atol=1e-8 * scale)


def _reference_mitigated_counts(samples, cm):
    """Reference REM solve on the sampled bitstrings only (Nation et al.), keyed by bitstring."""
    unique, counts = np.unique(samples, axis=0, return_counts=True)
    n_qubits = samples.shape[1]
    transition = np.array(
        [[np.prod([cm[k][m[k], p[k]] for k in range(n_qubits)]) for p in unique] for m in unique]
    )
    transition /= transition.sum(axis=0, keepdims=True)
    mitigated = np.linalg.solve(transition, counts.astype(np.float64))
    return {tuple(b): c for b, c in zip(unique, mitigated)}


@pytest.mark.parametrize(
    "n_qubits,n_shots",
    [
        (3, 64),  # 2**n <= n_shots: histogram path
        (3, 6),  # 2**n > n_shots: sort-based path
    ],
)
def test_apply_samples_per_wire_asymmetric(n_qubits, n_shots):
    """Both _rem_apply_to_samples paths match a reference solve with per-wire, asymmetric noise."""
    cm = _per_qubit_asymmetric_confusion(n_qubits)
    rng = np.random.default_rng(3)
    samples = rng.integers(0, 2, size=(n_shots, n_qubits)).astype(np.int32)

    bitstrings, counts = _rem_apply_to_samples(jnp.asarray(samples), jnp.asarray(cm), n_qubits)

    # Sum over repeated rows: the sort-based path stores each count at the first occurrence.
    got = {}
    for b, c in zip(np.asarray(bitstrings), np.asarray(counts)):
        got[tuple(b)] = got.get(tuple(b), 0.0) + c
    expected = _reference_mitigated_counts(samples, cm)
    for b, c in expected.items():
        assert np.isclose(got[b], c)
    assert np.allclose([c for b, c in got.items() if b not in expected], 0.0)


@pytest.mark.parametrize("kind", ["probs", "counts"])
def test_cached_matrices_wire_order(kind):
    """mitigate_with_rem pairs calibration_matrices[k] with wire k of the device."""
    n_qubits, shots = 3, 100
    cm = _per_qubit_asymmetric_confusion(n_qubits)
    dev = qp.device("lightning.qubit", wires=n_qubits)

    @qp.qnode(dev, shots=shots)
    def circuit():
        qp.PauliX(wires=0)  # noiseless |100>
        return qp.probs() if kind == "probs" else qp.counts()

    @catalyst.qjit
    def mitigated(cm_in):
        return mitigate_with_rem(circuit, calibration_matrices=cm_in)()

    out = mitigated(jnp.asarray(cm))
    result = np.asarray(out if kind == "probs" else out[1])
    measured = np.zeros(2**n_qubits)
    measured[0b100] = 1.0 if kind == "probs" else shots
    expected = np.linalg.solve(_transition_matrix(cm), measured)
    assert np.allclose(result, expected)


def test_unsupported_measurement():
    """REM rejects non-sample/counts/probs measurements with a clear trace-time error."""
    dev = qp.device("lightning.qubit", wires=2)

    def circuit():
        qp.Hadamard(wires=0)
        return qp.expval(qp.PauliZ(0))

    qnode = qp.set_shots(qp.QNode(circuit, dev), shots=200)

    def mitigated():
        return mitigate_with_rem(qnode)()

    with pytest.raises(AssertionError, match="measurement process must be one of"):
        catalyst.qjit(mitigated)


def test_rem_calibrate_apply_reject_unknown_mp_kind():
    """The public dispatchers raise for measurement processes other than sample, counts, probs."""
    cm = _per_qubit_symmetric_confusion(1, 0.0)
    with pytest.raises(ValueError, match="Unsupported measurement process 'expval'"):
        rem_calibrate("expval", cm, cm)
    with pytest.raises(ValueError, match="Unsupported measurement process 'expval'"):
        rem_apply("expval", cm, cm, 1)
