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
    _stretch_confusion_matrix,
    rem_apply_to_counts,
    rem_apply_to_probs,
    rem_apply_to_samples,
    rem_calibrate_counts,
    rem_calibrate_probs,
    rem_calibrate_samples,
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
    """Cached path must not emit rem_calibrate_* helpers in the traced MLIR."""
    n_qubits, shots = 2, 200
    circuit = _ghz_qnode(n_qubits, shots, "probs")
    cm = jnp.broadcast_to(jnp.eye(2, dtype=jnp.float64), (n_qubits, 2, 2))

    @catalyst.qjit(target="mlir")
    def cached():
        return mitigate_with_rem(circuit, calibration_matrices=cm)()

    ir = cached.mlir
    assert "runCalibration(false)" in ir
    assert "rem_apply_to_probs" in ir
    assert "rem_calibrate_probs" not in ir


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
    the output K must follow the trace-time branch in rem_apply_to_samples.
    """
    cm = _per_qubit_symmetric_confusion(n_qubits, 0.0)

    rng = np.random.default_rng(0)
    samples = rng.integers(0, 2, size=(n_shots, n_qubits)).astype(np.int32)

    bitstrings, counts = rem_apply_to_samples(
        jnp.asarray(samples),
        cm,
        jnp.arange(n_qubits),
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

    _, counts = rem_apply_to_samples(
        jnp.asarray(bits_msb_first),
        cm,
        jnp.arange(n_qubits),
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

    recovered = rem_apply_to_probs(noisy, cm, jnp.arange(n_qubits), n_qubits)
    assert np.allclose(np.asarray(recovered), np.asarray(clean), atol=1e-8)


def test_inverse_recovers_counts():
    """Same round-trip for counts: float linear solve must invert the analytic channel exactly."""
    n_qubits, p = 2, 0.15
    clean = jnp.asarray([500.0, 0.0, 0.0, 500.0])
    cm = _per_qubit_symmetric_confusion(n_qubits, p)

    flip = np.array([[1.0 - p, p], [p, 1.0 - p]])
    forward = reduce(np.kron, [flip] * n_qubits)
    noisy = jnp.asarray(forward @ np.asarray(clean))

    recovered = rem_apply_to_counts(noisy, cm, jnp.arange(n_qubits), n_qubits)
    assert np.allclose(np.asarray(recovered), np.asarray(clean), atol=1e-6)


def test_calibrate_paths_agree():
    """rem_calibrate_{samples,counts,probs} on the same channel must produce matching matrices."""
    n_qubits, n_shots = 2, 4000
    p = 0.10
    rng = np.random.default_rng(0)

    flips_zero = rng.random((n_shots, n_qubits)) < p
    zeros_samples = flips_zero.astype(np.int32)
    flips_one = rng.random((n_shots, n_qubits)) < p
    ones_samples = (1 - flips_one).astype(np.int32)

    cm_from_samples = rem_calibrate_samples(jnp.asarray(zeros_samples), jnp.asarray(ones_samples))

    # Build counts/probs from the same samples so the three paths consume the same channel.
    powers = (1 << np.arange(n_qubits - 1, -1, -1)).astype(np.int32)
    z_codes = (zeros_samples[:, ::-1] * powers[None, :]).sum(axis=1)
    o_codes = (ones_samples[:, ::-1] * powers[None, :]).sum(axis=1)
    zeros_counts = np.bincount(z_codes, minlength=2**n_qubits).astype(np.int64)
    ones_counts = np.bincount(o_codes, minlength=2**n_qubits).astype(np.int64)

    cm_from_counts = rem_calibrate_counts(jnp.asarray(zeros_counts), jnp.asarray(ones_counts))
    cm_from_probs = rem_calibrate_probs(
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
