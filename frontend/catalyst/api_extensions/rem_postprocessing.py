# Copyright 2025-2026 Haiqu, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
JAX-compilable post-processing helpers for Readout Error Mitigation (REM).

These helpers integrate with the catalyst frontend so ``mitigate_with_rem``
can apply classical SampleMP post-processing - per-qubit confusion-matrix
calibration, reduced-confusion-matrix construction, and a linear solve -
to the three sample arrays emitted by the ``mitigation.rem`` op after
lowering.
"""

from functools import partial, reduce
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike


def _bitstrings_for_n_qubits(n_qubits: int) -> jax.Array:
    """All ``2**n_qubits`` bitstrings in MSB-first order, shape (2**n, n_qubits)."""
    n_states = 2**n_qubits
    indices = jnp.arange(n_states, dtype=jnp.int32)
    bit_positions = jnp.arange(n_qubits - 1, -1, -1, dtype=jnp.int32)
    return ((indices[:, None] >> bit_positions[None, :]) & 1).astype(jnp.int32)


def _bitstring_codes(shot_result_array: jax.Array, n_qubits: int) -> jax.Array:
    """Encode each (n_qubits,) bitstring row as an MSB-first integer in [0, 2**n)."""
    powers = 1 << jnp.arange(n_qubits - 1, -1, -1, dtype=jnp.int32)
    return (shot_result_array * powers[None, :]).sum(axis=1).astype(jnp.int32)


def _count_shots_histogram(shot_result_array: jax.Array, n_qubits: int) -> jax.Array:
    """Per-bitstring shot counts over the full ``2**n_qubits`` state space.
    Each shot is encoded as an integer code, one-hot encoded across the
    ``2**n_qubits`` possible bitstrings, then reduced with a sum.
    Uses ``O(n_shots * 2**n_qubits)`` memory; only invoked when
    the dispatcher in :func:`rem_apply_to_samples` has confirmed that
    ``2**n_qubits <= n_shots``.
    """
    codes = _bitstring_codes(shot_result_array, n_qubits)
    one_hot = jax.nn.one_hot(codes, 2**n_qubits, dtype=jnp.int32)
    return one_hot.sum(axis=0)


def _sort_based_unique(samples: jax.Array, n_qubits: int, n_shots: int):
    """Per-bistring shot counts over ``n_shots`` space. Each unique bistring in
    resulting array has a non-zero count assotiated with it at the same index.
    The resulting array is also sorted by bitstrings.

    Returns tuple ``(sorted_bitstrings, counts_at_first)`` of shapes
    ``(n_shots, n_qubits)`` and ``(n_shots,)``. ``counts_at_first[i] > 0`` iff
    position ``i`` is the first occurrence of a unique bitstring after sorting,
    and the value equals that bitstring's total shot count. The remaining
    ``n_shots - n_unique`` positions are zero-padded to maintain fixed tensor
    dimensions; :func:`rem_apply_to_samples` pins them to identity in the linear solve.
    """
    powers = 1 << jnp.arange(n_qubits - 1, -1, -1, dtype=jnp.int32)
    codes = (samples * powers[None, :]).sum(axis=1).astype(jnp.int32)  # (n_shots,)

    # Stable sort by code so equal bitstrings land in consecutive positions.
    order = jnp.argsort(codes)
    sorted_codes = codes[order]
    sorted_bitstrings = samples[order]

    # boundaries[i] = True iff i is the first position of a new code run.
    diff = sorted_codes[1:] != sorted_codes[:-1]
    boundaries = jnp.concatenate([jnp.array([True]), diff])

    # segment_id[i] = which run position i belongs to (monotone non-decreasing).
    segment_id = jnp.cumsum(boundaries.astype(jnp.int32)) - 1  # (n_shots,)

    # Run-length per segment via one-hot bincount. The one-hot is (n_shots,
    # n_shots), which is the same order as the (n_shots, n_shots) reduced
    # confusion matrix we'll allocate next - bounded by n_shots, NOT 2**n_qubits.
    one_hot_seg = jax.nn.one_hot(segment_id, n_shots, dtype=jnp.int32)
    counts_per_seg = one_hot_seg.sum(axis=0)  # (n_shots,)

    counts_at_first = jnp.where(boundaries, counts_per_seg[segment_id], 0)
    return sorted_bitstrings, counts_at_first


def _stretch_confusion_matrix(qubit_k_values: jax.Array, confusion_matrix: jax.Array) -> jax.Array:
    """``out[i, j] = confusion_matrix[qubit_k_values[i], qubit_k_values[j]]``.

    Note: The natural ``confusion_matrix[ix_(v, v)]`` form trips catalyst's stricter
    gather lowering, so for binary ``qubit_k_values``the same map is expressed
    as ``M @ C @ M.T`` with a one-hot ``M``. All matmul, no advanced indexing.
    """
    one_hot = jnp.stack([1 - qubit_k_values, qubit_k_values], axis=-1).astype(
        confusion_matrix.dtype
    )
    return one_hot @ confusion_matrix @ one_hot.T


# ---------------------------------------------------------------------------
# Top-level entry points. Each is `@jax.jit`'d so that catalyst emits it as
# a single private `func.func` and the @qjit'd kernel ends up calling each
# one exactly once. The end goal is a small top-level function whose body
# is `<callee> -> rem_calibrate -> rem_apply_to_samples`. Each callable
# here is the smallest such unit; helper math used inside lives in the
# private functions above.
# ---------------------------------------------------------------------------


@jax.jit
def rem_calibrate_samples(zeros_samples: jax.Array, ones_samples: jax.Array) -> jax.Array:
    """Build per-qubit (n_qubits, 2, 2) confusion matrices from calibration samples.

    Step 1 of the REM pipeline: consume the all-zeros and all-ones
    calibration samples emitted by the lowered ``mitigation.rem`` op and
    return the normalized confusion matrices.
    """
    rev_zeros = zeros_samples[:, ::-1].astype(jnp.float64)
    rev_ones = ones_samples[:, ::-1].astype(jnp.float64)
    n_zeros_shots = zeros_samples.shape[0]
    n_ones_shots = ones_samples.shape[0]

    # Column 0: all-zeros calibration. ones_per_qubit[k] = count of '1' on qubit k.
    ones_per_qubit_zerocal = rev_zeros.sum(axis=0)
    zeros_per_qubit_zerocal = n_zeros_shots - ones_per_qubit_zerocal
    col_zero = jnp.stack([zeros_per_qubit_zerocal, ones_per_qubit_zerocal], axis=-1)

    # Column 1: all-ones calibration.
    ones_per_qubit_onecal = rev_ones.sum(axis=0)
    zeros_per_qubit_onecal = n_ones_shots - ones_per_qubit_onecal
    col_one = jnp.stack([zeros_per_qubit_onecal, ones_per_qubit_onecal], axis=-1)

    confusion_matrix = jnp.stack([col_zero, col_one], axis=-1)  # (n_qubits, 2, 2)
    norm = jnp.sum(confusion_matrix, axis=1, keepdims=True)
    return confusion_matrix / norm


@jax.jit
def rem_calibrate_probs(zeros_probs: jax.Array, ones_probs: jax.Array) -> jax.Array:
    """Build per-qubit (n_qubits, 2, 2) confusion matrices from calibration probs.

    Step 1 of the REM pipeline: consume the all-zeros and all-ones
    calibration probs emitted by the lowered ``mitigation.rem`` op and
    return the normalized confusion matrices.
    """
    n = zeros_probs.shape[0]
    n_qubits = np.log2(n).astype(jnp.uint64)
    total_zeros = jnp.sum(zeros_probs)
    total_ones = jnp.sum(ones_probs)
    # Generate the full array of indices once: [0, 1, ..., 2^n]
    indices = jnp.arange(n, dtype=jnp.uint32)

    def _process_one_bit(carry, bit_position):

        mask = ((indices >> bit_position) & 1).astype(jnp.float64)
        # Calculate FP and TP for this specific bit via 1D dot products
        bit_fp = jnp.dot(mask, zeros_probs)
        bit_tp = jnp.dot(mask, ones_probs)

        return carry, (bit_fp, bit_tp)

    bit_positions = jnp.arange(n_qubits, dtype=jnp.uint32)
    _, (fp, tp) = jax.lax.scan(_process_one_bit, None, bit_positions)

    tn = total_zeros - fp
    fn = total_ones - tp

    confusion_matrices = jnp.stack(
        [jnp.stack([tn, fp], axis=-1), jnp.stack([fn, tp], axis=-1)], axis=1
    )
    norm = jnp.sum(confusion_matrices, axis=-1, keepdims=True)
    return confusion_matrices / norm


@jax.jit
def rem_calibrate_counts(zeros_counts: jax.Array, ones_counts: jax.Array) -> jax.Array:
    """Build per-qubit (n_qubits, 2, 2) confusion matrices from calibration counts.

    Re-uses `rem_calibrate_probs`, normalizing counts into the [0, 1] range first.
    """
    zeros_probs = zeros_counts / zeros_counts.max()
    ones_probs = ones_counts / ones_counts.max()
    return rem_calibrate_probs(zeros_probs, ones_probs)


@partial(jax.jit, static_argnames=("n_qubits",))
def rem_apply_to_samples(
    mitigatee_samples: jax.Array,
    confusion_matrices: jax.Array,
    measured_qubits: jax.Array,
    n_qubits: int,
) -> Tuple[jax.Array, jax.Array]:
    """Apply per-qubit confusion matrices to mitigatee samples -> mitigated histogram.

    Step 2 of the REM pipeline. Create a reduced transition matrix from confusion
    matrices and mitigatee measurement results, then solve the linear system to
    obtain mitigated counts histogram. The matrix that  goes into the linear
    solve has size ``K x K`` where ``K`` is either ``2**n_qubits`` or
    the total number of bitstrings, ``n_shots``. The Two strategies are
    wired up here and e smaller-``K`` one is chosen at trace time.
    The math is identical in both paths; only the in memory representation differ.

    Returns ``(unique_bitstrings, mitigated_counts)``: in path A the
    bitstrings enumerate the full ``2**n_qubits`` state space (MSB-first);
    in path B they are the sorted ``n_shots`` rows of the mitigatee
    sampled bitstrings, with ``mitigated_counts[i] != 0`` only for
    first-occurrence rows (other rows are pinned to identity in the solve).
    This results in total size of ``n_shots``, where first ``n_unique_bitstrings``
    of elements have meaningful results, while the rest are zero-padded.
    """
    n_shots = mitigatee_samples.shape[0]

    # ======================================================================
    # Path dispatch (TRACE-TIME branch on Python ints `n_qubits` / `n_shots`).
    # ----------------------------------------------------------------------
    # Path A (full histogram): K = 2**n_qubits. Cheap when n_qubits is small
    # but allocates a (n_shots, 2**n_qubits) one-hot intermediate plus a
    # (2**n_qubits, 2**n_qubits) transition matrix. Used only when
    # 2**n_qubits <= n_shots.
    # Path B: K = n_shots. No 2**n_qubits intermediates
    # anywhere - safe for arbitrarily many qubits. The (n_shots, n_shots)
    # transition matrix bounds memory usage.
    # The path that minimizes K (and thus the dominant O(K**3) solve
    # cost) is chosen.
    # ======================================================================
    if 2**n_qubits <= n_shots:
        unique_bitstrings = _bitstrings_for_n_qubits(n_qubits)
        user_counts = _count_shots_histogram(mitigatee_samples, n_qubits).astype(jnp.float64)
    else:
        unique_bitstrings, raw_counts = _sort_based_unique(mitigatee_samples, n_qubits, n_shots)
        user_counts = raw_counts.astype(jnp.float64)

    selected = confusion_matrices[measured_qubits]  # (n_measured, 2, 2)
    stretched = jax.vmap(_stretch_confusion_matrix, in_axes=(1, 0))(
        unique_bitstrings, selected
    )  # (n_measured, K, K)
    transition_probs = jnp.prod(stretched, axis=0)

    # Pin rows/cols for non-counted bitstrings to the identity, so the linear
    # solve sees a well-conditioned reduced system.
    observed = user_counts > 0
    keep_mask = observed[:, None] & observed[None, :]
    identity = jnp.eye(transition_probs.shape[0], dtype=transition_probs.dtype)
    transition_probs = jnp.where(keep_mask, transition_probs, identity)
    transition_probs = transition_probs / jnp.sum(transition_probs, axis=0, keepdims=True)

    mitigated_counts = jnp.linalg.solve(transition_probs, user_counts)
    return unique_bitstrings, mitigated_counts


@partial(jax.jit, static_argnames=("n_qubits",))
def rem_apply_to_counts(
    mitigatee_counts: jax.Array,
    confusion_matrices: jax.Array,
    measured_qubits: jax.Array,
    n_qubits: int,
) -> jax.Array:
    """Apply per-qubit confusion matrices to user counts -> mitigated histogram.

    Note: for CountsMP, the size is always bounded by 2 ** n, therefore
    the histogram path is always dispatched.

    Note: The mitigated counts are returned as float64 instead of int64, due to
    how linear solve works. Further discretization is possible, but leads to
    loss of precision.

    Create a transition matrix from a kronecker product of confusion
    matrices, then solve the linear system to obtain mitigated counts histogram.

    Returns ``mitigated_counts``, where the
    indices enumerate the full ``2**n_qubits`` state space.
    """
    # List comprehension runs at JAX compile time, when `n_qubits` is static and already known.
    # this is then unrolled, just like `reduce`, no loop is present at runtime.
    selected = [
        jnp.linalg.inv(confusion_matrices[measured_qubits[i]]) for i in range(n_qubits)
    ]  # (n_measured, 2, 2)
    inv_transition_matrix = reduce(jnp.kron, selected)

    # transition_probs = transition_probs / jnp.sum(transition_probs, axis=0, keepdims=True)
    # Direct matrix-vector product to solve the system, since inverting 2x2 matrices is much faster
    mitigated_counts = jnp.dot(inv_transition_matrix, mitigatee_counts)
    return mitigated_counts


@partial(jax.jit, static_argnames=("n_qubits",))
def rem_apply_to_probs(
    mitigatee_probs: jax.Array,
    confusion_matrices: jax.Array,
    measured_qubits: jax.Array,
    n_qubits: int,
) -> jax.Array:
    """Apply per-qubit confusion matrices to user probs -> mitigated histogram.

    Note: for ProbsMP, the size is always bounded by 2 ** n, therefore
    the histogram path is always dispatched.

    Create a transition matrix from a kronecker product of confusion
    matrices, then solve the linear system to obtain mitigated probs histogram.

    Returns ``mitigated_probs``, where the
    indices enumerate the full ``2**n_qubits`` state space.
    """
    # List comprehension runs at JAX compile time, when `n_qubits` is static and already known.
    # this is then unrolled, just like `reduce`, no loop is present at runtime.
    selected = [
        jnp.linalg.inv(confusion_matrices[measured_qubits[i]]) for i in range(n_qubits)
    ]  # (n_measured, 2, 2)
    inv_transition_matrix = reduce(jnp.kron, selected)

    # transition_probs = transition_probs / jnp.sum(transition_probs, axis=0, keepdims=True)
    # Direct matrix-vector product to solve the system, since inverting 2x2 matrices is much faster
    mitigated_probs = jnp.dot(inv_transition_matrix, mitigatee_probs)
    return mitigated_probs
