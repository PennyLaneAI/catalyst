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
"""Helpers for attaching Catalyst resource-estimation hints to SCF ops."""

from collections.abc import Sequence

from jaxlib.mlir import ir

ESTIMATED_ITERATIONS_ATTR = "catalyst.estimated_iterations"
ESTIMATED_PROBABILITY_ATTR = "catalyst.estimated_probability"
ESTIMATED_PROBABILITIES_ATTR = "catalyst.estimated_probabilities"


def set_estimated_iterations_attr(op, value: int | float) -> None:
    """Attach a trip-count hint to an ``scf.for`` or ``scf.while`` op."""
    if value is None:
        return
    value = float(value)
    if value < 0:
        raise ValueError(f"'estimated_iterations' must be non-negative, but got {value}.")
    ctx = op.context
    f64_type = ir.F64Type.get(ctx)
    op.attributes[ESTIMATED_ITERATIONS_ATTR] = ir.FloatAttr.get(f64_type, value)


def set_estimated_probability_attr(op, value: float) -> None:
    """Attach a branch probability hint to an ``scf.if`` op."""
    if value is None:
        return
    ctx = op.context
    f64_type = ir.F64Type.get(ctx)
    op.attributes[ESTIMATED_PROBABILITY_ATTR] = ir.FloatAttr.get(f64_type, value)


def set_estimated_probabilities_attr(op, values: Sequence[float]) -> None:
    """Attach branch probability hints to an ``scf.index_switch`` op."""
    if values is None:
        return
    ctx = op.context
    f64_type = ir.F64Type.get(ctx)
    attrs = [ir.FloatAttr.get(f64_type, value) for value in values]
    op.attributes[ESTIMATED_PROBABILITIES_ATTR] = ir.ArrayAttr.get(attrs)


def normalize_estimated_probabilities_for_cond(
    value: float | Sequence[float] | None, num_branches: int
) -> tuple[float, ...] | None:
    """Normalize and validate probability hints for ``cond``."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        probs = (float(value),)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        probs = tuple(float(v) for v in value)
    else:
        raise TypeError(
            "'estimated_probabilities' must be a float or sequence of floats in [0, 1]."
        )

    for p in probs:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"'estimated_probabilities' must be in [0, 1], but got {p}.")
    if sum(probs) > 1.0 + 1e-10:
        raise ValueError(
            f"'estimated_probabilities' entries must sum to at most 1, but got {sum(probs)}."
        )
    if len(probs) != num_branches:
        raise ValueError(
            f"'estimated_probabilities' must have one entry per non-default branch, but got "
            f"{len(probs)} probabilities for {num_branches} branch(es)."
        )
    return probs


def unconditional_to_conditional_if_probs(probs: Sequence[float] | None) -> tuple[float, ...] | None:
    """Convert unconditional branch probabilities to per-``scf.if`` conditional probabilities.

    ``qp.cond`` with ``elif`` branches lowers to nested ``scf.if`` ops.
    Resource analysis expects each ``scf.if`` to carry the probability that its "then" branch is
    taken *at that decision point*, so we convert from the user-facing unconditional branch
    probabilities to those conditional probabilities that need to be passed to the ``scf.if``.
    """
    if probs is None:
        return None
    conditional = []
    remaining = 1.0
    for p in probs:
        if remaining <= 0.0:
            conditional.append(0.0)
        else:
            conditional.append(p / remaining)
            remaining -= p
    return tuple(conditional)
