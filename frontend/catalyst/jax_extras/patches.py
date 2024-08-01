# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Jax extras module containing Jax patches  """

# pylint: disable=too-many-arguments

from __future__ import annotations

import jax
from jax._src.lax.lax import _nary_lower_hlo
from jax._src.lax.slicing import (
    _gather_shape_computation,
    _is_sorted,
    _no_duplicate_dims,
    _rank,
    _sorted_dims_in_range,
)
from jax._src.lib.mlir.dialects import hlo
from jax.core import AbstractValue, Tracer, concrete_aval

__all__ = (
    "get_aval2",
    "_no_clean_up_dead_vars",
    "_gather_shape_rule_dynamic",
    "_sin_lowering2",
    "_cos_lowering2",
)


def get_aval2(x):
    """An extended version of `jax.core.get_aval` which also accepts AbstractValues."""
    # TODO: remove this patch when https://github.com/google/jax/pull/18579 is merged
    if isinstance(x, AbstractValue):
        return x
    elif isinstance(x, Tracer):
        return x.aval
    else:
        return concrete_aval(x)


def _no_clean_up_dead_vars(_eqn, _env, _last_used):
    """A stub to workaround the Jax ``KeyError 'a'`` bug during the lowering of Jaxpr programs to
    MLIR with the dynamic API enabled."""
    return None


def _gather_shape_rule_dynamic(
    operand,
    indices,
    *,
    dimension_numbers,
    slice_sizes,
    unique_indices,
    indices_are_sorted,
    mode,
    fill_value,
):  # pragma: no cover
    """Validates the well-formedness of the arguments to Gather. Compared to the original version,
    this implementation skips static shape checks if variable dimensions are used.

    This function has been modified from its original form in the JAX project at
    https://github.com/google/jax/blob/88a60b808c1f91260cc9e75b9aa2508aae5bc9f9/jax/_src/lax/slicing.py#L1438
    version released under the Apache License, Version 2.0, with the following copyright notice:

    Copyright 2021 The JAX Authors.
    TODO(@grwlf): delete once PR https://github.com/google/jax/pull/19083 has been merged
    """
    # pylint: disable=unused-argument
    # pylint: disable=too-many-branches
    # pylint: disable=consider-using-enumerate
    # pylint: disable=chained-comparison
    offset_dims = dimension_numbers.offset_dims
    collapsed_slice_dims = dimension_numbers.collapsed_slice_dims
    start_index_map = dimension_numbers.start_index_map

    # Note: in JAX, index_vector_dim is always computed as below, cf. the
    # documentation of the GatherDimensionNumbers class.
    index_vector_dim = _rank(indices) - 1

    # This case should never happen in JAX, due to the implicit construction of
    # index_vector_dim, but is included for completeness.
    if _rank(indices) < index_vector_dim or index_vector_dim < 0:
        raise TypeError(
            f"Gather index leaf dimension must be within [0, rank("
            f"indices) + 1). rank(indices) is {_rank(indices)} and "
            f"gather index leaf dimension is {index_vector_dim}."
        )

    # Start ValidateGatherDimensions
    # In the error messages output by XLA, "offset_dims" is called "Output window
    # dimensions" in error messages. For consistency's sake, our error messages
    # stick to "offset_dims".
    _is_sorted(offset_dims, "gather", "offset_dims")
    _no_duplicate_dims(offset_dims, "gather", "offset_dims")

    output_offset_dim_count = len(offset_dims)
    output_shape_rank = len(offset_dims) + _rank(indices) - 1

    for i in range(output_offset_dim_count):
        offset_dim = offset_dims[i]
        if offset_dim < 0 or offset_dim >= output_shape_rank:
            raise TypeError(
                f"Offset dimension {i} in gather op is out of bounds; "
                f"got {offset_dim}, but should have been in "
                f"[0, {output_shape_rank})"
            )

    if len(start_index_map) != indices.shape[index_vector_dim]:
        raise TypeError(
            f"Gather op has {len(start_index_map)} elements in "
            f"start_index_map and the bound of dimension "
            f"{index_vector_dim=} of indices is "
            f"{indices.shape[index_vector_dim]}. These two "
            f"numbers must be equal."
        )

    for i in range(len(start_index_map)):
        operand_dim_for_start_index_i = start_index_map[i]
        if operand_dim_for_start_index_i < 0 or operand_dim_for_start_index_i >= _rank(operand):
            raise TypeError(
                f"Invalid start_index_map; domain is "
                f"[0, {_rank(operand)}), got: "
                f"{i}->{operand_dim_for_start_index_i}."
            )

    _no_duplicate_dims(start_index_map, "gather", "start_index_map")

    # _is_sorted and _sorted_dims_in_range are checked in the opposite order
    # compared to the XLA implementation. In cases when the input is not sorted
    # AND there are problematic collapsed_slice_dims, the error message will thus
    # be different.
    _is_sorted(collapsed_slice_dims, "gather", "collapsed_slice_dims")
    _sorted_dims_in_range(collapsed_slice_dims, _rank(operand), "gather", "collapsed_slice_dims")
    _no_duplicate_dims(collapsed_slice_dims, "gather", "collapsed_slice_dims")
    # End ValidateGatherDimensions

    if _rank(operand) != len(slice_sizes):
        raise TypeError(
            f"Gather op must have one slice size for every input "
            f"dimension; got: len(slice_sizes)={len(slice_sizes)}, "
            f"input_shape.rank={_rank(operand)}"
        )

    if len(slice_sizes) != len(offset_dims) + len(collapsed_slice_dims):
        raise TypeError(
            f"All components of the offset index in a gather op must "
            f"either be a offset dimension or explicitly collapsed; "
            f"got len(slice_sizes)={len(slice_sizes)}, "
            f"output_slice_sizes={offset_dims}, collapsed_slice_dims="
            f"{collapsed_slice_dims}."
        )

    # This section contains a patch suggested to the upstream.
    for i in range(len(slice_sizes)):
        slice_size = slice_sizes[i]
        corresponding_input_size = operand.shape[i]

        if jax.core.is_constant_dim(corresponding_input_size):
            if not (slice_size >= 0 and corresponding_input_size >= slice_size):
                raise TypeError(
                    f"Slice size at index {i} in gather op is out of range, "
                    f"must be within [0, {corresponding_input_size} + 1), "
                    f"got {slice_size}."
                )

    for i in range(len(collapsed_slice_dims)):
        bound = slice_sizes[collapsed_slice_dims[i]]
        if bound != 1:
            raise TypeError(
                f"Gather op can only collapse slice dims with bound 1, "
                f"but bound is {bound} for index "
                f"{collapsed_slice_dims[i]} at position {i}."
            )

    return _gather_shape_computation(indices, dimension_numbers, slice_sizes)


def _sin_lowering2(ctx, x):
    """Use hlo.sine lowering instead of the new sin lowering from jax 0.4.28"""
    return _nary_lower_hlo(hlo.sine, ctx, x)


def _cos_lowering2(ctx, x):
    """Use hlo.cosine lowering instead of the new cosine lowering from jax 0.4.28"""
    return _nary_lower_hlo(hlo.cosine, ctx, x)
