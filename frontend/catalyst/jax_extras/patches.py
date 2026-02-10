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
"""Jax extras module containing Jax patches"""

# pylint: disable=too-many-arguments

from __future__ import annotations

from functools import partial

import jax
import jax._src.interpreters.partial_eval as pe
from jax._src import config, core, source_info_util
from jax._src.core import JaxprEqnContext, abstractify, standard_vma_rule
from jax._src.interpreters import mlir
from jax._src.interpreters.partial_eval import (
    DynamicJaxprTracer,
    TracingEqn,
    compute_on,
    xla_metadata_lib,
)
from jax._src.lax.slicing import (
    _argnum_weak_type,
    _gather_dtype_rule,
    _gather_shape_computation,
    _gather_sharding_rule,
    _is_sorted,
    _no_duplicate_dims,
    _rank,
    _sorted_dims_in_range,
    standard_primitive,
)
from jax._src.pjit import _out_type, _pjit_forwarding, jit_p
from jax._src.sharding_impls import UnspecifiedValue
from jax.core import AbstractValue, Tracer

__all__ = (
    "_drop_unused_vars2",
    "get_aval2",
    "_no_clean_up_dead_vars",
    "_gather_shape_rule_dynamic",
    "gather2_p",
    "patched_drop_unused_vars",
    "patched_make_eqn",
    "patched_dyn_shape_staging_rule",
    "patched_pjit_staging_rule",
    "patched_multi_broadcast_in_dim",
)


def mock_attributes(obj, attrs: dict[str, any]):
    """Mock the attribute of an object by returning a wrapper.

    Args:
        obj: The object to mock the attributes of.
        attrs: A dictionary of attributes to mock.
            Example: {"attribute_name": attribute_value}
    """

    class MockAttributeWrapper:
        """Wrapper to mock the attribute of an object."""

        def __init__(self, original):
            self.original = original

        def __getattr__(self, name):
            if name in attrs:
                return attrs[name]
            return getattr(self.original, name)

    return MockAttributeWrapper(obj)


def _drop_unused_vars2(
    constvars, constvals, eqns=None, outvars=None
):  # pylint: disable=unused-argument
    """
    A patch to not drop unused vars during classical tracing of control flow.
    """
    return constvars, list(constvals)


def get_aval2(x):
    """An extended version of `jax.core.get_aval` which also accepts AbstractValues."""
    # TODO: remove this patch when https://github.com/google/jax/pull/18579 is merged
    if isinstance(x, AbstractValue):
        return x
    elif isinstance(x, Tracer):
        return x.aval
    else:
        return abstractify(x)


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


# TODO: See the `_gather_shape_rule_dynamic` comment. Remove once the upstream change is
# applied.
gather2_p = standard_primitive(
    _gather_shape_rule_dynamic,
    _gather_dtype_rule,
    "gather",
    weak_type_rule=_argnum_weak_type(0),
    sharding_rule=_gather_sharding_rule,
    vma_rule=partial(standard_vma_rule, "gather"),
)

# pylint: disable=protected-access
original_drop_unused_vars = pe._drop_unused_vars


# pylint: disable=too-many-function-args
def patched_drop_unused_vars(constvars, constvals, eqns=None, outvars=None):
    """Patched drop_unused_vars to ensure constvals is a list."""
    constvars, constvals = original_drop_unused_vars(constvars, constvals, eqns, outvars)
    return constvars, list(constvals)


# pylint: disable=too-many-positional-arguments
def patched_make_eqn(
    self,
    in_tracers,
    out_avals,
    primitive,
    params,
    effects,
    source_info=None,
    ctx=None,
    out_tracers=None,
):
    """Patched make_eqn for DynamicJaxprTrace"""

    # Helper function (replaces make_eqn_internal)
    def make_eqn_internal(out_avals_list, out_tracers):
        source_info_final = source_info or source_info_util.new_source_info()
        ctx_final = ctx or JaxprEqnContext(
            compute_on.current_compute_type(),
            config.threefry_partitionable.value,
            xla_metadata_lib.current_xla_metadata(),
        )

        if out_tracers is not None:
            outvars = [tracer.val for tracer in out_tracers]
            eqn = TracingEqn(
                in_tracers,
                outvars,
                primitive,
                params,
                effects,
                source_info_final,
                ctx_final,
            )
            return eqn, out_tracers
        else:
            outvars = list(map(self.frame.newvar, out_avals_list))
            eqn = TracingEqn(
                in_tracers,
                outvars,
                primitive,
                params,
                effects,
                source_info_final,
                ctx_final,
            )
            out_tracers_new = [
                DynamicJaxprTracer(self, aval, v, source_info_final, eqn)
                for aval, v in zip(out_avals_list, outvars)
            ]
            return eqn, out_tracers_new

    # Normalize out_avals to a list if it's a single AbstractValue
    if not isinstance(out_avals, (list, tuple)):
        # It's a single aval, wrap it in a list
        out_avals_list = [out_avals]
        eqn, out_tracers_result = make_eqn_internal(out_avals_list, out_tracers)
        # Return single tracer instead of list
        return eqn, (out_tracers_result[0] if len(out_tracers_result) == 1 else out_tracers_result)
    else:
        return make_eqn_internal(out_avals, out_tracers)


def patched_multi_broadcast_in_dim(ctx, ops, ops_avals, out_shape, out_sharding=None):
    """Patched version that uses DShapedArray for dynamic shapes."""
    out = []
    for op, op_aval in zip(ops, ops_avals):
        op_aval_shape = op_aval.shape

        # Use DShapedArray if shape contains dynamic dimensions
        if core.is_constant_shape(out_shape):
            out_aval = core.ShapedArray(out_shape, op_aval.dtype, sharding=out_sharding)
        else:
            # DShapedArray doesn't support sharding parameter
            out_aval = core.DShapedArray(
                out_shape, op_aval.dtype, weak_type=getattr(op_aval, "weak_type", False)
            )

        if core.definitely_equal_shape(op_aval_shape, out_shape):
            out.append(op)
        else:
            assert len(op_aval_shape) <= len(out_shape), (op_aval_shape, out_shape)
            broadcast_dimensions = list(range(len(out_shape) - len(op_aval_shape), len(out_shape)))
            b_out = mlir.broadcast_in_dim(
                ctx, op, out_aval, broadcast_dimensions=broadcast_dimensions
            )
            b_out = mlir.lower_with_sharding_in_types(ctx, b_out, out_aval)
            out.append(b_out)
    return out


def patched_dyn_shape_staging_rule(trace, source_info, prim, out_aval, *args, **params):
    """Patched _dyn_shape_staging_rule for dynamic shape handling."""
    eqn, out_tracer = trace.make_eqn(args, out_aval, prim, params, core.no_effects, source_info)
    trace.frame.add_eqn(eqn)
    return out_tracer


def patched_pjit_staging_rule(trace, source_info, *args, **params):
    """Patched pjit_staging_rule for pjit compatibility."""
    # If we're inlining, no need to compute forwarding information; the inlined
    # computation will in effect forward things.
    if (
        params["inline"]
        and all(isinstance(i, UnspecifiedValue) for i in params["in_shardings"])
        and all(isinstance(o, UnspecifiedValue) for o in params["out_shardings"])
        and all(i is None for i in params["in_layouts"])
        and all(o is None for o in params["out_layouts"])
    ):
        jaxpr = params["jaxpr"]
        if config.dynamic_shapes.value:
            # Inline jaxpr doesn't handle dynamic shapes when inlining. If dynamic
            # shapes are enabled, use eval_jaxpr, which uses the tracing machinery,
            # but redundantly performs abstract evaluation again.
            with core.set_current_trace(trace):
                out = core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args, propagate_source_info=False)
        else:
            out = pe.inline_jaxpr_into_trace(trace, source_info, jaxpr.jaxpr, jaxpr.consts, *args)
        return [trace.to_jaxpr_tracer(x, source_info) for x in out]

    jaxpr = params["jaxpr"]
    if config.dynamic_shapes.value:
        jaxpr, in_fwd, out_shardings, out_layouts = _pjit_forwarding(
            jaxpr, params["out_shardings"], params["out_layouts"]
        )
        params = dict(params, jaxpr=jaxpr, out_shardings=out_shardings, out_layouts=out_layouts)
        outvars = list(map(trace.frame.newvar, _out_type(jaxpr)))
        out_avals = [v.aval for v in outvars]
        out_tracers = [
            pe.DynamicJaxprTracer(trace, aval, v, source_info)
            for aval, v in zip(out_avals, outvars)
        ]
        eqn, out_tracers = trace.make_eqn(
            args,
            out_avals,
            jit_p,
            params,
            jaxpr.effects,
            source_info,
            out_tracers=out_tracers,
        )
        trace.frame.add_eqn(eqn)
        out_tracers_ = iter(out_tracers)
        out_tracers = [args[f] if isinstance(f, int) else next(out_tracers_) for f in in_fwd]
        assert next(out_tracers_, None) is None
    else:
        out_tracers = trace.default_process_primitive(jit_p, args, params, source_info=source_info)
    return out_tracers
