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
from jax._src.core import abstractify, standard_vma_rule
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
from jax.core import AbstractValue, Tracer

__all__ = (
    "_drop_unused_vars2",
    "get_aval2",
    "_no_clean_up_dead_vars",
    "_gather_shape_rule_dynamic",
    "gather2_p",
    "patch_primitives",
)


def mock_attribute(obj, mock_attribute_name, mock_attribute_value):
    """Mock the attribute of an object by returning a wrapper."""

    class MockAttributeWrapper:
        """Wrapper to mock the attribute of an object."""

        def __init__(self, original):
            self.original = original

        def __getattr__(self, name):
            if name == mock_attribute_name:
                return mock_attribute_value
            return getattr(self.original, name)

    return MockAttributeWrapper(obj)


def _drop_unused_vars2(
    constvars, constvals, eqns=None, outvars=None
):  # pylint: disable=unused-argument
    """
    A patch to not drop unused vars during classical tracing of control flow.

    This function matches the JAX 0.7 signature but doesn't actually drop any vars.
    Returns the constvars and constvals unchanged (except converting constvals to list).
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


# TODO: Remove this patch when JAX/PennyLane are updated to use the new JAX 0.7+ API.
# pylint: disable=protected-access
def patch_primitives():
    """Patch PennyLane/JAX primitives to make them compatible with JAX 0.7+.

    JAX 0.7+ requires all primitive parameters to be hashable, but PennyLane
    passes **lists** for some parameters like control_values, jaxpr_branches, etc.
    This patch wraps the bind method to convert lists to tuples to make them hashable.
    """

    def make_hashable(value):
        """Recursively convert lists to tuples to make them hashable."""
        if isinstance(value, list):
            return tuple(make_hashable(item) for item in value)
        return value

    try:
        from pennylane.capture.primitives import ctrl_transform_prim
        from pennylane.capture.primitives import cond_prim
        from pennylane.ops.op_math.controlled import Controlled
        from jax._src.interpreters import partial_eval as pe

        original_ctrl_bind = ctrl_transform_prim.bind
        original_cond_bind = cond_prim.bind
        original_controlled_bind = Controlled._primitive.bind
        original_drop_unused_vars = pe._drop_unused_vars

        def patched_ctrl_bind(*args, **kwargs):
            # Convert control_values from list to tuple if present
            if "control_values" in kwargs:
                kwargs["control_values"] = make_hashable(kwargs["control_values"])
            return original_ctrl_bind(*args, **kwargs)

        def patched_cond_bind(*args, **kwargs):
            # Convert list parameters to tuples
            if "jaxpr_branches" in kwargs:
                kwargs["jaxpr_branches"] = make_hashable(kwargs["jaxpr_branches"])
            if "consts_slices" in kwargs:
                kwargs["consts_slices"] = make_hashable(kwargs["consts_slices"])
            return original_cond_bind(*args, **kwargs)

        def patched_controlled_bind(*args, **kwargs):
            # Convert control_values from list to tuple if present
            if "control_values" in kwargs:
                kwargs["control_values"] = make_hashable(kwargs["control_values"])
            return original_controlled_bind(*args, **kwargs)

        def patched_drop_unused_vars(
            constvars, constvals, eqns=None, outvars=None
        ):  # pylint: disable=unused-argument
            constvars, constvals = original_drop_unused_vars(constvars, constvals, eqns, outvars)
            return constvars, list(constvals)

        # Replace the bind method
        ctrl_transform_prim.bind = patched_ctrl_bind
        cond_prim.bind = patched_cond_bind
        Controlled._primitive.bind = patched_controlled_bind
        pe._drop_unused_vars = patched_drop_unused_vars

    except ImportError:
        pass

    # patch DynamicJaxprTrace members: makevar and getvar
    try:
        from jax._src.interpreters import partial_eval as pe

        def patched_makevar(self, tracer):
            assert tracer.val is None, "a jaxpr variable must be created only once per tracer"
            tracer.val = self.frame.newvar(tracer.aval)
            return tracer.val

        def patched_getvar(self, tracer):  # pylint: disable=unused-argument
            if var := tracer.val:
                return var
            raise jax.core.escaped_tracer_error(tracer)

        pe.DynamicJaxprTrace.makevar = patched_makevar
        pe.DynamicJaxprTrace.getvar = patched_getvar

        # Patch make_eqn to handle both single aval and list of avals
        # original_make_eqn = pe.DynamicJaxprTrace.make_eqn

        import jax._src.source_info_util as source_info_util
        from jax._src.core import JaxprEqnContext, Var
        from jax._src.interpreters.partial_eval import DynamicJaxprTracer
        from jax._src.interpreters.partial_eval import TracingEqn
        from jax._src.interpreters.partial_eval import compute_on
        from jax._src import config
        from jax._src.interpreters.partial_eval import xla_metadata_lib

        def internal_make_eqn(
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
            source_info = source_info or source_info_util.new_source_info()
            ctx = ctx or JaxprEqnContext(
                compute_on.current_compute_type(),
                config.threefry_partitionable.value,
                xla_metadata_lib.current_xla_metadata(),
            )

            if out_tracers is not None:
                outvars = [tracer.val for tracer in out_tracers]
                if config.enable_checks.value:
                    assert all(isinstance(x, DynamicJaxprTracer) for x in in_tracers)
                    assert all(isinstance(v, Var) for v in outvars)
                eqn = TracingEqn(in_tracers, outvars, primitive, params, effects, source_info, ctx)
                return eqn, out_tracers
            else:
                outvars = list(map(lambda aval: self.frame.newvar(aval), out_avals))
                if config.enable_checks.value:
                    assert all(isinstance(x, DynamicJaxprTracer) for x in in_tracers)
                    assert all(isinstance(v, Var) for v in outvars)
                eqn = TracingEqn(in_tracers, outvars, primitive, params, effects, source_info, ctx)
                out_tracers = [
                    DynamicJaxprTracer(self, aval, v, source_info, eqn)
                    for aval, v in zip(out_avals, outvars)
                ]
                return eqn, out_tracers

        pe.DynamicJaxprTrace.make_eqn_internal = internal_make_eqn

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
            # Normalize out_avals to a list if it's a single AbstractValue
            if not isinstance(out_avals, (list, tuple)):
                # It's a single aval, wrap it in a list
                out_avals = [out_avals]
                eqn, out_tracers = self.make_eqn_internal(
                    in_tracers,
                    out_avals,
                    primitive,
                    params,
                    effects,
                    source_info,
                    ctx,
                    out_tracers,
                )
                # Return single tracer instead of list
                return eqn, out_tracers[0] if len(out_tracers) == 1 else out_tracers
            else:
                return self.make_eqn_internal(
                    in_tracers,
                    out_avals,
                    primitive,
                    params,
                    effects,
                    source_info,
                    ctx,
                    out_tracers,
                )

        pe.DynamicJaxprTrace.make_eqn = patched_make_eqn

        # Patch eqns property
        def patched_eqns_getter(self):
            return self.tracing_eqns

        def patched_eqns_setter(self, value):
            self.tracing_eqns = value

        pe.JaxprStackFrame.eqns = property(patched_eqns_getter, patched_eqns_setter)

        import jax._src.lax.lax as lax
        import jax._src.core as core

        def patched_dyn_shape_staging_rule(trace, source_info, prim, out_aval, *args, **params):
            eqn, out_tracer = trace.make_eqn(
                args, out_aval, prim, params, core.no_effects, source_info
            )
            trace.frame.add_eqn(eqn)
            return out_tracer

        lax._dyn_shape_staging_rule = patched_dyn_shape_staging_rule

    except ImportError:
        pass
