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
""" Jax extras module containing functions related to the Python program tracing  """

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Set, Type

import jax
from jax import ShapeDtypeStruct
from jax._src import state, util
from jax._src.core import _update_thread_local_jit_state
from jax._src.interpreters.mlir import _module_name_regex, register_lowering
from jax._src.interpreters.partial_eval import (
    _input_type_to_tracers,
    infer_lambda_input_type,
    trace_to_jaxpr_dynamic2,
)
from jax._src.lax.control_flow import _initial_style_jaxpr, _initial_style_open_jaxpr
from jax._src.lax.lax import _abstractify
from jax._src.lax.slicing import (
    _argnum_weak_type,
    _gather_dtype_rule,
    _gather_lower,
    standard_primitive,
)
from jax._src.linear_util import annotate
from jax._src.pjit import _extract_implicit_args, _flat_axes_specs
from jax._src.source_info_util import current as jax_current
from jax._src.util import partition_list, safe_map, unzip2, unzip3, wraps
from jax.api_util import flatten_fun
from jax.core import ClosedJaxpr, Jaxpr, JaxprEqn, MainTrace, OutputType
from jax.core import Primitive as JaxprPrimitive
from jax.core import ShapedArray, Trace, eval_jaxpr, gensym, thread_local_state
from jax.extend.linear_util import wrap_init
from jax.interpreters.partial_eval import (
    DynamicJaxprTrace,
    DynamicJaxprTracer,
    convert_constvars_jaxpr,
    make_jaxpr_effects,
)
from jax.lax import convert_element_type
from jax.tree_util import (
    PyTreeDef,
    tree_flatten,
    tree_structure,
    tree_unflatten,
    treedef_is_leaf,
)
from jaxlib.xla_extension import PyTreeRegistry

from catalyst.jax_extras.patches import _gather_shape_rule_dynamic, get_aval2
from catalyst.utils.patching import Patcher

# pylint: disable=protected-access

__all__ = (
    "ClosedJaxpr",
    "DynshapedClosedJaxpr",
    "DynamicJaxprTrace",
    "DynamicJaxprTracer",
    "Jaxpr",
    "PyTreeDef",
    "PyTreeRegistry",
    "ShapedArray",
    "ShapeDtypeStruct",
    "convert_constvars_jaxpr",
    "convert_element_type",
    "eval_jaxpr",
    "initial_style_jaxprs_with_common_consts1",
    "initial_style_jaxprs_with_common_consts2",
    "_abstractify",
    "_initial_style_jaxpr",
    "_input_type_to_tracers",
    "_module_name_regex",
    "make_jaxpr_effects",
    "make_jaxpr2",
    "new_dynamic_main2",
    "new_inner_tracer",
    "sort_eqns",
    "transient_jax_config",
    "treedef_is_leaf",
    "tree_flatten",
    "tree_structure",
    "tree_unflatten",
    "unzip2",
    "wrap_init",
)

map, unsafe_map = safe_map, map  # pylint: disable=redefined-builtin


class DynshapedClosedJaxpr(ClosedJaxpr):
    """A wrapper class to handle implicit/explicit result information used by JAX for dynamically
    shaped arrays. Can be used inplace of any other ClosedJaxpr instance."""

    def __init__(self, jaxpr: Jaxpr, consts: Sequence, output_type: OutputType):
        super().__init__(jaxpr, consts)
        self.output_type = output_type

    def remove_implicit_results(self):
        """Remove all implicit result values from this JAXPR.

        Returns:
            ClosedJaxpr
        """
        # Note: a more idiomatic way of doing this would be to re-trace the jaxpr and drop the
        # unneeded tracers.
        if not self.output_type:
            return self

        jaxpr = self.jaxpr
        out_keep = tuple(zip(*self.output_type))[1]
        outvars = [o for o, keep in zip(jaxpr._outvars, out_keep) if keep]
        filtered_jaxpr = Jaxpr(
            jaxpr.constvars, jaxpr.invars, outvars, jaxpr.eqns, jaxpr.effects, jaxpr.debug_info
        )

        return ClosedJaxpr(filtered_jaxpr, self.consts)


@contextmanager
def transient_jax_config() -> Generator[None, None, None]:
    """Context manager which updates transient JAX configuration options,
    yields, and then restores the original configuration values.
    """
    want_vals = {"jax_dynamic_shapes": True}
    prev_vals = {}

    for name, val in want_vals.items():
        # Using ``read()`` to retrieve the value of an option is not permitted
        # for JAX context manager flags.
        prev_vals[name] = jax.config.values[name]
        jax.config.update(name, val)

    yield

    for name, val in prev_vals.items():
        jax.config.update(name, val)


@contextmanager
def new_dynamic_main2(
    trace_type: Type[Trace],
    main: Optional[MainTrace] = None,
    **payload,
) -> Generator[MainTrace, None, None]:
    """A verison of JAX `new_main` function that knows how to re-use an already existing `MainTrace`
    object"""

    stack = thread_local_state.trace_state.trace_stack
    level = stack.next_level() if main is None else main.level
    main = MainTrace(level, trace_type, **payload) if main is None else main
    stack.push(main)
    prev_dynamic, stack.dynamic = stack.dynamic, main
    _update_thread_local_jit_state(stack.dynamic)

    try:
        yield main
    finally:
        stack.pop()
        stack.dynamic = prev_dynamic
        _update_thread_local_jit_state(stack.dynamic)


def stable_toposort(end_nodes: list) -> list:
    """Stable topoligy sorting. Input objects are required to have `id` and `parents` members.

    Args:
        end_nodes (List of objects): Objects to sort
    """
    if not end_nodes:
        return []
    # end_nodes = _remove_duplicates(end_nodes)

    child_counts = {}
    stack = list(end_nodes)
    while stack:
        node = stack.pop()
        if node.id in child_counts:
            child_counts[node.id] += 1
        else:
            child_counts[node.id] = 1
            stack.extend(node.parents)
    for node in end_nodes:
        child_counts[node.id] -= 1

    sorted_nodes = []
    childless_nodes = [node for node in end_nodes if child_counts[node.id] == 0]
    assert childless_nodes
    while childless_nodes:
        node = childless_nodes.pop()
        sorted_nodes.append(node)
        for parent in node.parents:
            if child_counts[parent.id] == 1:
                childless_nodes.append(parent)
            else:
                child_counts[parent.id] -= 1
    sorted_nodes = sorted_nodes[::-1]

    # check_toposort
    visited = set()
    for node in sorted_nodes:
        assert all(parent.id in visited for parent in node.parents)
        visited.add(node.id)
    return sorted_nodes


def sort_eqns(eqns: List[JaxprEqn], forced_order_primitives: Set[JaxprPrimitive]) -> List[JaxprEqn]:
    """Topologically sort JAXRR equations in a unsorted list of equations, based on their
    input/output variables and additional criterias."""

    # The procedure goes as follows: [1] - initialize the `origin` map mapping variable identifiers
    # to the origin Boxed equations, [2] - initialize `parents` fields of boxes with the
    # correct values, [3] - add additional equation order restrictions to boxes, [4] - call the
    # topological sorting.

    class Box:
        """Wrapper for JaxprEqn keeping track of its id and parents."""

        def __init__(self, boxid: int, e: JaxprEqn):
            self.id: int = boxid
            self.e: JaxprEqn = e
            self.parents: List["Box"] = []  # to be filled later

    boxes = [Box(i, e) for i, e in enumerate(eqns)]
    fixedorder = [(i, b) for (i, b) in enumerate(boxes) if b.e.primitive in forced_order_primitives]
    origin: Dict[int, Box] = {}
    for b in boxes:
        origin.update({ov.count: b for ov in b.e.outvars})  # [1]
    for b in boxes:
        b.parents = [origin[v.count] for v in b.e.invars if v.count in origin]  # [2]
    for i, q in fixedorder:
        for b in boxes[i + 1 :]:
            b.parents.append(q)  # [3]
    return [b.e for b in stable_toposort(boxes)]  # [4]


def initial_style_jaxprs_with_common_consts1(
    funs: Sequence[Callable], in_tree, in_avals, primitive_name: str
):
    """This function is the head (shorter) part of the original
    `lax.control_flow.common._initial_style_jaxprs_with_common_consts` of JAX. The algorithm is the
    same at the time of this writing, we use this function only to avoid conflicts with future
    versions of JAX.
    """
    jaxprs, all_consts, all_out_trees = unzip3(
        _initial_style_open_jaxpr(fun, in_tree, in_avals, primitive_name) for fun in funs
    )
    closed_jaxprs, consts = initial_style_jaxprs_with_common_consts2(jaxprs, all_consts)
    return closed_jaxprs, consts, all_out_trees


def initial_style_jaxprs_with_common_consts2(jaxprs, all_consts):
    """This function is the tail (largest) part of the
    `lax.control_flow.common._initial_style_jaxprs_with_common_consts` of JAX. The JAX version
    traces argument Python functions in order to determine signatures to be unified. Here we rely on
    the fact that the tracing was already done elsewhere - and this is the only difference.
    """

    all_const_avals = [map(_abstractify, consts) for consts in all_consts]
    for consts_avals in all_const_avals:
        for aval in consts_avals:
            assert not isinstance(
                aval, state.AbstractRef
            ), "AbstractRefs are not supported in this Catalyst version of this function"
    canonical_ref_indices = []
    canonical_refs: List[Any] = []
    all_nonref_consts = []
    canonical_ref_avals = []
    all_nonref_const_avals = []
    for consts, consts_avals in zip(all_consts, all_const_avals):
        ref_indices = []
        nonref_consts = []
        nonref_const_avals = []
        for c, aval in zip(consts, consts_avals):
            assert not isinstance(
                aval, state.AbstractRef
            ), "AbstractRefs are not supported in this Catalyst version of this function"
            nonref_consts.append(c)
            nonref_const_avals.append(aval)
        all_nonref_consts.append(nonref_consts)
        all_nonref_const_avals.append(nonref_const_avals)
        canonical_ref_indices.append(ref_indices)

    newvar = gensym(jaxprs, suffix="_")
    unused_ref_const_vars = map(newvar, canonical_ref_avals)
    unused_const_vars = [map(newvar, const_avals) for const_avals in all_nonref_const_avals]

    def pad_jaxpr_constvars(i, jaxpr):
        is_ref = [isinstance(v.aval, state.AbstractRef) for v in jaxpr.constvars]
        nonref_constvars, ref_constvars = partition_list(is_ref, jaxpr.constvars)
        padded_ref_constvars = unused_ref_const_vars[:]
        for canonical_id, ref_var in zip(canonical_ref_indices[i], ref_constvars):
            padded_ref_constvars[canonical_id] = ref_var  # pragma: no cover
        const_prefix = util.concatenate(unused_const_vars[:i])
        const_suffix = util.concatenate(unused_const_vars[i + 1 :])
        constvars = [*padded_ref_constvars, *const_prefix, *nonref_constvars, *const_suffix]
        jaxpr = jaxpr.replace(constvars=constvars)
        effects = make_jaxpr_effects(jaxpr.constvars, jaxpr.invars, jaxpr.outvars, jaxpr.eqns)
        jaxpr = jaxpr.replace(effects=effects)
        return jaxpr

    consts = [*canonical_refs, *util.concatenate(all_nonref_consts)]
    jaxprs = tuple(pad_jaxpr_constvars(i, jaxpr) for i, jaxpr in enumerate(jaxprs))
    closed_jaxprs = [ClosedJaxpr(convert_constvars_jaxpr(jaxpr), ()) for jaxpr in jaxprs]
    return closed_jaxprs, consts


def deduce_avals(f: Callable, args, kwargs):
    """Wraps the callable ``f`` into a WrappedFun container accepting collapsed flatten arguments
    and returning expanded flatten results. Calculate input abstract values and output_tree promise.
    The promise must be called after the resulting wrapped function is evaluated."""
    # TODO: deprecate in favor of `deduce_signatures`
    flat_args, in_tree = tree_flatten((args, kwargs))
    abstracted_axes = None
    axes_specs = _flat_axes_specs(abstracted_axes, *args, **kwargs)
    in_type = infer_lambda_input_type(axes_specs, flat_args)
    in_avals, keep_inputs = unzip2(in_type)
    wf = wrap_init(f)
    wff, out_tree_promise = flatten_fun(wf, in_tree)
    wffa = annotate(wff, in_type)
    return wffa, in_avals, keep_inputs, out_tree_promise


def new_inner_tracer(trace: DynamicJaxprTrace, aval) -> DynamicJaxprTracer:
    """Create a JAX tracer tracing an abstract value ``aval`, without specifying its source
    primitive."""
    dt = DynamicJaxprTracer(trace, aval, jax_current())
    trace.frame.tracers.append(dt)
    trace.frame.tracer_to_var[id(dt)] = trace.frame.newvar(aval)
    return dt


def get_implicit_and_explicit_flat_args(abstracted_axes, *args, **kwargs):
    """Get implicit arguments from explicit arguments and abstracted_axes."""
    axes_specs = _flat_axes_specs(abstracted_axes, *args, **kwargs)
    explicit_args, _ = tree_flatten(args)
    in_type = infer_lambda_input_type(axes_specs, explicit_args)
    implicit_args = _extract_implicit_args(in_type, explicit_args)
    args_flat = [*implicit_args, *explicit_args]
    return args_flat


def make_jaxpr2(
    fun: Callable,
    static_argnums: Any | None = None,
    abstracted_axes: Any | None = None,
) -> Callable[..., (tuple[ClosedJaxpr, PyTreeDef])]:
    """A customized version of ``jax.make_jaxpr``, compatible with the JAX dynamic API."""

    def abstractify(args, kwargs):
        flat_args, in_tree = tree_flatten((args, kwargs))
        axes_specs = _flat_axes_specs(abstracted_axes, *args, **kwargs)
        in_type = infer_lambda_input_type(axes_specs, flat_args)
        return in_type, in_tree

    # TODO: See the `_gather_shape_rule_dynamic` comment. Remove once the upstream change is
    # applied.
    gather2_p = standard_primitive(
        _gather_shape_rule_dynamic,
        _gather_dtype_rule,
        "gather",
        weak_type_rule=_argnum_weak_type(0),
    )
    register_lowering(gather2_p, _gather_lower)

    @wraps(fun)
    def make_jaxpr_f(*args, **kwargs):
        # TODO: re-use `deduce_avals` here.
        with Patcher(
            (jax._src.interpreters.partial_eval, "get_aval", get_aval2),
            (jax._src.lax.slicing, "gather_p", gather2_p),
        ), ExitStack():
            f = wrap_init(fun)
            if static_argnums:
                argnums = [static_argnums] if isinstance(static_argnums, int) else static_argnums
                dynamic_argnums = [i for i in range(len(args)) if i not in argnums]
                f, args = jax._src.api_util.argnums_partial(f, dynamic_argnums, args)
            in_type, in_tree = abstractify(args, kwargs)
            f, out_tree_promise = flatten_fun(f, in_tree)
            f = annotate(f, in_type)
            jaxpr, output_type, consts = trace_to_jaxpr_dynamic2(f)
        closed_jaxpr = DynshapedClosedJaxpr(jaxpr, consts, output_type)
        return closed_jaxpr, out_tree_promise()

    make_jaxpr_f.__name__ = f"make_jaxpr2({make_jaxpr2.__name__})"
    return make_jaxpr_f
