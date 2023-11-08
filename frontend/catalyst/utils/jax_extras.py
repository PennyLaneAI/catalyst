# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module isolates utility functions that depend on JAX low-level internals
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Set, Type

from jax import ShapeDtypeStruct
from jax._src import state, util
from jax._src.core import _update_thread_local_jit_state, eval_jaxpr
from jax._src.dispatch import jaxpr_replicas
from jax._src.effects import ordered_effects as jax_ordered_effects
from jax._src.interpreters.mlir import _module_name_regex
from jax._src.interpreters.partial_eval import _input_type_to_tracers
from jax._src.lax.control_flow import _initial_style_jaxpr, _initial_style_open_jaxpr
from jax._src.lax.lax import _abstractify, xla
from jax._src.linear_util import annotate
from jax._src.sharding_impls import ReplicaAxisContext
from jax._src.source_info_util import current as jax_current
from jax._src.source_info_util import new_name_stack
from jax._src.util import partition_list, safe_map, unzip2, unzip3, wrap_name
from jax.api_util import flatten_fun, shaped_abstractify
from jax.core import ClosedJaxpr, Jaxpr, JaxprEqn, MainTrace
from jax.core import Primitive as JaxprPrimitive
from jax.core import ShapedArray, Trace, gensym, thread_local_state
from jax.interpreters.mlir import (
    AxisContext,
    ModuleContext,
    ir,
    lower_jaxpr_to_fun,
    lowerable_effects,
)
from jax.interpreters.partial_eval import (
    DynamicJaxprTrace,
    DynamicJaxprTracer,
    convert_constvars_jaxpr,
    make_jaxpr_effects,
)
from jax.lax import convert_element_type
from jax.linear_util import wrap_init
from jax.tree_util import (
    PyTreeDef,
    tree_flatten,
    tree_structure,
    tree_unflatten,
    treedef_is_leaf,
)
from jaxlib.xla_extension import pytree

__all__ = (
    "ClosedJaxpr",
    "DynamicJaxprTrace",
    "DynamicJaxprTracer",
    "Jaxpr",
    "PyTreeDef",
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
    "jaxpr_to_mlir",
    "make_jaxpr_effects",
    "new_dynamic_main2",
    "new_inner_tracer",
    "pytree",
    "sort_eqns",
    "treedef_is_leaf",
    "tree_flatten",
    "tree_structure",
    "tree_unflatten",
    "unzip2",
    "wrap_init",
)

map, unsafe_map = safe_map, map  # pylint: disable=redefined-builtin


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
    """Wraps the callable ``f`` into a WrappedFun JAX container. Calculate input abstract values
    and output_tree promise. The promise must be called after the resulting wrapped function is
    evaluated."""
    flat_args, in_tree = tree_flatten((args, kwargs))
    wf = wrap_init(f)
    in_avals, keep_inputs = list(map(shaped_abstractify, flat_args)), [True] * len(flat_args)
    in_type = tuple(zip(in_avals, keep_inputs))
    wff, out_tree_promise = flatten_fun(wf, in_tree)
    wffa = annotate(wff, in_type)
    return wffa, in_avals, out_tree_promise


def jaxpr_to_mlir(func_name, jaxpr, shape):
    """Lower a Python function into an MLIR module.

    Args:
        func: python function to be lowered
        args: arguments to ``func``
        kwargs: keyword arguments to ``func``

    Returns:
        module: the MLIR module corresponding to ``func``
        context: the MLIR context corresponding
        jaxpr: the jaxpr corresponding to ``func``
        shape: the shape of the return values in ``PyTreeDef``
    """

    nrep = jaxpr_replicas(jaxpr)
    effects = jax_ordered_effects.filter_in(jaxpr.effects)
    axis_context = ReplicaAxisContext(xla.AxisEnv(nrep, (), ()))
    name_stack = new_name_stack(wrap_name("ok", "jit"))
    module, context = custom_lower_jaxpr_to_module(
        func_name="jit_" + func_name,
        module_name=func_name,
        jaxpr=jaxpr,
        effects=effects,
        platform="cpu",
        axis_context=axis_context,
        name_stack=name_stack,
    )

    return module, context, jaxpr, tree_structure(shape)


# pylint: disable=too-many-arguments
def custom_lower_jaxpr_to_module(
    func_name: str,
    module_name: str,
    jaxpr: ClosedJaxpr,
    effects,
    platform: str,
    axis_context: AxisContext,
    name_stack,
    replicated_args=None,
    arg_shardings=None,
    result_shardings=None,
):
    """Lowers a top-level jaxpr to an MHLO module.

    Handles the quirks of the argument/return value passing conventions of the
    runtime.

    This function has been modified from its original form in the JAX project at
    https://github.com/google/jax/blob/c4d590b1b640cc9fcfdbe91bf3fe34c47bcde917/jax/interpreters/mlir.py#L625version
    released under the Apache License, Version 2.0, with the following copyright notice:

    Copyright 2021 The JAX Authors.
    """

    if any(lowerable_effects.filter_not_in(jaxpr.effects)):  # pragma: no cover
        raise ValueError(f"Cannot lower jaxpr with effects: {jaxpr.effects}")

    assert platform == "cpu"
    assert arg_shardings is None
    assert result_shardings is None

    # MHLO channels need to start at 1
    channel_iter = 1
    # Create a keepalives list that will be mutated during the lowering.
    keepalives = []
    host_callbacks = []
    ctx = ModuleContext(
        None, platform, axis_context, name_stack, keepalives, channel_iter, host_callbacks
    )
    ctx.context.allow_unregistered_dialects = True
    with ctx.context, ir.Location.unknown(ctx.context):
        # register_dialect()
        # Remove module name characters that XLA would alter. This ensures that
        # XLA computation preserves the module name.
        module_name = _module_name_regex.sub("_", module_name)
        ctx.module.operation.attributes["sym_name"] = ir.StringAttr.get(module_name)
        lower_jaxpr_to_fun(
            ctx,
            func_name,
            jaxpr,
            effects,
            public=True,
            create_tokens=True,
            replace_tokens_with_dummy=True,
            replicated_args=replicated_args,
            arg_shardings=arg_shardings,
            result_shardings=result_shardings,
        )

        for op in ctx.module.body.operations:
            func_name = str(op.name)
            is_entry_point = func_name.startswith('"jit_')
            if is_entry_point:
                continue
            op.attributes["llvm.linkage"] = ir.Attribute.parse("#llvm.linkage<internal>")

    return ctx.module, ctx.context


def new_inner_tracer(trace: DynamicJaxprTrace, aval) -> DynamicJaxprTracer:
    """Create a JAX tracer tracing an abstract value ``aval`, without specifying its source
    primitive."""
    dt = DynamicJaxprTracer(trace, aval, jax_current())
    trace.frame.tracers.append(dt)
    trace.frame.tracer_to_var[id(dt)] = trace.frame.newvar(aval)
    return dt
