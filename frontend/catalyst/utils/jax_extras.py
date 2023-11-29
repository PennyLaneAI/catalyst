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

from contextlib import ExitStack, contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Set, Type, TypeVar
from copy import copy
from dataclasses import dataclass

import jax
from jax import ShapeDtypeStruct
from jax._src import state, util
from jax._src.core import (_update_thread_local_jit_state, DBIdx, same_referent)
from jax._src.dispatch import jaxpr_replicas
from jax._src.effects import ordered_effects as jax_ordered_effects
from jax._src.interpreters.mlir import _module_name_regex
from jax._src.interpreters.partial_eval import (
    _input_type_to_tracers,
    infer_lambda_input_type,
    trace_to_jaxpr_dynamic2,
)
from jax._src.lax.control_flow import _initial_style_jaxpr, _initial_style_open_jaxpr
from jax._src.lax.lax import _abstractify, xla
from jax._src.linear_util import annotate
from jax._src.pjit import _extract_implicit_args, _flat_axes_specs
from jax._src.sharding_impls import ReplicaAxisContext
from jax._src.source_info_util import current as jax_current
from jax._src.source_info_util import new_name_stack
from jax._src.util import partition_list, safe_map, unzip2, unzip3, wrap_name, wraps
from jax.api_util import flatten_fun
from jax.core import (
    AbstractValue, ClosedJaxpr, Jaxpr, JaxprEqn, MainTrace, OutputType,
    Primitive,
    ShapedArray,
    DShapedArray,
    InDBIdx,
    OutDBIdx,
    Trace,
    Tracer,
    concrete_aval,
    eval_jaxpr,
    gensym,
    thread_local_state,
    get_referent,
    new_jaxpr_eqn,
    find_top_trace
)
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
from jax.linear_util import wrap_init, transformation, transformation_with_aux
from jax.tree_util import (
    PyTreeDef,
    tree_flatten,
    tree_structure,
    tree_unflatten,
    treedef_is_leaf,
)
from jaxlib.xla_extension import PyTreeRegistry

from catalyst.utils.patching import Patcher

# pylint: disable=protected-access

__all__ = (
    "ClosedJaxpr",
    "DynamicJaxprTrace",
    "DynamicJaxprTracer",
    "Jaxpr",
    "PyTreeDef",
    "PyTreeRegistry",
    "ShapedArray",
    "DShapedArray",
    "ShapeDtypeStruct",
    "DynshapePrimitive",
    "convert_constvars_jaxpr",
    "convert_element_type",
    "collapse",
    "deduce_avals",
    "deduce_avals2",
    "deduce_avals3",
    "infer_output_type",
    "infer_lambda_input_type",
    "expand_args",
    "expand_results",
    "eval_jaxpr",
    "initial_style_jaxprs_with_common_consts1",
    "initial_style_jaxprs_with_common_consts2",
    "_abstractify",
    "_initial_style_jaxpr",
    "_input_type_to_tracers",
    "input_type_to_tracers",
    "_extract_implicit_args",
    "output_type_to_tracers",
    "jaxpr_to_mlir",
    "jaxpr_remove_implicit",
    "jaxpr_force_outvars",
    "make_jaxpr_effects",
    "make_jaxpr2",
    "new_dynamic_main2",
    "new_inner_tracer",
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


def sort_eqns(eqns: List[JaxprEqn], forced_order_primitives: Set[Primitive]) -> List[JaxprEqn]:
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


@transformation
def expanded_fun(in_type, *args_expanded):
    args_collapsed = [a for a,(_,k) in zip(args_expanded, in_type) if k]
    ans = yield args_collapsed, {}
    yield ans

@transformation_with_aux
def expanded_fun2(static_args, *args_expanded):
    (in_type, force_implicit_indbidx) = static_args
    args_collapsed = [a for a,(_,k) in zip(args_expanded, in_type) if k]
    res_flat = yield args_collapsed, {}
    yield expand_results(args_expanded, res_flat, force_implicit_indbidx=force_implicit_indbidx)

@dataclass
class InputSignature:
    in_type: InputType
    in_tree: PyTreeDef


@dataclass
class OutputSignature:
    out_type: Callable[[],OutputType]
    out_tree: Callable[[],PyTreeDef]


def deduce_avals3(f: Callable, args, kwargs, force_implicit_indbidx=False):
    """Wraps the callable ``f`` into a WrappedFun container accepting collapsed flatten arguments
    and returning expanded flatten results. Calculate input abstract values and output_tree promise.
    The promise must be called after the resulting wrapped function is evaluated."""
    flat_args, in_tree = tree_flatten((args, kwargs))
    axes_specs = _flat_axes_specs(None, *args, **kwargs)
    in_type = infer_lambda_input_type(axes_specs, flat_args)
    wf = wrap_init(f)
    wf, out_tree_promise = flatten_fun(wf, in_tree)
    wf, out_type_promise = expanded_fun2(wf, (in_type, force_implicit_indbidx))
    wf = annotate(wf, in_type)
    return (
        wf,
        InputSignature(in_type, in_tree),
        OutputSignature(out_type_promise, out_tree_promise)
    )


def deduce_avals2(f: Callable, args, kwargs):
    """Wraps the callable ``f`` into a WrappedFun container accepting collapsed flatten arguments
    and returning expanded flatten results. Calculate input abstract values and output_tree promise.
    The promise must be called after the resulting wrapped function is evaluated."""
    flat_args, in_tree = tree_flatten((args, kwargs))
    abstracted_axes = None
    axes_specs = _flat_axes_specs(abstracted_axes, *args, **kwargs)
    in_type = infer_lambda_input_type(axes_specs, flat_args)
    in_avals, keep_inputs = unzip2(in_type)
    wf = wrap_init(f)
    wf, out_tree_promise = flatten_fun(wf, in_tree)
    wf = expanded_fun(wf, in_type)
    wf = annotate(wf, in_type)
    return wf, in_avals, keep_inputs, out_tree_promise


def deduce_avals(f: Callable, args, kwargs):
    """Wraps the callable ``f`` into a WrappedFun container accepting collapsed flatten arguments
    and returning expanded flatten results. Calculate input abstract values and output_tree promise.
    The promise must be called after the resulting wrapped function is evaluated."""
    flat_args, in_tree = tree_flatten((args, kwargs))
    abstracted_axes = None
    axes_specs = _flat_axes_specs(abstracted_axes, *args, **kwargs)
    in_type = infer_lambda_input_type(axes_specs, flat_args)
    in_avals, keep_inputs = unzip2(in_type)
    wf = wrap_init(f)
    wff, out_tree_promise = flatten_fun(wf, in_tree)
    wffa = annotate(wff, in_type)
    return wffa, in_avals, keep_inputs, out_tree_promise


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


def jaxpr_to_mlir(func_name, jaxpr):
    """Lower a Jaxpr into an MLIR module.

    Args:
        func_name(str): function name
        jaxpr(Jaxpr): Jaxpr code to lower

    Returns:
        module: the MLIR module corresponding to ``func``
        context: the MLIR context corresponding
    """

    with Patcher(
        (jax._src.interpreters.partial_eval, "get_aval", get_aval2),
        (jax._src.core, "clean_up_dead_vars", _no_clean_up_dead_vars),
    ):
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

    return module, context


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


def new_inner_tracer2(frame, trace, aval) -> DynamicJaxprTracer:
    """Create a JAX tracer tracing an abstract value ``aval`, without specifying its source
    primitive."""
    dt = DynamicJaxprTracer(trace, aval, jax_current())
    frame.tracers.append(dt)
    frame.tracer_to_var[id(dt)] = frame.newvar(aval)
    return dt


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


def jaxpr_remove_implicit(
    closed_jaxpr: ClosedJaxpr, out_type: OutputType
) -> tuple[ClosedJaxpr, OutputType]:
    """Remove all the implicit result values of the ``closed_jaxpr``."""
    # Note: a more idiomatic way of doing this would be to re-trace the jaxpr and drop the unneeded
    # tracers.
    jaxpr = closed_jaxpr.jaxpr
    out_keep = list(tuple(zip(*out_type))[1]) if len(out_type) > 0 else []
    outvars = [o for o, keep in zip(jaxpr._outvars, out_keep) if keep]
    out_type2 = [o for o, keep in zip(out_type, out_keep) if keep]
    jaxpr2 = Jaxpr(
        jaxpr.constvars, jaxpr.invars, outvars, jaxpr.eqns, jaxpr.effects, jaxpr.debug_info
    )
    return ClosedJaxpr(jaxpr2, closed_jaxpr.consts), out_type2


def jaxpr_force_outvars(
    closed_jaxpr: ClosedJaxpr, out_type: OutputType
) -> tuple[ClosedJaxpr, OutputType]:
    """Turn all the InDBIdx references in a Jaxpr program into OutDBIdx."""
    jaxpr = closed_jaxpr.jaxpr
    out_aval, out_keep = unzip2(out_type)
    num_inrefs = max(sum(([d.val for d in a.shape if isinstance(d, InDBIdx)] for a in out_aval
                              if hasattr(a, "shape")), [0]))
    def _new_shape(shape):
        shape2 = []
        for d in shape:
            if isinstance(d, InDBIdx):
                d2 = OutDBIdx(d.val)
            elif isinstance(d, OutDBIdx):
                d2 = OutDBIdx(d.val + num_inrefs)
            else:
                d2 = d
            shape2.append(d2)
        return tuple(shape2)

    for v in out_aval:
        if hasattr(v, "shape"):
            v.update(shape=_new_shape(v.shape))
    outvars2 = jaxpr.invars[:num_inrefs] + jaxpr.outvars
    out_type2 = tuple((i.aval,False) for i in jaxpr.invars[:num_inrefs]) + tuple(zip(out_aval, out_keep))
    jaxpr2 = Jaxpr(
        jaxpr.constvars, jaxpr.invars, outvars2, jaxpr.eqns, jaxpr.effects, jaxpr.debug_info
    )
    return ClosedJaxpr(jaxpr2, closed_jaxpr.consts), out_type2


def make_jaxpr2(
    fun: Callable,
    abstracted_axes: Any | None = None,
) -> Callable[..., (tuple[ClosedJaxpr, PyTreeDef])]:
    """A customized version of ``jax.make_jaxpr``, compatible with the JAX dynamic API."""

    def abstractify(args, kwargs):
        flat_args, in_tree = tree_flatten((args, kwargs))
        axes_specs = _flat_axes_specs(abstracted_axes, *args, **kwargs)
        in_type = infer_lambda_input_type(axes_specs, flat_args)
        return in_type, in_tree

    @wraps(fun)
    def make_jaxpr_f(*args, **kwargs):
        # TODO: re-use `deduce_avals` here.
        with Patcher((jax._src.interpreters.partial_eval, "get_aval", get_aval2)), ExitStack():
            f = wrap_init(fun)
            in_type, in_tree = abstractify(args, kwargs)
            f, out_tree_promise = flatten_fun(f, in_tree)
            f = annotate(f, in_type)
            jaxpr, out_type, consts = trace_to_jaxpr_dynamic2(f)
        closed_jaxpr = ClosedJaxpr(jaxpr, consts)
        return closed_jaxpr, out_type, out_tree_promise()

    make_jaxpr_f.__name__ = f"make_jaxpr2({make_jaxpr2.__name__})"
    return make_jaxpr_f

def input_type_to_tracers(in_type: InputType,
                          maker: Callable[[AbstractValue],DynamicJaxprTracer]
                          ) -> List[DynamicJaxprTracer]:
    """ Creates an expanded list of tracers representing an input values of a Jaxpr program """
    in_aval, in_keep = unzip2(in_type)
    return _input_type_to_tracers(maker, in_aval)


def get_referent_frame(self, frame):
    """ Find referents in the specific frame """
    # TODO: clarify the logic! Can we need other frames?
    val = frame.constvar_to_val.get(frame.tracer_to_var.get(id(self)))
    return self if val is None else get_referent_frame(val)


def output_type_to_tracers(out_type: OutputType,
                           in_tracers: List[DynamicJaxprTracer],
                           maker: Callable[[AbstractValue],DynamicJaxprTracer]
                           ) -> List[DynamicJaxprTracer]:
    """ Creates an expanded list of tracers representing an output values of a Jaxpr program
    based on the ``out_type`` of this program. The resulting tracers might be nested as required by
    the Jax dynamic API and might contain ``in_tracers`` of the same Jaxpr program. """
    out_tracers = []
    for aval, _ in out_type:
        if type(aval) is DShapedArray:
            shape = [[*in_tracers][d.val] if type(d) is InDBIdx else
                     out_tracers[d.val] if type(d) is OutDBIdx else
                     d for d in aval.shape]
            aval = aval.update(shape=tuple(get_referent(d) for d in shape))
        out_tracers.append(maker(aval))
    return out_tracers


TracerLike = TypeVar("TracerLike")

def infer_output_type2(inputs:List[TracerLike],
                      outputs:List[TracerLike],
                      # eq_fun:Callable[[TracerLike,TracerLike],bool],
                      force_implicit_indbidx:bool = True,
                      ) -> OutputType:
    """ Deduce the Jax ``out_type`` given input and ouputs abstract entities. By abstract entities
    we mean either Jax tracers or Jaxpr variables. """

    def _is_tracer_like(x):
        return hasattr(x, "aval")

    expl_outs = outputs
    impl_outs = []
    seen = set() if force_implicit_indbidx else set(inputs)

    for o in expl_outs:
        assert _is_tracer_like(o)
        if isinstance(o.aval, DShapedArray):
            for d in o.aval.shape:
                if _is_tracer_like(d) and (d not in seen):
                    impl_outs.append(d)
                    seen.add(d)
        seen.add(o)

    all_ins = [*inputs]
    all_outs = [*impl_outs, *expl_outs]
    in_map : dict[TracerLike,  InDBIdx] = {v:  InDBIdx(i) for i, v in enumerate(all_ins)}
    out_map: dict[TracerLike, OutDBIdx] = {x: OutDBIdx(i) for i, x in enumerate(all_outs)}

    out_avals_ = (x.aval for x in all_outs)
    out_avals = [a.update(shape=tuple(in_map.get(d, out_map.get(d))
                                      if _is_tracer_like(d) else d for d in a.shape))
                 if isinstance(a, DShapedArray) else a for a in out_avals_]

    kept_outs = [False] * len(impl_outs) + [True] * len(expl_outs)
    out_type = tuple(zip(out_avals, kept_outs))
    return all_outs, out_type


def infer_output_type(inputs:List[TracerLike],
                      outputs:List[TracerLike],
                      force_implicit_indbidx:bool = True,
                      ) -> OutputType:

    return infer_output_type2(inputs, outputs, force_implicit_indbidx=force_implicit_indbidx)[1]


def expand_args(args:List[TracerLike], in_type=None) -> List[TracerLike]:
    in_type = in_type if in_type is not None else infer_lambda_input_type(None, args)
    return list(_extract_implicit_args(in_type, args)) + list(args)


def expand_results(
    args:List[TracerLike],
    results:List[TracerLike],
    force_implicit_indbidx:bool=False,
) -> Tuple[List[TracerLike], OutputType]:
    return infer_output_type2(args, results, force_implicit_indbidx=force_implicit_indbidx)


def collapse(typ:InputType|OutputType, params:List[TracerLike]) -> List[TracerLike]:
    return [t for t,(_,k) in zip(params, typ) if k]


class DynshapePrimitive(Primitive):
    # TODO: Contribute this class to Jax, namely to the
    # `jax.DynamicJaxprTrace.default_process_primitive`

    def bind(primitive, *args, **params):
        trace = find_top_trace(args)
        tracers = map(trace.full_raise, args)
        # avals = [get_aval2(t) for t in tracers]
        source_info = jax_current()

        in_type = infer_lambda_input_type(None, tracers)
        # print("BIND_IN_TYPE")
        # for t in in_type: print("I", t)

        out_type, effects = primitive.abstract_eval(*in_type, **params)
        # print("BIND_OUT_TYPE")
        # for t in out_type: print("O", t)
        assert len(effects) == 0, f"Effects are not supported, got ({effects})"

        out_tracers = output_type_to_tracers(
            out_type, tracers,
            maker=lambda a: DynamicJaxprTracer(trace, a, source_info))

        invars = map(trace.getvar, tracers)
        outvars = map(trace.makevar, out_tracers)

        eqn = new_jaxpr_eqn(invars, outvars, primitive, params, [], source_info)
        trace.frame.add_eqn(eqn)
        # out_tracers = [t for t,(_,k) in zip(out_tracers, out_type) if k]
        return out_tracers if primitive.multiple_results else out_tracers.pop()

