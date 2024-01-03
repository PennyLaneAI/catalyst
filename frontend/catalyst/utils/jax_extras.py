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
from copy import copy
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
)

import jax
from jax import ShapeDtypeStruct
from jax._src.core import DBIdx, _update_thread_local_jit_state
from jax._src.dispatch import jaxpr_replicas
from jax._src.effects import ordered_effects as jax_ordered_effects
from jax._src.interpreters.mlir import _module_name_regex
from jax._src.interpreters.partial_eval import (
    AbstractedAxesSpec,
    _input_type_to_tracers,
    infer_lambda_input_type,
    trace_to_jaxpr_dynamic2,
)
from jax._src.lax.control_flow import _initial_style_jaxpr
from jax._src.lax.lax import _abstractify, xla
from jax._src.linear_util import annotate
from jax._src.pjit import _extract_implicit_args, _flat_axes_specs
from jax._src.sharding_impls import ReplicaAxisContext
from jax._src.source_info_util import current as jax_current
from jax._src.source_info_util import new_name_stack
from jax._src.util import safe_map, unzip2, wrap_name, wraps
from jax.api_util import flatten_fun
from jax.core import (
    AbstractValue,
    ClosedJaxpr,
    ConcreteArray,
    DShapedArray,
    InDBIdx,
    InputType,
    Jaxpr,
    JaxprEqn,
    MainTrace,
    OutDBIdx,
    OutputType,
    Primitive,
    ShapedArray,
    Trace,
    Tracer,
    concrete_aval,
    eval_jaxpr,
    find_top_trace,
    gensym,
    new_jaxpr_eqn,
    thread_local_state,
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
from jax.linear_util import transformation_with_aux, wrap_init
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
    "ExpansionStrategy",
    "for_loop_expansion_strategy",
    "cond_expansion_strategy",
    "while_loop_expansion_strategy",
    "Jaxpr",
    "PyTreeDef",
    "PyTreeRegistry",
    "ShapedArray",
    "DShapedArray",
    "ConcreteArray",
    "ShapeDtypeStruct",
    "DynshapePrimitive",
    "convert_constvars_jaxpr",
    "convert_element_type",
    "collapse",
    "deduce_avals",
    "deduce_signatures",
    "infer_output_type_python",
    "infer_output_type_jaxpr",
    "infer_lambda_input_type",
    "expand_args",
    "expand_results",
    "eval_jaxpr",
    "_abstractify",
    "_initial_style_jaxpr",
    "_input_type_to_tracers",
    "input_type_to_tracers",
    "_extract_implicit_args",
    "output_type_to_tracers",
    "jaxpr_to_mlir",
    "jaxpr_remove_implicit",
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


def jaxpr_pad_consts(jaxprs: List[Jaxpr]) -> List[ClosedJaxpr]:
    """Align the constants of Jaxpr programs. Return the list of corresponding programs accepting
    same constants."""
    newvar = gensym(jaxprs, suffix="_")

    # pylint: disable=too-many-nested-blocks
    all_padded_constvars = []
    for jaxpr in jaxprs:
        padded_constvars = []
        for jaxpr2 in jaxprs:
            if jaxpr2 is jaxpr:
                padded_constvars.extend(jaxpr2.constvars)
            else:
                cmap = {}
                for cv in jaxpr2.constvars:
                    aval = cv.aval
                    if isinstance(aval, DShapedArray):
                        shape2 = []
                        for d in aval.shape:
                            if hasattr(d, "aval"):
                                shape2.append(cmap[d])
                            else:
                                shape2.append(d)
                        aval = aval.update(shape=tuple(shape2))
                    nv = newvar(aval)
                    cmap[cv] = nv
                    padded_constvars.append(nv)
        all_padded_constvars.append(padded_constvars)

    acc = []
    for jaxpr, padded_constvars in zip(jaxprs, all_padded_constvars):
        acc.append(
            ClosedJaxpr(convert_constvars_jaxpr(jaxpr.replace(constvars=padded_constvars)), ())
        )
    return acc


@transformation_with_aux
def expanded_fun(static_args, *args_expanded):
    """Function transformation making the function to accept its arguments in the expanded format
    (with the dimension variables added)"""
    (in_type, expansion_strategy) = static_args
    args_collapsed = [a for a, (_, k) in zip(args_expanded, in_type) if k]
    res_flat = yield args_collapsed, {}
    num_implicit_inputs = len([() for _, k in in_type if not k])
    all_outs, out_sig = infer_output_type_python(
        args_expanded, res_flat, expansion_strategy, num_implicit_inputs
    )
    yield all_outs, out_sig


@dataclass
class InputSignature:
    """Meta-parameters of a function which are available before the tracing to the function
    starts."""

    in_type: InputType
    in_tree: PyTreeDef
    in_expanded_args: List[DynamicJaxprTracer]

    def num_implicit_inputs(self) -> int:
        """Return the number of implicit input arguments of the function"""
        return len([() for _, k in self.in_type if not k])


@dataclass
class OutputSignature:
    """Meta-parameters of a function which become available after the tracing to the function is
    complete."""

    out_jaxpr: Callable[[], ClosedJaxpr]
    out_type: Callable[[], OutputType]
    out_consts: Callable[[], list]
    out_tree: Callable[[], PyTreeDef]
    out_initial_jaxpr: Callable[[], Jaxpr]

    def num_implicit_outputs(self) -> int:
        """Return the number of implicit resuts of the function"""
        return len([() for _, k in self.out_type() if not k])


def deduce_signatures(
    f: Callable, args, kwargs, expansion_strategy
) -> Tuple[Callable, InputSignature, OutputSignature]:
    """Prepares the callable ``f`` for tracing by wrapping it into a WrappedFun container accepting
    expanded flattened arguments and returning expanded flatten results. Jax input and output types
    are returned along with the other related information aggregated into input and output signature
    datatypes.

    Args:
        f (Callable): Python function to trace. By definition, the function accepts collapsed
                      unflattened arguments and returns collapsed unflattened results.
        args (Any): Positional arguments to be passed to ``f``, typically Jax tracers.
        kwargs (dict): Keyword arguments to be passed to ``f``.
        expansion_strategy (ExpansionStrategy): Argument and result expansion settings to use when
                                                expanding arguments and results.

    Returns:
        Callable: Python function, accepting flattened expanded arguments and returning flattened
                  expanded results
        InputSignature: Input signature of the function
        OutputSignature: Output signature of the function. Fields of the output signature are
                         available after exiting from the Callable.
    """
    flat_args, in_tree = tree_flatten((args, kwargs))
    trace: DynamicJaxprTrace = find_top_trace(flat_args)
    flat_tracers = [trace.full_raise(a) for a in flat_args]
    # flat_axes_specs = _flat_axes_specs(abstracted_axes, *args, **kwargs)
    in_expanded_args, in_type = expand_args(
        flat_tracers, expansion_strategy=expansion_strategy  # axes_specs=flat_axes_specs,
    )
    wf = wrap_init(f)
    wf, out_tree_promise = flatten_fun(wf, in_tree)
    wf, out_sig_promise = expanded_fun(wf, (in_type, expansion_strategy))
    wf = annotate(wf, in_type)
    return (
        wf,
        InputSignature(in_type, in_tree, in_expanded_args),
        OutputSignature(
            lambda: ClosedJaxpr(convert_constvars_jaxpr(out_sig_promise()[0]), ()),
            lambda: out_sig_promise()[1],
            lambda: out_sig_promise()[2],
            out_tree_promise,
            lambda: out_sig_promise()[0],
        ),
    )


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


def input_type_to_tracers(
    in_type: InputType, maker: Callable[[AbstractValue], DynamicJaxprTracer]
) -> List[DynamicJaxprTracer]:
    """Creates an expanded list of tracers representing an input values of a Jaxpr program"""
    in_aval, _ = unzip2(in_type)
    return _input_type_to_tracers(maker, in_aval)


# def get_referent_frame(frame):
#     """Find referents in the specific frame"""
#     # TODO: clarify the logic! Can we need other frames?
#     val = frame.constvar_to_val.get(frame.tracer_to_var.get(id(self)))
#     return self if val is None else get_referent_frame(val)


def output_type_to_tracers(
    out_type: OutputType,
    out_consts: List[DynamicJaxprTracer],
    in_tracers: List[DynamicJaxprTracer],
    maker: Callable[[AbstractValue], DynamicJaxprTracer],
) -> List[DynamicJaxprTracer]:
    """Creates an expanded list of tracers representing an output values of a Jaxpr program
    based on the ``out_type`` of this program. The resulting tracers might be nested as required by
    the Jax dynamic API and might contain ``in_tracers`` of the same Jaxpr program."""
    out_tracers = []
    for aval, _ in out_type:
        if isinstance(aval, DShapedArray):
            shape = [
                [*out_consts, *in_tracers][d.val]
                if isinstance(d, InDBIdx)
                else out_tracers[d.val]
                if isinstance(d, OutDBIdx)
                else d
                for d in aval.shape
            ]
            aval = aval.update(shape=tuple(shape))
        out_tracers.append(maker(aval))
    return out_tracers


TracerLike = TypeVar("TracerLike")


def infer_input_type_unshared(inputs: List[TracerLike]) -> InputType:
    """Infer the input type of a function having `inputs` Jax tracers."""

    def _is_tracer_like(x):
        return hasattr(x, "aval")

    expl_ins = inputs
    impl_avals = []
    expl_avals = []
    for o in expl_ins:
        assert _is_tracer_like(o), (o,)
        if isinstance(o.aval, DShapedArray):
            shape2 = []
            for d in o.aval.shape:
                if _is_tracer_like(d):
                    shape2.append(DBIdx(len(impl_avals)))
                    impl_avals.append(d.aval)
                else:
                    shape2.append(d)
            expl_avals.append(o.aval.update(shape=tuple(shape2)))
        else:
            expl_avals.append(o.aval)
    return (*[(i, False) for i in impl_avals], *[(i, True) for i in expl_avals])


@dataclass
class ExpansionStrategy:
    """Describe the settings affecting the Jax dyanmic API tracing of the nested programs.
    Args:
        axes_specs: Axes specification used to convert static dimensions to the dynamic ones
        input_unshare_variables: Treat each dynamic dimension as a distinct dimension, even if some
                                 of them are described by the same dimension variables.
        output_include_indbidx_vars: Include variables corresponding to the InDBIdx references in
                                     the result
        output_force_arg0_outdbidx: Force all references pointing to the first arguments to be
                                    OutDBIdx. This is typically required to avoid input references
                                    to the loop iteration variable.
    """

    axes_specs: Sequence[AbstractedAxesSpec] | None
    input_unshare_variables: bool
    output_include_indbidx_vars: bool
    output_force_arg0_outdbidx: bool


def default_expansion_strategy(axes_specs):
    return ExpansionStrategy(axes_specs, False, False, False)


def while_loop_expansion_strategy(preserve_dimensions=False):
    """Arguments and results expansion strategy for while-loops."""
    return ExpansionStrategy(None, not preserve_dimensions, True, False)


def for_loop_expansion_strategy(preserve_dimensions=False):
    """Arguments and results expansion strategy for for-loops."""
    return ExpansionStrategy(None, not preserve_dimensions, True, True)


def cond_expansion_strategy():
    """Arguments and results expansion strategy for conditionals."""
    return ExpansionStrategy(None, False, True, False)


def infer_output_type(
    constants: List[TracerLike],
    expanded_inputs: List[TracerLike],
    outputs: List[TracerLike],
    expansion_strategy: ExpansionStrategy,
    num_implicit_inputs: int | None = None,
) -> Tuple[List[TracerLike], OutputType]:
    """Deduce the Jax ``OutputType`` of a part of program (typically, a function) given its
    constants, input and ouput tracers or variables. Return the expanded outputs along with the
    output type calculated.

    The core task of this function is to find out which tracers have dynamic dimensions and
    translate this information into the language of the De Brujin indices residing in Jax types. In
    order to do this, we scan the outputs and mind what dimensions are already known (from the
    intputs) and what are not known. The known dimensions are marked with InDBIdx and the unknown
    dimensions are treated as calculated and marked using OutDBIdx.


    Args:
        constants: Constants which are required to evaluate the program.
        expanded_inputs: Input tracers of the program. Implicit dimension tracers must be included.
        outputs: Explicit output tracers of the program, as return by Python function.
        expansion_strategy: Output expansion options.
        num_implicit_inputs: The number of implicit inputs residing in the `expanded_inputs`
                             argument.

    Returns:
        List[TracerLike]: Expanded outputs, with the dynamic dimension variabiles set correctly
        OutputType: Jax ``OutputType`` of the program, including all the required *DBIdx.
    """
    s = expansion_strategy

    def _is_tracer_like(x):
        return hasattr(x, "aval")

    expl_outs = outputs
    impl_outs = []
    seen = set() if s.output_include_indbidx_vars else set(map(id, [*constants, *expanded_inputs]))

    for o in expl_outs:
        assert _is_tracer_like(o)
        if isinstance(o.aval, DShapedArray):
            for d in o.aval.shape:
                if _is_tracer_like(d) and (id(d) not in seen):
                    impl_outs.append(d)
                    if not s.input_unshare_variables:
                        seen.add(id(d))
        if not s.input_unshare_variables:
            seen.add(id(o))

    all_ins = [*constants, *expanded_inputs]
    all_outs = [*impl_outs, *expl_outs]
    in_map: dict[TracerLike, InDBIdx] = {id(v): InDBIdx(i) for i, v in enumerate(all_ins)}
    out_map: dict[TracerLike, OutDBIdx] = {id(x): OutDBIdx(i) for i, x in enumerate(all_outs)}

    out_avals_ = (x.aval for x in all_outs)
    out_avals = [
        a.update(
            shape=tuple(
                in_map.get(id(d), out_map.get(id(d))) if _is_tracer_like(d) else d for d in a.shape
            )
        )
        if isinstance(a, DShapedArray)
        else a
        for a in out_avals_
    ]

    kept_outs = [False] * len(impl_outs) + [True] * len(expl_outs)
    out_type = tuple(zip(out_avals, kept_outs))

    if s.output_force_arg0_outdbidx:
        assert s.output_include_indbidx_vars
        assert num_implicit_inputs is not None
        out_type = out_type_force_outdbidx(
            out_type,
            len(constants) + num_implicit_inputs,  # for-loop index argument
            constants,
            expanded_inputs,
            all_outs,
        )
    return all_outs, out_type


def infer_output_type_jaxpr(
    constants: List[TracerLike],
    expanded_inputs: List[TracerLike],
    outputs: List[TracerLike],
    expansion_strategy,
    num_implicit_inputs: int | None = None,
) -> OutputType:
    """Infers the Jax ``OutputType`` based on the Jaxpr program's result variables, expanded inputs
    and constants.

    See the ``infer_output_type`` for the additional explanation.
    """
    _, out_type = infer_output_type(
        constants, expanded_inputs, outputs, expansion_strategy, num_implicit_inputs
    )
    return out_type


def infer_output_type_python(
    expanded_inputs: List[TracerLike],
    outputs: List[TracerLike],
    expansion_strategy: ExpansionStrategy,
    num_implicit_inputs: int,
) -> Tuple[List[TracerLike], Tuple[Jaxpr, OutputType, List[TracerLike]]]:
    """Infers the Jax ``OutputType`` of a traced Python program. In addition to the
    output type, alaso return the corresponding Jaxpr program, expanded list of outputs, and the
    constants.

    In this function we attempt to overcome Jax incompatibilities regarding the dynamic API support
    in loops.  Namely, we (1) handle additional expansion options encoded as ``expansion_strategy``,
    (2) Make sure that Jaxpr constants are counted correctly.

    See the ``infer_output_type`` for the additional explanation.
    """

    trace: DynamicJaxprTrace = find_top_trace(expanded_inputs)
    outputs = [trace.full_raise(t) for t in outputs]

    # Infer output type assuming the empty list of constants
    expanded_outputs, out_type1 = infer_output_type(
        [], expanded_inputs, outputs, expansion_strategy, num_implicit_inputs
    )

    # Calculate constants using the expanded outputs
    jaxpr, _, consts = trace.frame.to_jaxpr2(expanded_outputs)

    # Calculate output type containing the correct De Brjuin indices
    expanded_outputs2, out_type2 = infer_output_type(
        [trace.full_raise(t) for t in consts],
        expanded_inputs,
        outputs,
        expansion_strategy,
        num_implicit_inputs,
    )

    # Combine the explicitness information with the correct De Brjuin indices
    assert len(out_type1) == len(out_type2), f"\n{out_type1=}\n{out_type2=}"
    _, out_keep1 = unzip2(out_type1)
    out_aval2, _ = unzip2(out_type2)
    out_type3 = tuple(zip(out_aval2, out_keep1))

    # Return the final results
    return expanded_outputs2, (jaxpr, out_type3, consts)


def expand_args(
    args: List[TracerLike],
    expansion_strategy: ExpansionStrategy,
) -> Tuple[List[TracerLike], InputType]:
    """Calculate the expanded list of arguments of a Python program, based on the list of its
    explicit arguments.

    Args:
        args: List of explicit arguments of the Python program
        expansion_strategy: Argument expansion options

    Returns:
        List of arguments containing both explicit and implicit arguments
        OutputType describing the expanded result
    """
    s = expansion_strategy
    if s.input_unshare_variables is True:
        assert s.axes_specs is None
        in_type = infer_input_type_unshared(args)
    else:
        in_type = infer_lambda_input_type(s.axes_specs, args)
    return list(_extract_implicit_args(in_type, args)) + list(args), in_type


def expand_results(
    constants: List[TracerLike],
    expanded_inputs: List[TracerLike],
    results: List[TracerLike],
    expansion_strategy: ExpansionStrategy,
    num_implicit_inputs: int | None = None,
) -> Tuple[List[TracerLike], OutputType]:
    """Calculate the expanded list of results of a Python function, based on its input and output
    tracers/variables.

    Args:
        constants: List of constants of the program
        expanded_inputs: Expanded list of arguments of the program
        results: List of explicit results
        expansion_strategy: Expansion options
        num_implicit_inputs: Number of implicit inputs found in the `expanded_inputs` parameter

    Returns:
        List of arguments containing both explicit and implicit arguments
        OutputType describing the expanded result

    """
    return infer_output_type(
        constants, expanded_inputs, results, expansion_strategy, num_implicit_inputs
    )


def collapse(typ: InputType | OutputType, params: List[TracerLike]) -> List[TracerLike]:
    """Collapse the expanded list of parameters by cutting the implicit dimension variables, as
    specified by the type of the parameters."""
    return [t for t, (_, k) in zip(params, typ) if k]


def tracer_index(x: TracerLike, ls: List[TracerLike]) -> Optional[int]:
    """Return the index of Jax tracer `x` in the list of tracers `ls`, or None."""
    for i, t in enumerate(ls):
        if x is t:
            return i
    return None


def out_type_force_outdbidx(
    out_type: OutputType,
    input_idx: int,
    consts: List[TracerLike],
    inputs: List[TracerLike],
    outputs: List[TracerLike],
) -> OutputType:
    """Convert all references to the specified input argument of a Python or Jaxpr program into the
    OutDBIdx format. This function is typically needed to remove references to the loop body
    iterator.

    Args:
        out_type: Jax OutputType of results of the program
        input_idx: Index of the input argument that should be converted from InDBIdx to OutDBIdx
        consts: Constants used in the program
        inputs: Expanded list of inputs of the program
        outputs: Expanded list of outputs of the program

    Returns:
        New OutputType containing no InDBIdx references to the specified argument.
    """
    assert len(out_type) == len(outputs), "Outputs must be expanded"
    x_in_idx = input_idx
    x_out_idx = tracer_index([*consts, *inputs][x_in_idx], outputs)

    out_type2 = []
    for i, ((aval, k), t) in enumerate(zip(out_type, outputs)):
        assert hasattr(t, "aval"), "Outputs are expected to be Jax tracers or Jaxpr variables"
        if isinstance(aval, DShapedArray):
            shape2 = []
            for d in aval.shape:
                if isinstance(d, InDBIdx) and d.val == x_in_idx:
                    assert x_out_idx is not None, (
                        "Target tracer does not exist in the outputs "
                        "(see force_implicit_indbidx=True)"
                    )
                    assert x_out_idx < i, "Target tracer is not available for OutDBIdx"
                    shape2.append(OutDBIdx(x_out_idx))
                else:
                    shape2.append(d)
            aval2 = aval.update(shape=tuple(shape2))
        else:
            aval2 = copy(aval)
        out_type2.append((aval2, k))
    return out_type2


class DynshapePrimitive(Primitive):
    """Primitive containing nested Jaxpr programs accepting and returning Jax values with shapes
    containing dynamic dimensions."""
    pass


class DynamicJaxprTraceEx(DynamicJaxprTrace):

    def __init__(self, *args, dynamic=True, **kwargs):
        super().__init__(*args, **kwargs)

    def default_process_primitive(self, primitive, tracers, params):

        if not isinstance(primitive, DynshapePrimitive):
            return super().default_process_primitive(primitive, tracers, params)

        trace = self
        tracers = map(trace.full_raise, tracers)
        source_info = jax_current()

        in_type = infer_lambda_input_type(None, tracers)
        out_type, effects = primitive.abstract_eval(*in_type, **params)

        assert len(effects) == 0, f"Jax effects are not supported, got ({effects})"

        out_tracers = output_type_to_tracers(
            out_type,
            # FIXME: we have no information about the constants at this point so we expect that the
            # `abstract_eval` returned `out_type` calculated for empty constants.
            [],
            tracers,
            maker=lambda a: DynamicJaxprTracer(trace, a, source_info),
        )

        invars = map(trace.getvar, tracers)
        outvars = map(trace.makevar, out_tracers)

        eqn = new_jaxpr_eqn(invars, outvars, primitive, params, [], source_info)
        trace.frame.add_eqn(eqn)
        return out_tracers if primitive.multiple_results else out_tracers.pop()

