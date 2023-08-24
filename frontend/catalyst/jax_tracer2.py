from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from itertools import chain
from typing import Any, Callable, ContextManager, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import pennylane as qml
from jax._src import linear_util as lu
from jax._src.api import ShapeDtypeStruct
from jax._src.api_util import (
    _ensure_index,
    _ensure_index_tuple,
    _ensure_str_tuple,
    apply_flat_fun,
    apply_flat_fun_nokwargs,
    argnums_partial,
    argnums_partial_except,
    check_callable,
    debug_info,
    debug_info_final,
    donation_vector,
    flat_out_axes,
    flatten_axes,
    flatten_fun,
    flatten_fun_nokwargs,
    flatten_fun_nokwargs2,
    rebase_donate_argnums,
    result_paths,
    shaped_abstractify,
)
from jax._src.core import ClosedJaxpr, JaxprEqn
from jax._src.core import MainTrace as JaxMainTrace
from jax._src.core import ShapedArray
from jax._src.core import Tracer as JaxprTracer
from jax._src.core import (
    Var,
    check_jaxpr,
    cur_sublevel,
    get_aval,
    new_base_main,
    new_main,
)
from jax._src.dispatch import jaxpr_replicas
from jax._src.interpreters.mlir import _constant_handlers
from jax._src.interpreters.partial_eval import (
    DynamicJaxprTrace,
    DynamicJaxprTracer,
    Jaxpr,
    JaxprStackFrame,
    _add_implicit_outputs,
    _const_folding_and_forwarding,
    _inline_literals,
    _input_type_to_tracers,
    convert_constvars_jaxpr,
    extend_jaxpr_stack,
    make_jaxpr_effects,
    new_jaxpr_eqn,
    trace_to_subjaxpr_dynamic2,
)
from jax._src.lax.control_flow import _initial_style_jaxpr
from jax._src.lax.lax import _abstractify, xb, xla
from jax._src.source_info_util import current as jax_current
from jax._src.source_info_util import new_name_stack, reset_name_stack
from jax._src.tree_util import (
    PyTreeDef,
    tree_flatten,
    tree_structure,
    tree_unflatten,
    treedef_is_leaf,
)
from jax._src.util import unzip2
from jax.interpreters import mlir
from jax.interpreters.mlir import (
    AxisContext,
    ModuleContext,
    ReplicaAxisContext,
    ir,
    lower_jaxpr_to_fun,
    lowerable_effects,
)
from pennylane import Device, QubitDevice, QubitUnitary, QueuingManager
from pennylane.measurements import MeasurementProcess, SampleMP
from pennylane.operation import AnyWires, Operation, Operator, Wires
from pennylane.tape import QuantumTape

from catalyst.jax_primitives import (
    AbstractQbit,
    AbstractQreg,
    Qreg,
    adjoint_p,
    compbasis,
    compbasis_p,
    counts,
    expval,
    hamiltonian,
    hermitian,
    namedobs,
    probs,
    qalloc,
    qcond_p,
    qdealloc,
    qdevice,
    qextract,
    qfor_p,
    qinsert,
    qinst,
    qmeasure_p,
    qunitary,
    qwhile_p,
    sample,
    state,
    tensorobs,
)
from catalyst.jax_primitives import var as jprim_var
from catalyst.jax_tracer import (
    KNOWN_NAMED_OBS,
    custom_lower_jaxpr_to_module,
    trace_observables,
)
from catalyst.utils.exceptions import CompileError
from catalyst.utils.jax_extras import (
    initial_style_jaxprs_with_common_consts1,
    initial_style_jaxprs_with_common_consts2,
    new_main2,
    sort_eqns,
)
from catalyst.utils.tracing import TracingContext


class EvaluationMode(Enum):
    QJIT_QNODE = 0
    QJIT = 1
    EXEC = 2


def get_evaluation_mode() -> Tuple[EvaluationMode, Any]:
    is_tracing = TracingContext.is_tracing()
    if is_tracing:
        ctx = qml.QueuingManager.active_context()
        if ctx is not None:
            mctx = get_main_tracing_context()
            assert mctx is not None
            return (EvaluationMode.QJIT_QNODE, mctx)
        else:
            return (EvaluationMode.QJIT, None)
    else:
        return (EvaluationMode.EXEC, None)


def _aval_to_primitive_type(aval):
    if isinstance(aval, DynamicJaxprTracer):
        aval = aval.strip_weak_type()
    if isinstance(aval, ShapedArray):
        aval = aval.dtype
    assert not isinstance(aval, (list, dict)), f"Unexpected type {aval}"
    return aval


def _check_single_bool_value(tree: PyTreeDef, avals: List[Any], hint=None) -> None:
    hint = f"{hint}: " if hint else ""
    if not treedef_is_leaf(tree):
        raise TypeError(
            f"{hint}A single boolean scalar was expected, got value of tree-shape: {tree}."
        )
    assert len(avals) == 1, f"{avals} does not match {tree}"
    dtype = _aval_to_primitive_type(avals[0])
    if dtype not in (bool, jnp.bool_):
        raise TypeError(f"{hint}A single boolean scalar was expected, got value {avals[0]}.")


def _check_same_types(trees: List[PyTreeDef], avals: List[List[Any]], hint=None) -> None:
    assert len(trees) == len(avals), f"Input trees ({trees}) don't match input avals ({avals})"
    hint = f"{hint}: " if hint else ""
    expected_tree, expected_dtypes = trees[0], [_aval_to_primitive_type(a) for a in avals[0]]
    for i, (tree, aval) in list(enumerate(zip(trees, avals)))[1:]:
        if tree != expected_tree:
            raise TypeError(
                f"{hint}Same return types were expected, got:\n"
                f" - Branch at index 0: {expected_tree}\n"
                f" - Branch at index {i}: {tree}\n"
                "Please specify the default branch if none was specified."
            )
        dtypes = [_aval_to_primitive_type(a) for a in aval]
        if dtypes != expected_dtypes:
            raise TypeError(
                f"{hint}Same return types were expected, got:\n"
                f" - Branch at index 0: {expected_dtypes}\n"
                f" - Branch at index {i}: {dtypes}\n"
                "Please specify the default branch if none was specified."
            )


@dataclass
class MainTracingContext:
    main: JaxMainTrace
    frames: Dict[DynamicJaxprTrace, JaxprStackFrame]
    mains: Dict[DynamicJaxprTrace, JaxMainTrace]
    trace: Optional[DynamicJaxprTrace] = None

TRACING_CONTEXT : Optional[MainTracingContext] = None

@contextmanager
def main_tracing_context() -> ContextManager[MainTracingContext]:
    global TRACING_CONTEXT
    with new_base_main(DynamicJaxprTrace, dynamic=True) as main:
        main.jaxpr_stack = ()
        TRACING_CONTEXT = ctx = MainTracingContext(main, {}, {})
        try:
            yield ctx
        finally:
            TRACING_CONTEXT = None

def get_main_tracing_context(hint=None) -> MainTracingContext:
    """ Checks a number of tracing conditions and return the MainTracingContext """
    msg = f"{hint or 'catalyst functions'} can only be used from within @qjit decorated code."
    TracingContext.check_is_tracing(msg)
    if TRACING_CONTEXT is None:
        raise CompileError(f"{hint} can only be used from within a qml.qnode.")
    return TRACING_CONTEXT


@contextmanager
def frame_tracing_context(ctx: MainTracingContext,
                          trace: Optional[DynamicJaxprTrace] = None
                          ) -> ContextManager[DynamicJaxprTrace]:
    main = ctx.mains[trace] if trace is not None else None
    with new_main2(DynamicJaxprTrace, dynamic=True, main=main) as nmain:
        nmain.jaxpr_stack = ()
        frame = JaxprStackFrame() if trace is None else ctx.frames[trace]
        with extend_jaxpr_stack(nmain, frame), reset_name_stack():
            parent_trace = ctx.trace
            ctx.trace = DynamicJaxprTrace(nmain, cur_sublevel()) if trace is None else trace
            ctx.frames[ctx.trace] = frame
            ctx.mains[ctx.trace] = nmain
            try:
                yield ctx.trace
            finally:
                ctx.trace = parent_trace


def deduce_avals(f: Callable, args, kwargs):
    flat_args, in_tree = tree_flatten((args, kwargs))
    wf = lu.wrap_init(f)
    in_avals, keep_inputs = list(map(shaped_abstractify, flat_args)), [True] * len(flat_args)
    in_type = tuple(zip(in_avals, keep_inputs))
    wff, out_tree_promise = flatten_fun(wf, in_tree)
    wffa = lu.annotate(wff, in_type)
    return wffa, in_avals, out_tree_promise


def new_inner_tracer(trace: DynamicJaxprTrace, aval) -> JaxprTracer:
    dt = DynamicJaxprTracer(trace, aval, jax_current())
    trace.frame.tracers.append(dt)
    trace.frame.tracer_to_var[id(dt)] = trace.frame.newvar(aval)
    return dt


@dataclass
class HybridOpRegion:
    """A code region of a nested HybridOp operation containing a JAX trace manager, a quantum tape,
    input and output classical tracers."""

    trace: DynamicJaxprTrace
    quantum_tape: Optional[QuantumTape]
    arg_classical_tracers: List[DynamicJaxprTracer]
    res_classical_tracers: List[DynamicJaxprTracer]


class HybridOp(Operation):
    """A model of an operation carrying nested quantum region. Simplified analog of
    catalyst.ForLoop, catalyst.WhileLoop, catalyst.Adjoin, etc"""

    num_wires = AnyWires

    def __init__(self, in_classical_tracers, out_classical_tracers, regions: List[HybridOpRegion]):
        self.in_classical_tracers = in_classical_tracers
        self.out_classical_tracers = out_classical_tracers
        self.regions = regions
        super().__init__(wires=Wires(HybridOp.num_wires))

    def __repr__(self):
        """Constructor-call-like representation."""
        return f"{self.name}(tapes={[r.quantum_tape.operations for r in self.regions]})"


def has_nested_tapes(op: Operation) -> bool:
    return (
        isinstance(op, HybridOp)
        and len(op.regions) > 0
        and any([r.quantum_tape is not None for r in op.regions])
    )


class ForLoop(HybridOp):
    pass


class MidCircuitMeasure(HybridOp):
    pass


class Cond(HybridOp):
    pass


class WhileLoop(HybridOp):
    pass


class Adjoint(HybridOp):
    pass


class CondCallable:
    def __init__(self, pred, true_fn):
        self.preds = [pred]
        self.branch_fns = [true_fn]
        self.otherwise_fn = lambda: None

    def else_if(self, pred):
        def decorator(branch_fn):
            if branch_fn.__code__.co_argcount != 0:
                raise TypeError(
                    "Conditional 'else if' function is not allowed to have any arguments"
                )
            self.preds.append(pred)
            self.branch_fns.append(branch_fn)
            return self

        return decorator

    def otherwise(self, otherwise_fn):
        if otherwise_fn.__code__.co_argcount != 0:
            raise TypeError("Conditional 'False' function is not allowed to have any arguments")
        self.otherwise_fn = otherwise_fn
        return self

    @staticmethod
    def _check_branches_return_types(branch_jaxprs):
        expected = branch_jaxprs[0].out_avals[:-1]
        for i, jaxpr in list(enumerate(branch_jaxprs))[1:]:
            if expected != jaxpr.out_avals[:-1]:
                raise TypeError(
                    "Conditional branches all require the same return type, got:\n"
                    f" - Branch at index 0: {expected}\n"
                    f" - Branch at index {i}: {jaxpr.out_avals[:-1]}\n"
                    "Please specify an else branch if none was specified."
                )

    def _call_with_quantum_ctx(self, ctx):
        outer_trace = ctx.trace
        in_classical_tracers = self.preds
        regions: List[HybridOpRegion] = []

        out_trees, out_avals = [], []
        for branch in self.branch_fns + [self.otherwise_fn]:
            quantum_tape = QuantumTape()
            with frame_tracing_context(ctx) as inner_trace:
                wffa, in_avals, out_tree = deduce_avals(branch, [], {})
                with QueuingManager.stop_recording(), quantum_tape:
                    res_classical_tracers = [inner_trace.full_raise(t) for t in wffa.call_wrapped()]
            regions.append(HybridOpRegion(inner_trace, quantum_tape, [], res_classical_tracers))
            out_trees.append(out_tree())
            out_avals.append(res_classical_tracers)

        _check_same_types(out_trees, out_avals, hint="Conditional branches")
        res_avals = list(map(shaped_abstractify, res_classical_tracers))
        out_classical_tracers = [new_inner_tracer(outer_trace, aval) for aval in res_avals]
        Cond(in_classical_tracers, out_classical_tracers, regions)
        return tree_unflatten(out_tree(), out_classical_tracers)

    def _call_with_classical_ctx(self):
        args, args_tree = tree_flatten([])
        args_avals = tuple(map(_abstractify, args))
        branch_jaxprs, consts, out_trees = initial_style_jaxprs_with_common_consts1(
            (*self.branch_fns, self.otherwise_fn), args_tree, args_avals, "cond"
        )
        _check_same_types(
            out_trees, [j.out_avals for j in branch_jaxprs], hint="Conditional branches"
        )
        out_classical_tracers = qcond_p.bind(*(self.preds + consts), branch_jaxprs=branch_jaxprs)
        return tree_unflatten(out_trees[0], out_classical_tracers)

    def _call_during_interpretation(self):
        for pred, branch_fn in zip(self.preds, self.branch_fns):
            if pred:
                return branch_fn()
        return self.otherwise_fn()

    def __call__(self):
        mode, ctx = get_evaluation_mode()
        if mode == EvaluationMode.QJIT_QNODE:
            return self._call_with_quantum_ctx(ctx)
        elif mode == EvaluationMode.QJIT:
            return self._call_with_classical_ctx()
        elif mode == EvaluationMode.EXEC:
            return self._call_during_interpretation()
        raise RuntimeError(f"Unsupported evaluation mode {mode}")


def cond(pred: DynamicJaxprTracer):
    def _decorator(true_fn: Callable):
        if true_fn.__code__.co_argcount != 0:
            raise TypeError("Conditional 'True' function is not allowed to have any arguments")
        return CondCallable(pred, true_fn)

    return _decorator


def for_loop(lower_bound, upper_bound, step):
    def _body_query(body_fn):
        def _call_handler(*init_state):
            def _call_with_quantum_ctx(ctx: MainTracingContext):
                quantum_tape = QuantumTape()
                outer_trace = ctx.trace
                with frame_tracing_context(ctx) as inner_trace:
                    in_classical_tracers = [
                        lower_bound,
                        upper_bound,
                        step,
                        lower_bound,
                    ] + tree_flatten(init_state)[0]
                    wffa, in_avals, body_tree = deduce_avals(
                        body_fn, [lower_bound] + list(init_state), {}
                    )
                    arg_classical_tracers = _input_type_to_tracers(inner_trace.new_arg, in_avals)
                    with QueuingManager.stop_recording(), quantum_tape:
                        res_classical_tracers = [
                            inner_trace.full_raise(t)
                            for t in wffa.call_wrapped(*arg_classical_tracers)
                        ]

                res_avals = list(map(shaped_abstractify, res_classical_tracers))
                out_classical_tracers = [new_inner_tracer(outer_trace, aval) for aval in res_avals]
                ForLoop(
                    in_classical_tracers,
                    out_classical_tracers,
                    [
                        HybridOpRegion(
                            inner_trace, quantum_tape, arg_classical_tracers, res_classical_tracers
                        )
                    ],
                )

                return tree_unflatten(body_tree(), out_classical_tracers)

            def _call_with_classical_ctx():
                iter_arg = lower_bound
                init_vals, in_tree = tree_flatten((iter_arg, *init_state))
                init_avals = tuple(_abstractify(val) for val in init_vals)
                body_jaxpr, body_consts, body_tree = _initial_style_jaxpr(
                    body_fn, in_tree, init_avals, "for_loop"
                )

                apply_reverse_transform = isinstance(step, int) and step < 0
                out_classical_tracers = qfor_p.bind(
                    lower_bound,
                    upper_bound,
                    step,
                    *(body_consts + init_vals),
                    body_jaxpr=body_jaxpr,
                    body_nconsts=len(body_consts),
                    apply_reverse_transform=apply_reverse_transform,
                )

                return tree_unflatten(body_tree, out_classical_tracers)

            def _call_during_interpretation():
                args = init_state
                fn_res = args if len(args) > 1 else args[0] if len(args) == 1 else None
                for i in range(lower_bound, upper_bound, step):
                    fn_res = body_fn(i, *args)
                    args = fn_res if len(args) > 1 else (fn_res,) if len(args) == 1 else ()
                return fn_res

            mode, ctx = get_evaluation_mode()
            if mode == EvaluationMode.QJIT_QNODE:
                return _call_with_quantum_ctx(ctx)
            elif mode == EvaluationMode.QJIT:
                return _call_with_classical_ctx()
            elif mode == EvaluationMode.EXEC:
                return _call_during_interpretation()
            raise RuntimeError(f"Unsupported evaluation mode {mode}")

        return _call_handler

    return _body_query


def while_loop(cond_fn):
    def _body_query(body_fn):
        def _call_handler(*init_state):
            def _call_with_quantum_ctx(ctx:MainTracingContext):
                outer_trace = ctx.trace
                in_classical_tracers, in_tree = tree_flatten(init_state)

                with frame_tracing_context(ctx) as cond_trace:
                    cond_wffa, cond_in_avals, cond_tree = deduce_avals(cond_fn, init_state, {})
                    arg_classical_tracers = _input_type_to_tracers(
                        cond_trace.new_arg, cond_in_avals
                    )
                    res_classical_tracers = [
                        cond_trace.full_raise(t)
                        for t in cond_wffa.call_wrapped(*arg_classical_tracers)
                    ]
                    cond_region = HybridOpRegion(
                        cond_trace, None, arg_classical_tracers, res_classical_tracers
                    )

                _check_single_bool_value(
                    cond_tree(), res_classical_tracers, hint="Condition return value"
                )

                with frame_tracing_context(ctx) as body_trace:
                    wffa, in_avals, body_tree = deduce_avals(body_fn, init_state, {})
                    arg_classical_tracers = _input_type_to_tracers(body_trace.new_arg, in_avals)
                    quantum_tape = QuantumTape()
                    with QueuingManager.stop_recording(), quantum_tape:
                        res_classical_tracers = [
                            body_trace.full_raise(t)
                            for t in wffa.call_wrapped(*arg_classical_tracers)
                        ]
                    body_region = HybridOpRegion(
                        body_trace, quantum_tape, arg_classical_tracers, res_classical_tracers
                    )

                res_avals = list(map(shaped_abstractify, res_classical_tracers))
                out_classical_tracers = [new_inner_tracer(outer_trace, aval) for aval in res_avals]

                WhileLoop(in_classical_tracers, out_classical_tracers, [cond_region, body_region])
                return tree_unflatten(body_tree(), out_classical_tracers)

            def _call_with_classical_ctx():
                init_vals, in_tree = tree_flatten(init_state)
                init_avals = tuple(_abstractify(val) for val in init_vals)
                cond_jaxpr, cond_consts, cond_tree = _initial_style_jaxpr(
                    cond_fn, in_tree, init_avals, "while_cond"
                )
                body_jaxpr, body_consts, body_tree = _initial_style_jaxpr(
                    body_fn, in_tree, init_avals, "while_loop"
                )
                _check_single_bool_value(
                    cond_tree, cond_jaxpr.out_avals, hint="Condition return value"
                )
                out_classical_tracers = qwhile_p.bind(
                    *(cond_consts + body_consts + init_vals),
                    cond_jaxpr=cond_jaxpr,
                    body_jaxpr=body_jaxpr,
                    cond_nconsts=len(cond_consts),
                    body_nconsts=len(body_consts),
                )
                return tree_unflatten(body_tree, out_classical_tracers)

            def _call_during_interpretation():
                args = init_state
                fn_res = args if len(args) > 1 else args[0] if len(args) == 1 else None
                while cond_fn(*args):
                    fn_res = body_fn(*args)
                    args = fn_res if len(args) > 1 else (fn_res,) if len(args) == 1 else ()
                return fn_res

            mode, ctx = get_evaluation_mode()
            if mode == EvaluationMode.QJIT_QNODE:
                return _call_with_quantum_ctx(ctx)
            elif mode == EvaluationMode.QJIT:
                return _call_with_classical_ctx()
            elif mode == EvaluationMode.EXEC:
                return _call_during_interpretation()
            raise RuntimeError(f"Unsupported evaluation mode {mode}")

        return _call_handler

    return _body_query


def measure(wires) -> JaxprTracer:
    ctx = get_main_tracing_context("catalyst.measure")
    wires = list(wires) if isinstance(wires, (list, tuple)) else [wires]
    if len(wires) != 1:
        raise TypeError(f"One classical argument (a wire) is expected, got {wires}")
    # assert len(ctx.trace.frame.eqns) == 0, ctx.trace.frame.eqns
    out_classical_tracer = new_inner_tracer(ctx.trace, jax.core.get_aval(True))
    MidCircuitMeasure(
        in_classical_tracers=wires, out_classical_tracers=[out_classical_tracer], regions=[]
    )
    return out_classical_tracer


def adjoint(f: Union[Callable, Operator]) -> Union[Callable, Operator]:
    def _call_handler(*args, _callee: Callable, **kwargs):
        ctx = get_main_tracing_context()
        with frame_tracing_context(ctx) as inner_trace:
            in_classical_tracers, _ = tree_flatten((args, kwargs))
            wffa, in_avals, _ = deduce_avals(_callee, args, kwargs)
            arg_classical_tracers = _input_type_to_tracers(inner_trace.new_arg, in_avals)
            quantum_tape = QuantumTape()
            with QueuingManager.stop_recording(), quantum_tape:
                # FIXME: move all full_raise calls into a separate function
                res_classical_tracers = [
                    inner_trace.full_raise(t)
                    for t in wffa.call_wrapped(*arg_classical_tracers)
                    if isinstance(t, DynamicJaxprTracer)
                ]

            if len(quantum_tape.measurements) > 0:
                raise ValueError("Quantum measurements are not allowed in Adjoints")

            adjoint_region = HybridOpRegion(
                inner_trace, quantum_tape, arg_classical_tracers, res_classical_tracers
            )

        Adjoint(
            in_classical_tracers=in_classical_tracers,
            out_classical_tracers=[],
            regions=[adjoint_region],
        )
        return None

    if isinstance(f, Callable):

        def _callable(*args, **kwargs):
            return _call_handler(*args, _callee=f, **kwargs)

        return _callable
    elif isinstance(f, Operator):
        QueuingManager.remove(f)

        def _callee():
            QueuingManager.append(f)

        return _call_handler(_callee=_callee)
    else:
        raise ValueError(f"Expected a callable or a qml.Operator, not {f}")


def trace_quantum_tape(quantum_tape:QuantumTape,
                       device:QubitDevice,
                       qreg:DynamicJaxprTracer,
                       ctx:MainTracingContext,
                       trace:DynamicJaxprTrace) -> DynamicJaxprTracer:
    """ Recursively trace the nested `quantum_tape` and produce the quantum tracers. With quantum
    tracers we can complete the set of tracers and finally emit the JAXPR of the whole quantum
    program."""
    # Notes:
    # [1] - We are interested only in a new quantum tracer, so we ignore all others.
    # [2] - HACK: We add alread existing classical tracers into the last JAX equation.

    def bind_overwrite_classical_tracers(op: HybridOp, binder, *args, **kwargs):
        """Binds the primitive `prim` but override the returned classical tracers with the already
        existing output tracers of the operation `op`."""
        out_quantum_tracer = binder(*args, **kwargs)[-1]
        eqn = ctx.frames[trace].eqns[-1]
        assert (len(eqn.outvars) - 1) == len(op.out_classical_tracers)
        for i, t in zip(range(len(eqn.outvars) - 1), op.out_classical_tracers):  # [2]
            eqn.outvars[i] = trace.getvar(t)
        return op.out_classical_tracers + [out_quantum_tracer]

    for op in device.expand_fn(quantum_tape):
        qreg2 = None
        if isinstance(op, HybridOp):
            if isinstance(op, ForLoop):
                inner_trace = op.regions[0].trace
                inner_tape = op.regions[0].quantum_tape
                res_classical_tracers = op.regions[0].res_classical_tracers

                with frame_tracing_context(ctx, inner_trace):
                    qreg_in = _input_type_to_tracers(inner_trace.new_arg, [AbstractQreg()])[0]
                    qreg_out = trace_quantum_tape(inner_tape, device, qreg_in, ctx, inner_trace)
                    jaxpr, typ, consts = ctx.frames[inner_trace].to_jaxpr2(
                        res_classical_tracers + [qreg_out]
                    )

                step = op.in_classical_tracers[2]
                apply_reverse_transform = isinstance(step, int) and step < 0
                qreg2 = bind_overwrite_classical_tracers(
                    op,
                    qfor_p.bind,
                    op.in_classical_tracers[0],
                    op.in_classical_tracers[1],
                    step,
                    *(consts + op.in_classical_tracers[3:] + [qreg]),
                    body_jaxpr=ClosedJaxpr(convert_constvars_jaxpr(jaxpr), ()),
                    body_nconsts=len(consts),
                    apply_reverse_transform=apply_reverse_transform,
                )[
                    -1
                ]  # [1]

            elif isinstance(op, Cond):
                jaxprs, consts = [], []
                for region in op.regions:
                    with frame_tracing_context(ctx, region.trace):
                        qreg_in = _input_type_to_tracers(region.trace.new_arg, [AbstractQreg()])[0]
                        qreg_out = trace_quantum_tape(
                            region.quantum_tape, device, qreg_in, ctx, region.trace
                        )
                        jaxpr, typ, const = ctx.frames[region.trace].to_jaxpr2(
                            region.res_classical_tracers + [qreg_out]
                        )
                        jaxprs.append(jaxpr)
                        consts.append(const)

                jaxprs2, combined_consts = initial_style_jaxprs_with_common_consts2(jaxprs, consts)

                qreg2 = bind_overwrite_classical_tracers(
                    op,
                    qcond_p.bind,
                    *(op.in_classical_tracers + combined_consts + [qreg]),
                    branch_jaxprs=jaxprs2,
                )[
                    -1
                ]  # [1]

            elif isinstance(op, WhileLoop):
                cond_trace = op.regions[0].trace
                res_classical_tracers = op.regions[0].res_classical_tracers
                with frame_tracing_context(ctx, cond_trace):
                    _input_type_to_tracers(cond_trace.new_arg, [AbstractQreg()])[0]
                    cond_jaxpr, _, cond_consts = ctx.frames[cond_trace].to_jaxpr2(
                        res_classical_tracers
                    )

                body_trace = op.regions[1].trace
                body_tape = op.regions[1].quantum_tape
                res_classical_tracers = op.regions[1].res_classical_tracers
                with frame_tracing_context(ctx, body_trace):
                    qreg_in = _input_type_to_tracers(body_trace.new_arg, [AbstractQreg()])[0]
                    qreg_out = trace_quantum_tape(body_tape, device, qreg_in, ctx, body_trace)
                    body_jaxpr, _, body_consts = ctx.frames[body_trace].to_jaxpr2(
                        res_classical_tracers + [qreg_out]
                    )

                qreg2 = bind_overwrite_classical_tracers(
                    op,
                    qwhile_p.bind,
                    *(cond_consts + body_consts + op.in_classical_tracers + [qreg]),
                    cond_jaxpr=ClosedJaxpr(convert_constvars_jaxpr(cond_jaxpr), ()),
                    body_jaxpr=ClosedJaxpr(convert_constvars_jaxpr(body_jaxpr), ()),
                    cond_nconsts=len(cond_consts),
                    body_nconsts=len(body_consts),
                )[
                    -1
                ]  # [1]

            elif isinstance(op, MidCircuitMeasure):
                wire = op.in_classical_tracers[0]
                qubit = qextract(qreg, wire)
                qubit2 = bind_overwrite_classical_tracers(op, qmeasure_p.bind, qubit)[-1]  # [1]
                qreg2 = qinsert(qreg, wire, qubit2)

            elif isinstance(op, Adjoint):
                body_trace = op.regions[0].trace
                body_tape = op.regions[0].quantum_tape
                res_classical_tracers = op.regions[0].res_classical_tracers
                with frame_tracing_context(ctx, body_trace):
                    qreg_in = _input_type_to_tracers(body_trace.new_arg, [AbstractQreg()])[0]
                    qreg_out = trace_quantum_tape(body_tape, device, qreg_in, ctx, body_trace)
                    body_jaxpr, _, body_consts = ctx.frames[body_trace].to_jaxpr2(
                        res_classical_tracers + [qreg_out]
                    )

                args, args_tree = tree_flatten((body_consts, op.in_classical_tracers, [qreg]))
                op_results = adjoint_p.bind(
                    *args,
                    args_tree=args_tree,
                    jaxpr=ClosedJaxpr(convert_constvars_jaxpr(body_jaxpr), ()),
                )
                qreg2 = op_results[0]
            else:
                raise NotImplementedError(f"{op=}")
        else:
            if isinstance(op, MeasurementProcess):
                qreg2 = qreg
            else:
                qubits = [qextract(qreg, wire) for wire in op.wires]
                if isinstance(op, QubitUnitary):
                    qubits2 = qunitary(*[*op.parameters, *qubits])
                else:
                    qubits2 = qinst(op.name, len(qubits), *qubits, *op.parameters)
                # FIXME: Port the qubit-state caching logic from the original tracer
                qreg2 = qreg
                for wire, qubit2 in zip(op.wires, qubits2):
                    qreg2 = qinsert(qreg2, wire, qubit2)

        assert qreg2 is not None
        qreg = qreg2

    ctx.frames[trace].eqns = sort_eqns(ctx.frames[trace].eqns)
    return qreg


def trace_observables(
    obs: Operation,
    device,
    qreg,
    m_wires
) -> Tuple[List[DynamicJaxprTracer], List[DynamicJaxprTracer]]:
    wires = obs.wires if (obs and len(obs.wires) > 0) else m_wires
    qubits = [qextract(qreg, w) for w in wires]
    if obs is None:
        obs_tracers = compbasis(*qubits)
    elif isinstance(obs, KNOWN_NAMED_OBS):
        obs_tracers = namedobs(type(obs).__name__, qubits[0])
    elif isinstance(obs, qml.Hermitian):
        # TODO: remove asarray once fixed upstream: https://github.com/PennyLaneAI/pennylane/issues/4263
        obs_tracers = hermitian(jax.numpy.asarray(*obs.parameters), *qubits)
    elif isinstance(obs, qml.operation.Tensor):
        nested_obs = [trace_observables(o, device, qreg, m_wires)[0] for o in obs.obs]
        obs_tracers = tensorobs(*nested_obs)
    elif isinstance(obs, qml.Hamiltonian):
        nested_obs = [trace_observables(o, device, qreg, m_wires)[0] for o in obs.ops]
        obs_tracers = hamiltonian(jax.numpy.asarray(obs.parameters), *nested_obs)
    else:
        raise NotImplementedError(f"Observable {obs} is not impemented")
    return obs_tracers, qubits


def trace_quantum_measurements(quantum_tape,
                               device:QubitDevice,
                               qreg:DynamicJaxprTracer,
                               ctx:MainTracingContext,
                               trace:DynamicJaxprTrace,
                               outputs:List[Union[MeasurementProcess, DynamicJaxprTracer]],
                               out_tree
                               ) -> List[DynamicJaxprTracer]:
    shots = device.shots
    out_classical_tracers = []

    for i, o in enumerate(outputs):
        if isinstance(o, MeasurementProcess):
            m_wires = o.wires if o.wires else range(device.num_wires)
            obs_tracers, qubits = trace_observables(o.obs, device, qreg, m_wires)

            using_compbasis = obs_tracers.primitive == compbasis_p
            if o.return_type.value == "sample":
                shape = (shots, len(qubits)) if using_compbasis else (shots,)
                out_classical_tracers.append(sample(obs_tracers, shots, shape))
            elif o.return_type.value == "expval":
                out_classical_tracers.append(expval(obs_tracers, shots))
            elif o.return_type.value == "var":
                out_classical_tracers.append(jprim_var(obs_tracers, shots))
            elif o.return_type.value == "probs":
                assert using_compbasis
                shape = (2 ** len(qubits),)
                out_classical_tracers.append(probs(obs_tracers, shape))
            elif o.return_type.value == "counts":
                shape = (2 ** len(qubits),) if using_compbasis else (2,)
                out_classical_tracers.extend(counts(obs_tracers, shots, shape))
                counts_tree = tree_structure(("keys", "counts"))
                meas_return_trees_children = out_tree.children()
                if len(meas_return_trees_children) > 0:
                    meas_return_trees_children[i] = counts_tree
                    out_tree = out_tree.make_from_node_data_and_children(
                        out_tree.node_data(), meas_return_trees_children
                    )
                else:
                    out_tree = counts_tree
            elif o.return_type.value == "state":
                assert using_compbasis
                shape = (2 ** len(qubits),)
                out_classical_tracers.append(state(obs_tracers, shape))
            else:
                raise NotImplementedError(f"Measurement {o.return_type.value} is not impemented")
        elif isinstance(o, (list, dict)):
            raise CompileError(f"Expected a tracer or a measurement, got {o}")
        elif isinstance(o, DynamicJaxprTracer):
            out_classical_tracers.append(o)
        else:
            # FIXME: Constants (numbers) all go here. What about explicitly listing the allowed
            # types and only allow these? Anyway, one must change type hints for `outputs`
            out_classical_tracers.append(o)

    return out_classical_tracers, out_tree


def trace_quantum_function(
    f: Callable, device: QubitDevice, args, kwargs
) -> Tuple[ClosedJaxpr, Any]:
    """Trace quantum function in a way that allows building a nested quantum tape describing the
    whole algorithm. Tape transformations are supported allowing users to modify the algorithm
    before the final jaxpr is created.
    The tracing is done in parts as follows: 1) Classical tracing, classical JAX tracers
    and the quantum tape are produced 2) Quantum tape transformation 3) Quantum tape tracing, the
    remaining quantum JAX tracers and the final JAXPR are produced."""

    with main_tracing_context() as ctx:
        # [1] - Classical tracing
        quantum_tape = QuantumTape()
        with frame_tracing_context(ctx) as trace:
            wffa, in_avals, out_tree_promise = deduce_avals(f, args, kwargs)
            in_classical_tracers = _input_type_to_tracers(trace.new_arg, in_avals)
            with QueuingManager.stop_recording(), quantum_tape:
                # [2] - Quantum tape transformations happen at the end of tracing
                ans = wffa.call_wrapped(*in_classical_tracers)
            out_classical_tracers_or_measurements = [
                (trace.full_raise(t) if isinstance(t, DynamicJaxprTracer) else t) for t in ans
            ]

        # [3] - Quantum tracing
        with frame_tracing_context(ctx, trace):
            qdevice("kwargs", str(device.backend_kwargs))
            qdevice("backend", device.backend_name)
            qreg_in = qalloc(len(device.wires))
            qreg_out = trace_quantum_tape(quantum_tape, device, qreg_in, ctx, trace)
            out_classical_tracers, out_classical_tree = trace_quantum_measurements(
                quantum_tape,
                device,
                qreg_out,
                ctx,
                trace,
                out_classical_tracers_or_measurements,
                out_tree_promise(),
            )

            qdealloc(qreg_in)

            out_classical_tracers = [trace.full_raise(t) for t in out_classical_tracers]
            out_quantum_tracers = [qreg_out]

            jaxpr, out_type, consts = ctx.frames[trace].to_jaxpr2(
                out_classical_tracers + out_quantum_tracers
            )
            jaxpr._outvars = jaxpr._outvars[:-1]
            out_type = out_type[:-1]
            # FIXME: `check_jaxpr` complains about the `AbstractQreg` type. Consider fixing.
            # check_jaxpr(jaxpr)

    closed_jaxpr = ClosedJaxpr(jaxpr, consts)
    out_avals, _ = unzip2(out_type)
    out_shape = tree_unflatten(
        out_classical_tree, [ShapeDtypeStruct(a.shape, a.dtype, a.named_shape) for a in out_avals]
    )
    return closed_jaxpr, out_shape
