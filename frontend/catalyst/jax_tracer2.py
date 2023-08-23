import jax
import pennylane as qml
from catalyst.utils.jax_extras import (new_main2, sort_eqns,
                                       initial_style_jaxprs_with_common_consts,
                                       initial_style_jaxprs_with_common_consts2)
from jax._src.core import (ClosedJaxpr, MainTrace as JaxMainTrace, new_main,
                           new_base_main, cur_sublevel, get_aval, Tracer as JaxprTracer,
                           check_jaxpr, ShapedArray, JaxprEqn, Var)
from jax._src.interpreters.partial_eval import (DynamicJaxprTrace, DynamicJaxprTracer,
                                                JaxprStackFrame, trace_to_subjaxpr_dynamic2,
                                                extend_jaxpr_stack, _input_type_to_tracers,
                                                new_jaxpr_eqn, make_jaxpr_effects, Jaxpr,
                                                _const_folding_and_forwarding, _inline_literals,
                                                _add_implicit_outputs, convert_constvars_jaxpr)
from jax._src.source_info_util import reset_name_stack, current as jax_current, new_name_stack
from jax._src.dispatch import jaxpr_replicas
from jax._src.lax.lax import _abstractify
from jax._src import linear_util as lu
from jax._src.tree_util import (tree_flatten, tree_structure, tree_unflatten)
from jax._src.api import ShapeDtypeStruct
from jax._src.api_util import (
    flatten_fun, apply_flat_fun, flatten_fun_nokwargs, flatten_fun_nokwargs2,
    argnums_partial, argnums_partial_except, flatten_axes, donation_vector,
    rebase_donate_argnums, _ensure_index, _ensure_index_tuple,
    shaped_abstractify, _ensure_str_tuple, apply_flat_fun_nokwargs,
    check_callable, debug_info, result_paths, flat_out_axes, debug_info_final)
from jax.interpreters import mlir
from jax.interpreters.mlir import (
    AxisContext,
    ModuleContext,
    ReplicaAxisContext,
    ir,
    lower_jaxpr_to_fun,
    lowerable_effects,
)
from jax._src.interpreters.mlir import (
    _constant_handlers
)
from jax._src.lax.control_flow import (
    _initial_style_jaxpr,
)
from jax._src.util import unzip2
from jax._src.lax.lax import xb, xla
from catalyst.jax_primitives import (Qreg, AbstractQreg, AbstractQbit, qinst, qextract, qinsert,
                                     qfor_p, qcond_p, qmeasure_p, qdevice, qalloc, qwhile_p,
                                     qdealloc, compbasis, probs, sample, namedobs, hermitian,
                                     expval, state, counts, compbasis_p)
from catalyst.jax_tracer import (custom_lower_jaxpr_to_module, trace_observables, KNOWN_NAMED_OBS)
from catalyst.utils.tracing import TracingContext
from catalyst.utils.exceptions import CompileError
from typing import Optional, Callable, List, ContextManager, Tuple, Any, Dict, Union
from pennylane import QubitDevice, QueuingManager, Device
from pennylane.operation import AnyWires, Operation, Wires
from pennylane.measurements import MeasurementProcess, SampleMP
from pennylane.tape import QuantumTape
from itertools import chain
from contextlib import contextmanager
from collections import defaultdict

from dataclasses import dataclass
from enum import Enum

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


@dataclass
class MainTracingContex:
    main: JaxMainTrace
    frames: Dict[DynamicJaxprTrace, JaxprStackFrame]
    mains: Dict[DynamicJaxprTrace, JaxMainTrace]
    trace: Optional[DynamicJaxprTrace] = None

TRACING_CONTEXT : Optional[MainTracingContex] = None

@contextmanager
def main_tracing_context() -> ContextManager[MainTracingContex]:
    global TRACING_CONTEXT
    with new_base_main(DynamicJaxprTrace, dynamic=True) as main:
        main.jaxpr_stack = ()
        TRACING_CONTEXT = ctx = MainTracingContex(main, {}, {})
        try:
            yield ctx
        finally:
            TRACING_CONTEXT = None

def get_main_tracing_context(hint=None) -> MainTracingContex:
    """ Checks a number of tracing conditions and return the MainTracingContex """
    msg = f"{hint or 'catalyst functions'} can only be used from within @qjit decorated code."
    TracingContext.check_is_tracing(msg)
    if TRACING_CONTEXT is None:
        raise CompileError(f"{hint} can only be used from within a qml.qnode.")
    return TRACING_CONTEXT


@contextmanager
def frame_tracing_context(ctx: MainTracingContex,
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


def deduce_avals(f:Callable, args, kwargs):
    flat_args, in_tree = tree_flatten((args, kwargs))
    wf = lu.wrap_init(f)
    in_avals, keep_inputs = list(map(shaped_abstractify,flat_args)), [True]*len(flat_args)
    in_type = tuple(zip(in_avals, keep_inputs))
    wff, out_tree_promise = flatten_fun(wf, in_tree)
    wffa = lu.annotate(wff, in_type)
    return wffa, in_avals, out_tree_promise

def new_inner_tracer(trace:DynamicJaxprTrace, aval) -> JaxprTracer:
    dt = DynamicJaxprTracer(trace, aval, jax_current())
    trace.frame.tracers.append(dt)
    trace.frame.tracer_to_var[id(dt)] = trace.frame.newvar(aval)
    return dt


@dataclass
class HybridOpRegion:
    """ A code region of a nested HybridOp operation containing a JAX trace manager, a quantum tape,
    input and output classical tracers. """
    trace:DynamicJaxprTrace
    quantum_tape:Optional[QuantumTape]
    arg_classical_tracers:List[DynamicJaxprTracer]
    res_classical_tracers:List[DynamicJaxprTracer]

class HybridOp(Operation):
    """ A model of an operation carrying nested quantum region. Simplified analog of
    catalyst.ForLoop, catalyst.WhileLoop, catalyst.Adjoin, etc """
    num_wires = AnyWires

    def __init__(self, in_classical_tracers, out_classical_tracers, regions:List[HybridOpRegion]):
        self.in_classical_tracers = in_classical_tracers
        self.out_classical_tracers = out_classical_tracers
        self.regions = regions
        super().__init__(wires = Wires(HybridOp.num_wires))

    def __repr__(self):
        """Constructor-call-like representation."""
        return f"{self.name}(tapes={[r.quantum_tape.operations for r in self.regions]})"

def has_nested_tapes(op:Operation) -> bool:
    return isinstance(op, HybridOp) and len(op.regions)>0 and \
        any([r.quantum_tape is not None for r in op.regions])

class ForLoop(HybridOp):
    pass

class MidCircuitMeasure(HybridOp):
    pass

class Cond(HybridOp):
    pass

class WhileLoop(HybridOp):
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
        regions:List[HybridOpRegion] = []

        for branch in self.branch_fns + [self.otherwise_fn]:
            quantum_tape = QuantumTape()
            with frame_tracing_context(ctx) as inner_trace:
                wffa, in_avals, out_tree = deduce_avals(branch, [], {})
                with QueuingManager.stop_recording(), quantum_tape:
                    res_classical_tracers = [inner_trace.full_raise(t) for t in wffa.call_wrapped()]
            regions.append(HybridOpRegion(inner_trace, quantum_tape, [], res_classical_tracers))

        res_avals = list(map(shaped_abstractify, res_classical_tracers))
        out_classical_tracers = [new_inner_tracer(outer_trace, aval) for aval in res_avals]
        Cond(in_classical_tracers, out_classical_tracers, regions)
        return tree_unflatten(out_tree(), out_classical_tracers)

    def _call_with_classical_ctx(self):
        args, args_tree = tree_flatten([])
        args_avals = tuple(map(_abstractify, args))
        branch_jaxprs, consts, out_trees = initial_style_jaxprs_with_common_consts(
            (*self.branch_fns, self.otherwise_fn), args_tree, args_avals, "cond"
        )
        out_classical_tracers = qcond_p.bind(*(self.preds + consts), branch_jaxprs=branch_jaxprs)
        return tree_unflatten(out_trees[0], out_classical_tracers)

    def _call_during_interpretation(self):
        raise NotImplementedError()

    def __call__(self):
        mode, ctx = get_evaluation_mode()
        if mode == EvaluationMode.QJIT_QNODE:
            return self._call_with_quantum_ctx(ctx)
        elif mode == EvaluationMode.QJIT:
            return self._call_with_classical_ctx()
        elif mode == EvaluationMode.EXEC:
            return self._call_during_interpretation()
        else:
            raise RuntimeError(f"Unsupported evaluation mode {mode}")


def cond(pred:DynamicJaxprTracer):

    def _decorator(true_fn:Callable):
        if true_fn.__code__.co_argcount != 0:
            raise TypeError("Conditional 'True' function is not allowed to have any arguments")
        return CondCallable(pred, true_fn)

    return _decorator

def for_loop(lower_bound, upper_bound, step):
    def _body_query(body_fn):
        def _call_handler(*init_state):
            def _call_with_quantum_ctx(ctx: MainTracingContex):
                quantum_tape = QuantumTape()
                outer_trace = ctx.trace
                with frame_tracing_context(ctx) as inner_trace:

                    in_classical_tracers = [lower_bound, upper_bound, step, lower_bound] + tree_flatten(init_state)[0]
                    wffa, in_avals, body_tree = deduce_avals(body_fn, [lower_bound] + list(init_state), {})
                    arg_classical_tracers = _input_type_to_tracers(inner_trace.new_arg, in_avals)
                    with QueuingManager.stop_recording(), quantum_tape:
                        res_classical_tracers = [inner_trace.full_raise(t) for t in
                                                 wffa.call_wrapped(*arg_classical_tracers)]

                res_avals = list(map(shaped_abstractify, res_classical_tracers))
                out_classical_tracers = [new_inner_tracer(outer_trace, aval) for aval in res_avals]

                ForLoop(in_classical_tracers,
                        out_classical_tracers,
                        [HybridOpRegion(inner_trace,
                                        quantum_tape,
                                        arg_classical_tracers,
                                        res_classical_tracers)])

                return tree_unflatten(body_tree(), out_classical_tracers)

            def _call_with_classical_ctx():
                iter_arg = lower_bound
                init_vals, in_tree = tree_flatten((iter_arg, *init_state))
                init_avals = tuple(_abstractify(val) for val in init_vals)
                body_jaxpr, body_consts, body_tree = _initial_style_jaxpr(
                    body_fn, in_tree, init_avals, "for_loop"
                )

                out_classical_tracers = qfor_p.bind(
                    lower_bound,
                    upper_bound,
                    step,
                    *(body_consts + init_vals),
                    body_jaxpr=body_jaxpr,
                    body_nconsts=len(body_consts),
                    # FIXME: handle the loop reverse conditions
                    apply_reverse_transform=False)

                return tree_unflatten(body_tree, out_classical_tracers)

            mode, ctx = get_evaluation_mode()
            if mode == EvaluationMode.QJIT_QNODE:
                return _call_with_quantum_ctx(ctx)
            elif mode == EvaluationMode.QJIT:
                return _call_with_classical_ctx()
            # elif mode == EvaluationMode.EXEC:
            #     return _call_during_interpretation()
            else:
                raise RuntimeError(f"Unsupported evaluation mode {mode}")
        return _call_handler
    return _body_query


def while_loop(cond_fn):
    def _body_query(body_fn):
        def _call_handler(*init_state):
            def _call_with_quantum_ctx(ctx:MainTracingContex):
                outer_trace = ctx.trace
                in_classical_tracers, in_tree = tree_flatten(init_state)

                with frame_tracing_context(ctx) as cond_trace:
                    cond_wffa, cond_in_avals, _ = deduce_avals(cond_fn, init_state, {})
                    arg_classical_tracers = _input_type_to_tracers(cond_trace.new_arg, cond_in_avals)
                    res_classical_tracers = [cond_trace.full_raise(t) for t in
                                             cond_wffa.call_wrapped(*arg_classical_tracers)]
                    cond_region = HybridOpRegion(
                        cond_trace,
                        None,
                        arg_classical_tracers,
                        res_classical_tracers)

                with frame_tracing_context(ctx) as body_trace:
                    wffa, in_avals, body_tree = deduce_avals(body_fn, init_state, {})
                    arg_classical_tracers = _input_type_to_tracers(body_trace.new_arg, in_avals)
                    quantum_tape = QuantumTape()
                    with QueuingManager.stop_recording(), quantum_tape:
                        res_classical_tracers = [body_trace.full_raise(t) for t in
                                                 wffa.call_wrapped(*arg_classical_tracers)]
                    body_region = HybridOpRegion(
                        body_trace,
                        quantum_tape,
                        arg_classical_tracers,
                        res_classical_tracers)

                res_avals = list(map(shaped_abstractify, res_classical_tracers))
                out_classical_tracers = [new_inner_tracer(outer_trace, aval) for aval in res_avals]

                WhileLoop(in_classical_tracers,
                          out_classical_tracers,
                          [cond_region, body_region])
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
                out_classical_tracers = qwhile_p.bind(
                    *(cond_consts + body_consts + init_vals),
                    cond_jaxpr=cond_jaxpr,
                    body_jaxpr=body_jaxpr,
                    cond_nconsts=len(cond_consts),
                    body_nconsts=len(body_consts))
                return tree_unflatten(body_tree, out_classical_tracers)

            mode, ctx = get_evaluation_mode()
            if mode == EvaluationMode.QJIT_QNODE:
                return _call_with_quantum_ctx(ctx)
            elif mode == EvaluationMode.QJIT:
                return _call_with_classical_ctx()
            # elif mode == EvaluationMode.EXEC:
            #     return _call_during_interpretation()
            else:
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
        in_classical_tracers=wires,
        out_classical_tracers=[out_classical_tracer],
        regions=[])
    return out_classical_tracer


def trace_quantum_tape(quantum_tape:QuantumTape,
                       device:QubitDevice,
                       qreg:DynamicJaxprTracer,
                       ctx:MainTracingContex,
                       trace:DynamicJaxprTrace) -> DynamicJaxprTracer:
    """ Recursively trace the nested `quantum_tape` and produce the quantum tracers. With quantum
    tracers we can complete the set of tracers and finally emit the JAXPR of the whole quantum
    program. """
    # Notes:
    # [1] - We are interested only in a new quantum tracer, so we ignore all others.
    # [2] - HACK: We add alread existing classical tracers into the last JAX equation.

    def bind_overwrite_classical_tracers(op:HybridOp, binder, *args, **kwargs):
        """ Binds the primitive `prim` but override the returned classical tracers with the already
        existing output tracers of the operation `op`."""
        out_quantum_tracer = binder(*args, **kwargs)[-1]
        eqn = ctx.frames[trace].eqns[-1]
        assert (len(eqn.outvars)-1) == len(op.out_classical_tracers)
        for i,t in zip(range(len(eqn.outvars)-1), op.out_classical_tracers): # [2]
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
                        res_classical_tracers + [qreg_out])

                qreg2 = bind_overwrite_classical_tracers(
                    op, qfor_p.bind,
                    op.in_classical_tracers[0],
                    op.in_classical_tracers[1],
                    op.in_classical_tracers[2],
                    *(consts + op.in_classical_tracers[3:] + [qreg]),
                    body_jaxpr=ClosedJaxpr(convert_constvars_jaxpr(jaxpr), ()),
                    body_nconsts=len(consts),
                    # FIXME: handle the loop reverse conditions
                    apply_reverse_transform=False)[-1] # [1]

            elif isinstance(op, Cond):
                jaxprs, consts = [], []
                for region in op.regions:
                    with frame_tracing_context(ctx, region.trace):
                        qreg_in = _input_type_to_tracers(region.trace.new_arg, [AbstractQreg()])[0]
                        qreg_out = trace_quantum_tape(region.quantum_tape, device, qreg_in, ctx, region.trace)
                        jaxpr, typ, const = ctx.frames[region.trace].to_jaxpr2(region.res_classical_tracers + [qreg_out])
                        jaxprs.append(jaxpr)
                        consts.append(const)

                jaxprs2, combined_consts = initial_style_jaxprs_with_common_consts2(jaxprs, consts)

                qreg2  = bind_overwrite_classical_tracers(
                    op, qcond_p.bind,
                    *(op.in_classical_tracers + combined_consts + [qreg]),
                    branch_jaxprs=jaxprs2)[-1] # [1]

            elif isinstance(op, WhileLoop):
                cond_trace = op.regions[0].trace
                res_classical_tracers = op.regions[0].res_classical_tracers
                with frame_tracing_context(ctx, cond_trace):
                    _input_type_to_tracers(cond_trace.new_arg, [AbstractQreg()])[0]
                    cond_jaxpr, _, cond_consts = ctx.frames[cond_trace].to_jaxpr2(
                        res_classical_tracers)

                body_trace = op.regions[1].trace
                body_tape = op.regions[1].quantum_tape
                res_classical_tracers = op.regions[1].res_classical_tracers
                with frame_tracing_context(ctx, body_trace):
                    qreg_in = _input_type_to_tracers(body_trace.new_arg, [AbstractQreg()])[0]
                    qreg_out = trace_quantum_tape(body_tape, device, qreg_in, ctx, body_trace)
                    body_jaxpr, _, body_consts = ctx.frames[body_trace].to_jaxpr2(
                        res_classical_tracers + [qreg_out])

                qreg2 = bind_overwrite_classical_tracers(
                    op, qwhile_p.bind,
                    *(cond_consts + body_consts + op.in_classical_tracers + [qreg]),
                    cond_jaxpr=ClosedJaxpr(convert_constvars_jaxpr(cond_jaxpr), ()),
                    body_jaxpr=ClosedJaxpr(convert_constvars_jaxpr(body_jaxpr), ()),
                    cond_nconsts=len(cond_consts),
                    body_nconsts=len(body_consts))[-1] # [1]

            elif isinstance(op, MidCircuitMeasure):
                wire = op.in_classical_tracers[0]
                qubit = qextract(qreg, wire)
                qubit2 = bind_overwrite_classical_tracers(op, qmeasure_p.bind, qubit)[-1] # [1]
                qreg2 = qinsert(qreg, wire, qubit2)
            else:
                raise NotImplementedError(f"{op=}")
        else:
            if isinstance(op, MeasurementProcess):
                qreg2 = qreg
            else:
                # FIXME: Port the qubit-state caching logic from the original tracer
                qubits = [qextract(qreg, wire) for wire in op.wires]
                qubits2 = qinst(op.name, len(qubits), *qubits, *op.parameters)
                qreg2 = qreg
                for wire, qubit2 in zip(op.wires, qubits2):
                    qreg2 = qinsert(qreg2, wire, qubit2)

        assert qreg2 is not None
        qreg = qreg2

    ctx.frames[trace].eqns = sort_eqns(ctx.frames[trace].eqns)
    return qreg


def trace_quantum_measurements(quantum_tape,
                               device:QubitDevice,
                               qreg:DynamicJaxprTracer,
                               ctx:MainTracingContex,
                               trace:DynamicJaxprTrace,
                               outputs:List[Union[MeasurementProcess, DynamicJaxprTracer]],
                               out_tree
                               ) -> List[DynamicJaxprTracer]:
    shots = device.shots
    out_classical_tracers = []

    for i, o in enumerate(outputs):
        if isinstance(o, DynamicJaxprTracer):
            out_classical_tracers.append(o)
        elif isinstance(o, MeasurementProcess):
            op,obs = o,o.obs
            wires = op.wires if len(op.wires)>0 else range(device.num_wires)
            qubits = [qextract(qreg, w) for w in wires]
            if obs is None:
                obs_tracers = compbasis(*qubits)
            elif isinstance(obs, KNOWN_NAMED_OBS):
                obs_tracers = namedobs(type(obs).__name__, qubits[0])
            elif isinstance(obs, qml.Hermitian):
                obs_tracers = hermitian(obs.matrix(), *qubits)
            # elif isinstance(obs, qml.operation.Tensor):
            #     nested_obs = [trace_observables(o, qubit_states, p, num_wires, qreg)[0] for o in obs.obs]
            #     obs_tracers = jprim.tensorobs(*nested_obs)
            # elif isinstance(obs, qml.Hamiltonian):
            #     nested_obs = [trace_observables(o, qubit_states, p, num_wires, qreg)[0] for o in obs.ops]
            #     obs_tracers = trace_hamiltonian(op_args, *nested_obs)
            else:
                raise NotImplementedError(f"Observable {obs} is not impemented")

            using_compbasis = obs_tracers.primitive == compbasis_p
            if o.return_type.value == "sample":
                shape = (shots, len(qubits)) if obs is None else (shots,)
                out_classical_tracers.append(sample(obs_tracers, shots, shape))
            elif o.return_type.value == "expval":
                out_classical_tracers.append(expval(obs_tracers, shots))
            elif o.return_type.value == "probs":
                assert using_compbasis
                shape = (2 ** len(qubits),)
                out_classical_tracers.append(probs(obs_tracers, shape))
            elif o.return_type.value == "counts":
                shape = (2 ** len(qubits),) if using_compbasis else (2,)
                out_classical_tracers.extend(counts(obs_tracers, shots, shape))
                counts_tree = tree_structure(("keys", "counts"))
                meas_return_trees_children = out_tree.children()
                if len(meas_return_trees_children)>0:
                    meas_return_trees_children[i] = counts_tree
                    out_tree = (
                        out_tree.make_from_node_data_and_children(
                            out_tree.node_data(), meas_return_trees_children
                        )
                    )
                else:
                    out_tree = counts_tree
            elif o.return_type.value == "state":
                assert using_compbasis
                shape = (2 ** len(qubits),)
                out_classical_tracers.append(state(obs_tracers, shape))
            else:
                raise NotImplementedError(f"Measurement {o.return_type.value} is not impemented")
        else:
            raise CompileError(f"Expected a tracer or a measurement, got {o}")

    return out_classical_tracers, out_tree


def trace_quantum_function(
    f:Callable,
    device:QubitDevice,
    args,
    kwargs) -> Tuple[ClosedJaxpr, Any]:
    """ Trace quantum function in a way that allows building a nested quantum tape describing the
    whole algorithm. Tape transformations are supported allowing users to modify the algorithm
    before the final jaxpr is created.
    The tracing is done in parts as follows: 1) Classical tracing, classical JAX tracers
    and the quantum tape are produced 2) Quantum tape transformation 3) Quantum tape tracing, the
    remaining quantum JAX tracers and the final JAXPR are produced. """

    with main_tracing_context() as ctx:

        # [1] - Classical tracing
        quantum_tape = QuantumTape()
        with frame_tracing_context(ctx) as trace:

            wffa, in_avals, out_tree_promise = deduce_avals(f, args, kwargs)
            in_classical_tracers = _input_type_to_tracers(trace.new_arg, in_avals)
            with QueuingManager.stop_recording(), quantum_tape:
                # [2] - Quantum tape transformations happen at the end of tracing
                ans = wffa.call_wrapped(*in_classical_tracers)
            out_classical_tracers_or_measurements = \
                [(trace.full_raise(t) if isinstance(t, DynamicJaxprTracer) else t) for t in ans]

        # [3] - Quantum tracing
        with frame_tracing_context(ctx, trace):

            qdevice("kwargs", str(device.backend_kwargs))
            qdevice("backend", device.backend_name)
            qreg_in = qalloc(len(device.wires))
            qreg_out = trace_quantum_tape(quantum_tape, device, qreg_in, ctx, trace)
            out_classical_tracers, out_classical_tree = \
                trace_quantum_measurements(
                    quantum_tape, device, qreg_out, ctx, trace,
                    out_classical_tracers_or_measurements, out_tree_promise())

            qdealloc(qreg_in)

            out_classical_tracers = [trace.full_raise(t) for t in out_classical_tracers]
            out_quantum_tracers = [qreg_out]

            jaxpr, out_type, consts = ctx.frames[trace].to_jaxpr2(out_classical_tracers + out_quantum_tracers)
            jaxpr._outvars = jaxpr._outvars[:-1]
            out_type = out_type[:-1]
            # FIXME: `check_jaxpr` complains about the `AbstractQreg` type. Consider fixing.
            # check_jaxpr(jaxpr)

    closed_jaxpr = ClosedJaxpr(jaxpr, consts)
    out_avals, _ = unzip2(out_type)
    out_shape = tree_unflatten(out_classical_tree,
                               [ShapeDtypeStruct(a.shape, a.dtype, a.named_shape) for a in out_avals])
    return closed_jaxpr, out_shape


# def lower_jaxpr_to_mlir(jaxpr):

#     nrep = jaxpr_replicas(jaxpr)
#     effects = [eff for eff in jaxpr.effects if eff in jax.core.ordered_effects]
#     axis_context = ReplicaAxisContext(xla.AxisEnv(nrep, (), ()))
#     name_stack = new_name_stack(wrap_name("ok", "jit"))
#     module, context = custom_lower_jaxpr_to_module(
#         func_name="jit_func",
#         module_name="mlir_module",
#         jaxpr=jaxpr,
#         effects=effects,
#         platform="cpu",
#         axis_context=axis_context,
#         name_stack=name_stack,
#         donated_args=[],
#     )
#     return module

