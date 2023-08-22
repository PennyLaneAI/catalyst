import jax
import pennylane as qml
from catalyst.utils.jax_extras import new_main2, sort_eqns
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
from jax._src.lax.control_flow.common import _initial_style_jaxprs_with_common_consts2
from jax._src import linear_util as lu
from jax._src.tree_util import (tree_flatten, tree_unflatten)
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
from jax._src.util import unzip2
from jax._src.lax.lax import xb, xla
from catalyst.jax_primitives import (Qreg, AbstractQreg, AbstractQbit, qinst, qextract, qinsert,
                                     qfor_p, qcond_p, qmeasure_p, qdevice, qalloc,
                                     qdealloc, compbasis,
                                     sample, namedobs, hermitian, expval, state, compbasis_p)
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

class HybridOp(Operation):
    """ A model of an operation carrying nested quantum region. Simplified analog of
    catalyst.ForLoop, catalyst.WhileLoop, catalyst.Adjoin, etc """
    num_wires = AnyWires

    def __init__(self, quantum_tapes, in_classical_tracers, out_classical_tracers,
                 arg_classical_tracers, res_classical_tracers, inner_traces):
        self.quantum_tapes = quantum_tapes
        self.in_classical_tracers = in_classical_tracers
        self.out_classical_tracers = out_classical_tracers
        self.arg_classical_tracers = arg_classical_tracers
        self.res_classical_tracers = res_classical_tracers
        self.traces = inner_traces
        # kwargs["wires"] = Wires(HybridOp.num_wires)
        super().__init__(wires = Wires(HybridOp.num_wires))

    def has_nested_tapes(self) -> bool:
        properties = [self.quantum_tapes, self.traces]
        assert all([len(x)>0 for x in properties]) or \
               all([len(x)==0 for  x in properties])
        return len(self.quantum_tapes)>0

    def __repr__(self):
        """Constructor-call-like representation."""
        return f"{self.name}(tape={self.quantum_tapes[0].operations})"

class ForLoop(HybridOp):
    pass

class MidCircuitMeasure(HybridOp):
    pass

class Cond(HybridOp):
    pass

def hybrid(callee:Callable):
    """ A model frontend extension function. Simplified analog of for_loop, while_loop, adjoint,
    etc. """
    def _call(*args, **kwargs):
        ctx = get_main_tracing_context()
        quantum_tape = QuantumTape()
        outer_trace = ctx.trace
        with frame_tracing_context(ctx) as inner_trace:

            in_classical_tracers = args # FIXME: save kwargs and the tree structure
            wffa, in_avals, _ = deduce_avals(callee, args, kwargs)
            arg_classical_tracers = _input_type_to_tracers(inner_trace.new_arg, in_avals)
            with QueuingManager.stop_recording(), quantum_tape:
                res_classical_tracers = wffa.call_wrapped(*arg_classical_tracers)
            res_avals = list(map(shaped_abstractify, res_classical_tracers))

        out_classical_tracers = [new_inner_tracer(outer_trace, aval) for aval in res_avals]

        HybridOp([quantum_tape],
                 in_classical_tracers,
                 out_classical_tracers,
                 arg_classical_tracers,
                 res_classical_tracers,
                 [inner_trace])

        return out_classical_tracers[0] # FIXME: generalize to the arbitrary number of retvals

    return _call

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
        ctx = get_main_tracing_context()
        outer_trace = ctx.trace

        quantum_tapes = []
        inner_traces = []
        in_classical_tracers = self.preds
        arg_classical_tracers = []
        res_classical_tracers = []

        for branch in self.branch_fns + [self.otherwise_fn]:
            quantum_tape = QuantumTape()
            with frame_tracing_context(ctx) as inner_trace:
                wffa, in_avals, _ = deduce_avals(branch, [], {})
                with QueuingManager.stop_recording(), quantum_tape:
                    rct = wffa.call_wrapped()
                    res_tracers = [inner_trace.full_raise(t) for t in rct]
                inner_traces.append(inner_trace)
                res_classical_tracers.append(res_tracers)
            quantum_tapes.append(quantum_tape)

        res_avals = list(map(shaped_abstractify, res_tracers))
        out_classical_tracers = [new_inner_tracer(outer_trace, aval) for aval in res_avals]

        Cond(quantum_tapes,
             in_classical_tracers,
             out_classical_tracers,
             arg_classical_tracers,
             res_classical_tracers,
             inner_traces)

        return out_classical_tracers

    def _call_with_classical_ctx(self):
        raise NotImplementedError()

    def _call_during_interpretation(self):
        raise NotImplementedError()

    def __call__(self):
        is_tracing = TracingContext.is_tracing()
        if is_tracing:
            ctx = qml.QueuingManager.active_context()
            if ctx is None:
                return self._call_with_classical_ctx()
            else:
                return self._call_with_quantum_ctx(ctx)
        else:
            return self._call_during_interpretation()


def cond(pred:DynamicJaxprTracer):

    def _decorator(true_fn:Callable):
        if true_fn.__code__.co_argcount != 0:
            raise TypeError("Conditional 'True' function is not allowed to have any arguments")
        return CondCallable(pred, true_fn)

    return _decorator

def for_loop(lower_bound, upper_bound, step):
    def _body_query(callee):
        def _call_handler(*init_state):
            ctx = get_main_tracing_context()
            quantum_tape = QuantumTape()
            outer_trace = ctx.trace
            with frame_tracing_context(ctx) as inner_trace:

                in_classical_tracers = [lower_bound, upper_bound, step, lower_bound] + list(init_state)
                wffa, in_avals, _ = deduce_avals(callee, [lower_bound] + list(init_state), {})
                arg_classical_tracers = _input_type_to_tracers(inner_trace.new_arg, in_avals)
                with QueuingManager.stop_recording(), quantum_tape:
                    rct = wffa.call_wrapped(*arg_classical_tracers)
                    res_classical_tracers = [inner_trace.full_raise(t) for t in rct]
                res_avals = list(map(shaped_abstractify, res_classical_tracers))

            out_classical_tracers = [new_inner_tracer(outer_trace, aval) for aval in res_avals]

            ForLoop([quantum_tape],
                    in_classical_tracers,
                    out_classical_tracers,
                    arg_classical_tracers,
                    res_classical_tracers,
                    [inner_trace])

            return out_classical_tracers
        return _call_handler
    return _body_query


def measure(wires) -> JaxprTracer:
    ctx = get_main_tracing_context("catalyst.measure")
    wires = list(wires) if isinstance(wires, (list, tuple)) else [wires]
    if len(wires) != 1:
        raise TypeError(f"One classical argument (a wire) is expected, got {wires}")
    # assert len(ctx.trace.frame.eqns) == 0, ctx.trace.frame.eqns
    result = new_inner_tracer(ctx.trace, jax.core.get_aval(True))
    MidCircuitMeasure(
        quantum_tapes=[],
        in_classical_tracers=wires,
        out_classical_tracers=[result],
        arg_classical_tracers=[],
        res_classical_tracers=[],
        inner_traces=[])
    return result

hybrid_p = jax.core.Primitive("hybrid")
hybrid_p.multiple_results = True


@hybrid_p.def_abstract_eval
def _abstract_eval(*args, body_jaxpr, **kwargs):
    return [AbstractQreg()]


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
            eqn = None
            if op.has_nested_tapes():

                if isinstance(op, ForLoop):
                    inner_trace = op.traces[0]
                    inner_tape = op.quantum_tapes[0]

                    with frame_tracing_context(ctx, inner_trace):
                        qreg_in = _input_type_to_tracers(inner_trace.new_arg, [AbstractQreg()])[0]
                        qreg_out = trace_quantum_tape(inner_tape, device, qreg_in, ctx, inner_trace)
                        jaxpr, typ, consts = ctx.frames[inner_trace].to_jaxpr2(op.res_classical_tracers + [qreg_out])

                    qreg2 = bind_overwrite_classical_tracers(
                        op, qfor_p.bind,
                        op.in_classical_tracers[0],
                        op.in_classical_tracers[1],
                        op.in_classical_tracers[2],
                        *(consts + op.in_classical_tracers[3:] + [qreg]),
                        body_jaxpr=ClosedJaxpr(convert_constvars_jaxpr(jaxpr), ()),
                        body_nconsts=len(consts),
                        apply_reverse_transform=False)[-1] # [1]

                elif isinstance(op, Cond):
                    jaxprs, consts = [], []
                    for inner_trace, inner_tape, rct in zip(op.traces,
                                                            op.quantum_tapes,
                                                            op.res_classical_tracers):
                        with frame_tracing_context(ctx, inner_trace):
                            qreg_in = _input_type_to_tracers(inner_trace.new_arg, [AbstractQreg()])[0]
                            qreg_out = trace_quantum_tape(inner_tape, device, qreg_in, ctx, inner_trace)
                            jaxpr, typ, const = ctx.frames[inner_trace].to_jaxpr2(rct + [qreg_out])
                            jaxprs.append(jaxpr)
                            consts.append(const)

                    jaxprs2, combined_consts = _initial_style_jaxprs_with_common_consts2(jaxprs, consts)

                    qreg2  = bind_overwrite_classical_tracers(
                        op, qcond_p.bind,
                        *(op.in_classical_tracers + combined_consts + [qreg]),
                        branch_jaxprs=jaxprs2)[-1]
                else:
                    raise TypeError(f"Operation {op} is not implemented")
            else:
                if isinstance(op, MidCircuitMeasure):
                    wire = op.in_classical_tracers[0]
                    qubit = qextract(qreg, wire)
                    qubit2 = bind_overwrite_classical_tracers(op, qmeasure_p.bind, qubit)[-1] # [1]
                    qreg2 = qinsert(qreg, wire, qubit2)
                else:
                    qreg2 = bind_overwrite_classical_tracers(op, hybrid_p.bind, qreg,
                                          *chain(op.in_classical_tracers, consts),
                                          body_jaxpr=ClosedJaxpr(jaxpr, consts))[-1] # [1]
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
                               outputs:List[Union[MeasurementProcess, DynamicJaxprTracer]]
                               ) -> List[DynamicJaxprTracer]:
    shots = device.shots
    out_classical_tracers = []

    for i, o in enumerate(outputs):
        if isinstance(o, DynamicJaxprTracer):
            out_classical_tracers.append(o)
        elif isinstance(o, MeasurementProcess):
            op,obs = o,o.obs
            qubits = [qextract(qreg, wire) for wire in op.wires]
            if obs is None:
                obs_tracers = compbasis(*qubits)
            elif isinstance(obs, KNOWN_NAMED_OBS):
                obs_tracers = namedobs(type(obs).__name__, qubits[0])
            elif isinstance(obs, qml.Hermitian):
                assert False, f"{obs.matrix=}"
                obs_tracers = hermitian(matrix, *qubits)
            # elif isinstance(obs, qml.operation.Tensor):
            #     nested_obs = [trace_observables(o, qubit_states, p, num_wires, qreg)[0] for o in obs.obs]
            #     obs_tracers = jprim.tensorobs(*nested_obs)
            # elif isinstance(obs, qml.Hamiltonian):
            #     nested_obs = [trace_observables(o, qubit_states, p, num_wires, qreg)[0] for o in obs.ops]
            #     obs_tracers = trace_hamiltonian(op_args, *nested_obs)
            else:
                raise NotImplementedError(f"Observable {obs} is not impemented")

            mres_tracer = None
            if o.return_type.value == "sample":
                shape = (shots, len(qubits)) if obs is None else (shots,)
                mres_tracer = sample(obs_tracers, shots, shape)
            elif o.return_type.value == "expval":
                mres_tracer = expval(obs_tracers, shots)
            elif o.return_type.value == "state":
                # assert obs is None, "Expected no observables for the state 'measurement'"
                assert obs_tracers.primitive == compbasis_p
                shape = (2 ** len(qubits),)
                mres_tracer = state(obs_tracers, shape)
            else:
                raise NotImplementedError(f"Measurement {o.return_type.value} is not impemented")
            out_classical_tracers.append(mres_tracer)
        else:
            raise CompileError(f"Expected a tracer or a measurement, got {o}")

    return out_classical_tracers


def trace_quantum_function(
    f:Callable,
    device:QubitDevice,
    args, kwargs,
    transform:Optional[Callable[[QuantumTape],QuantumTape]]=None) -> Tuple[ClosedJaxpr, Any]:
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
                ans = wffa.call_wrapped(*in_classical_tracers)
            # FIXME: Do we need this ? list(map(trace.full_raise, ans))
            out_classical_tracers_or_measurement = ans

        # [2] - Tape transformations
        transformed_tape = transform(quantum_tape) if transform else quantum_tape

        # [3] - Quantum tracing
        with frame_tracing_context(ctx, trace):

            qdevice("kwargs", str(device.backend_kwargs))
            qdevice("backend", device.backend_name)
            qreg_in = qalloc(len(device.wires))
            qreg_out = trace_quantum_tape(transformed_tape, device, qreg_in, ctx, trace)
            out_classical_tracers = trace_quantum_measurements(transformed_tape, device, qreg_out,
                                                               ctx, trace,
                                                               out_classical_tracers_or_measurement)
            qdealloc(qreg_in)
            # FIXME: Do we need full_rasie here ? [trace.full_raise(qreg_out)]
            out_quantum_tracers = [qreg_out]

            jaxpr, out_type, consts = ctx.frames[trace].to_jaxpr2(out_classical_tracers + out_quantum_tracers)
            jaxpr._outvars = jaxpr._outvars[:-1]
            out_type = out_type[:-1]
            # FIXME: `check_jaxpr` complains about the `AbstractQreg` type. Consider fixing.
            # check_jaxpr(jaxpr)

    closed_jaxpr = ClosedJaxpr(jaxpr, consts)
    out_avals, _ = unzip2(out_type)
    out_shape = tree_unflatten(out_tree_promise(),
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

