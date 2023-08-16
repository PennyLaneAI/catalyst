import jax
import pennylane as qml
from jax._src.core import (ClosedJaxpr, MainTrace as JaxMainTrace, new_main, cur_sublevel, get_aval,
                           Tracer as JaxprTracer, check_jaxpr, ShapedArray, JaxprEqn, Var)
from jax._src.interpreters.partial_eval import (DynamicJaxprTrace, DynamicJaxprTracer,
                                                JaxprStackFrame, trace_to_subjaxpr_dynamic2,
                                                extend_jaxpr_stack, _input_type_to_tracers,
                                                new_jaxpr_eqn, make_jaxpr_effects, Jaxpr,
                                                _const_folding_and_forwarding, _inline_literals,
                                                _add_implicit_outputs)
from jax._src.source_info_util import reset_name_stack, current as jax_current, new_name_stack
from jax._src.dispatch import jaxpr_replicas
from jax._src.lax.lax import _abstractify
from jax._src import linear_util as lu
from jax._src.tree_util import (tree_flatten, tree_unflatten)
from jax._src.util import wrap_name, toposort
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
from jax._src.util import unzip2
from jax._src.lax.lax import xb, xla
from catalyst.jax_primitives import (Qreg, AbstractQreg, AbstractQbit, qinst, qextract, qinsert,
                                     _qfor_lowering, qmeasure_p, qdevice, qalloc, qdealloc)
from catalyst.jax_tracer import custom_lower_jaxpr_to_module
from catalyst.utils.tracing import TracingContext
from catalyst.utils.exceptions import CompileError
from typing import Optional, Callable, List, ContextManager, Tuple, Any, Dict
from pennylane import QubitDevice, QueuingManager, Device
from pennylane.operation import AnyWires, Operation, Wires
from pennylane.tape import QuantumTape
from itertools import chain
from contextlib import contextmanager
from collections import defaultdict

from dataclasses import dataclass


@dataclass
class MainTracingContex:
    main: JaxMainTrace
    frames: Dict[DynamicJaxprTrace, JaxprStackFrame]
    trace: Optional[DynamicJaxprTrace] = None

TRACING_CONTEXT : Optional[MainTracingContex] = None

@contextmanager
def main_tracing_context() -> ContextManager[MainTracingContex]:
    global TRACING_CONTEXT
    with new_main(DynamicJaxprTrace, dynamic=True) as main:
        main.jaxpr_stack = ()
        TRACING_CONTEXT = ctx = MainTracingContex(main, {})
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
    frame = JaxprStackFrame() if trace is None else ctx.frames[trace]
    with extend_jaxpr_stack(ctx.main, frame), reset_name_stack():
        parent_trace = ctx.trace
        ctx.trace = DynamicJaxprTrace(ctx.main, cur_sublevel()) if trace is None else trace
        ctx.frames[ctx.trace] = frame
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

    def __init__(self, quantum_tape, in_classical_tracers, out_classical_tracers,
                 arg_classical_tracers, res_classical_tracers, inner_trace):
        self.quantum_tape = quantum_tape
        self.in_classical_tracers = in_classical_tracers
        self.out_classical_tracers = out_classical_tracers
        self.arg_classical_tracers = arg_classical_tracers
        self.res_classical_tracers = res_classical_tracers
        self.trace = inner_trace
        # kwargs["wires"] = Wires(HybridOp.num_wires)
        super().__init__(wires = Wires(HybridOp.num_wires))

    def has_nested_tape(self) -> bool:
        properties = [self.quantum_tape, self.trace]
        assert all([x is None for x in properties]) or \
               all([x is not None for  x in properties])
        return self.quantum_tape is not None

    def __repr__(self):
        """Constructor-call-like representation."""
        return f"{self.name}(tape={self.quantum_tape.operations})"

class ForLoop(HybridOp):
    pass

class MidCircuitMeasure(HybridOp):
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

        HybridOp(quantum_tape,
                 in_classical_tracers,
                 out_classical_tracers,
                 arg_classical_tracers,
                 res_classical_tracers,
                 inner_trace)

        return out_classical_tracers[0] # FIXME: generalize to the arbitrary number of retvals

    return _call

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
                    res_classical_tracers = wffa.call_wrapped(*arg_classical_tracers)
                res_avals = list(map(shaped_abstractify, res_classical_tracers))
                out_classical_tracers = [new_inner_tracer(outer_trace, aval) for aval in res_avals]

            ForLoop(quantum_tape,
                    in_classical_tracers,
                    out_classical_tracers,
                    arg_classical_tracers,
                    res_classical_tracers,
                    inner_trace)

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
        quantum_tape=None,
        in_classical_tracers=wires,
        out_classical_tracers=[result],
        arg_classical_tracers=[],
        res_classical_tracers=[],
        inner_trace=None)
    return result

hybrid_p = jax.core.Primitive("hybrid")
hybrid_p.multiple_results = True

@hybrid_p.def_abstract_eval
def _abstract_eval(*args, body_jaxpr, **kwargs):
    return [AbstractQreg()]


def sort_eqns(eqns:List[JaxprEqn])->List[JaxprEqn]:
    class Box:
        def __init__(self, e):
            self.e = e
            self.parents = {}
    boxes = [Box(e) for e in eqns]
    origin:Dict[int,Box] = {}
    for b in boxes:
        origin.update({ov.count:b for ov in b.e.outvars})
    for b in boxes:
        b.parents = {origin[v.count] for v in b.e.invars if v.count in origin}
    return [b.e for b in toposort(boxes)]


def trace_quantum_tape(quantum_tape:QuantumTape,
                       device:QubitDevice,
                       qreg:JaxprTracer,
                       ctx:MainTracingContex,
                       trace:DynamicJaxprTrace) -> JaxprTracer:
    """ Recursively trace the nested `quantum_tape` and produce the quantum tracers. With quantum
    tracers we can complete the set of tracers and finally emit the JAXPR of the whole quantum
    program. """
    # Notes:
    # [1] - We are interested only in a new quantum tracer, so we ignore all others.
    # [2] - HACK: We add alread existing classical tracers into the last JAX equation.

    def bind_overwrite_classical_tracers(op:HybridOp, prim, *args, **kwargs):
        """ Binds the primitive `prim` but override the returned classical tracers with the already
        existing output tracers of the operation `op`."""
        out_quantum_tracer = prim.bind(*args, **kwargs)[-1]
        eqn = ctx.frames[trace].eqns[-1]
        assert (len(eqn.outvars)-1) == len(op.out_classical_tracers)
        for i,t in zip(range(len(eqn.outvars)-1), op.out_classical_tracers): # [2]
            eqn.outvars[i] = trace.getvar(t)
        return op.out_classical_tracers + [out_quantum_tracer]

    for op in device.expand_fn(quantum_tape):
        qreg2 = None
        if isinstance(op, HybridOp):
            eqn = None
            if op.has_nested_tape():
                with frame_tracing_context(ctx, op.trace):
                    qreg_in = _input_type_to_tracers(op.trace.new_arg, [AbstractQreg()])[0]
                    qreg_out = trace_quantum_tape(op.quantum_tape, device, qreg_in, ctx, op.trace)
                    jaxpr, typ, consts = ctx.frames[trace].to_jaxpr2(op.res_classical_tracers + [qreg_out])

                if isinstance(op, ForLoop):
                    qreg2 = bind_overwrite_classical_tracers(
                        op, forloop_p,
                        op.in_classical_tracers[0],
                        op.in_classical_tracers[1],
                        op.in_classical_tracers[2],
                        *(consts + op.in_classical_tracers[3:] + [qreg]),
                        body_jaxpr=ClosedJaxpr(jaxpr, consts),
                        body_nconsts=len(consts),
                        apply_reverse_transform=False)[-1] # [1]
                else:
                    raise TypeError(f"Operation {op} is not implemented")
            else:
                if isinstance(op, MidCircuitMeasure):
                    wire = op.in_classical_tracers[0]
                    qubit = qextract(qreg, wire)
                    qubit2 = bind_overwrite_classical_tracers(op, qmeasure_p, qubit)[-1] # [1]
                    qreg2 = qinsert(qreg, wire, qubit2)
                else:
                    qreg2 = bind_overwrite_classical_tracers(op, hybrid_p, qreg,
                                          *chain(op.in_classical_tracers, consts),
                                          body_jaxpr=ClosedJaxpr(jaxpr, consts))[-1] # [1]
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
            out_classical_tracers = ans # FIXME: Do we need this ? list(map(trace.full_raise, ans))

        # [2] - Tape transformations
        transformed_tape = transform(quantum_tape) if transform else quantum_tape

        # [3] - Quantum tracing
        with frame_tracing_context(ctx, trace):

            qdevice("kwargs", str(device.backend_kwargs))
            qdevice("backend", device.backend_name)
            qreg_in = qalloc(len(device.wires))
            qreg_out = trace_quantum_tape(transformed_tape, device, qreg_in, ctx, trace)
            # FIXME: Calling `qdealloc` with `qreg_out` leads to a runtime error. Why is this?
            qdealloc(qreg_in)
            # FIXME: Port the observable-tracing logic from the original tracer
            out_quantum_tracers = [trace.full_raise(qreg_out)]

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


def lower_jaxpr_to_mlir(jaxpr):

    nrep = jaxpr_replicas(jaxpr)
    effects = [eff for eff in jaxpr.effects if eff in jax.core.ordered_effects]
    axis_context = ReplicaAxisContext(xla.AxisEnv(nrep, (), ()))
    name_stack = new_name_stack(wrap_name("ok", "jit"))
    module, context = custom_lower_jaxpr_to_module(
        func_name="jit_func",
        module_name="mlir_module",
        jaxpr=jaxpr,
        effects=effects,
        platform="cpu",
        axis_context=axis_context,
        name_stack=name_stack,
        donated_args=[],
    )
    return module

