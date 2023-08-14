import jax
from jax._src.core import (ClosedJaxpr, MainTrace as JaxMainTrace, new_main, cur_sublevel, get_aval,
                           Tracer as JaxTracer)
from jax._src.interpreters.partial_eval import (DynamicJaxprTrace, DynamicJaxprTracer,
                                                JaxprStackFrame, trace_to_subjaxpr_dynamic2,
                                                extend_jaxpr_stack, _input_type_to_tracers,
                                                new_jaxpr_eqn)
from jax._src.source_info_util import reset_name_stack, current as jax_current, new_name_stack
from jax._src.dispatch import jaxpr_replicas
from jax._src.lax.lax import _abstractify
from jax._src import linear_util as lu
from jax._src.tree_util import (tree_flatten, tree_unflatten)
from jax._src.util import wrap_name
from jax._src.api_util import (
    flatten_fun, apply_flat_fun, flatten_fun_nokwargs, flatten_fun_nokwargs2,
    argnums_partial, argnums_partial_except, flatten_axes, donation_vector,
    rebase_donate_argnums, _ensure_index, _ensure_index_tuple,
    shaped_abstractify, _ensure_str_tuple, apply_flat_fun_nokwargs,
    check_callable, debug_info, result_paths, flat_out_axes, debug_info_final)
from jax.interpreters.mlir import (
    AxisContext,
    ModuleContext,
    ReplicaAxisContext,
    ir,
    lower_jaxpr_to_fun,
    lowerable_effects,
)
from jax._src.lax.lax import xb, xla
from catalyst.jax_primitives import Qreg, AbstractQreg, qinst, qextract, qinsert
from catalyst.jax_tracer import custom_lower_jaxpr_to_module
from typing import Optional, Callable, List, ContextManager, Tuple
from pennylane import QueuingManager
from pennylane.operation import AnyWires, Operation, Wires
from pennylane.tape import QuantumTape
from itertools import chain
from contextlib import contextmanager

from dataclasses import dataclass


@dataclass
class MainTracingContex:
    main: JaxMainTrace
    trace: Optional[DynamicJaxprTrace] = None

TRACING_CONTEXT : Optional[MainTracingContex] = None

@contextmanager
def main_tracing_context() -> ContextManager[MainTracingContex]:
    global TRACING_CONTEXT
    with new_main(DynamicJaxprTrace, dynamic=True) as main:
        main.jaxpr_stack = ()
        TRACING_CONTEXT = ctx = MainTracingContex(main)
        try:
            yield ctx
        finally:
            TRACING_CONTEXT = None

def get_main_tracing_context() -> MainTracingContex:
    """ Should be called from within the `main_tracing_context` manager """
    assert TRACING_CONTEXT is not None
    return TRACING_CONTEXT

@contextmanager
def frame_tracing_context(ctx: MainTracingContex,
                          frame: Optional[JaxprStackFrame] = None,
                          trace: Optional[DynamicJaxprTrace] = None
                          ) -> ContextManager[Tuple[JaxprStackFrame,DynamicJaxprTrace]]:
    frame = JaxprStackFrame() if frame is None else frame
    with extend_jaxpr_stack(ctx.main, frame), reset_name_stack():
        parent_trace = ctx.trace
        ctx.trace = DynamicJaxprTrace(ctx.main, cur_sublevel()) if trace is None else trace
        try:
            yield frame, ctx.trace
        finally:
            ctx.trace = parent_trace




def deduce_avals(f:Callable, args, kwargs):
    wf = lu.wrap_init(f)
    flat_args, in_tree = tree_flatten((args, kwargs))
    in_avals, keep_inputs = list(map(shaped_abstractify,flat_args)), [True]*len(flat_args)
    in_type = tuple(zip(in_avals, keep_inputs))
    wff, out_tree = flatten_fun(wf, in_tree)
    wffa = lu.annotate(wff, in_type)
    return wffa, in_avals


class HybridOp(Operation):
    """ A model of an operation carrying nested quantum region. Simplified analog of
    catalyst.ForLoop, catalyst.WhileLoop, catalyst.Adjoin, etc """
    num_wires = AnyWires

    def __init__(self, quantum_tape, in_classical_tracers, out_classical_tracers,
                 arg_classical_tracers, result_classical_tracers, frame, trace):
        self.quantum_tape = quantum_tape
        self.in_classical_tracers = in_classical_tracers
        self.out_classical_tracers = out_classical_tracers
        self.arg_classical_tracers = arg_classical_tracers
        self.result_classical_tracers = result_classical_tracers
        self.frame = frame
        self.trace = trace
        # kwargs["wires"] = Wires(HybridOp.num_wires)
        super().__init__(wires = Wires(HybridOp.num_wires))

    def __repr__(self):
        """Constructor-call-like representation."""
        return f"{self.name}(tape={self.quantum_tape.operations})"


def hybrid(callee:Callable):
    """ A model frontend extension function. Simplified analog of for_loop, while_loop, adjoint,
    etc. """
    def _call(*args, **kwargs):
        ctx = get_main_tracing_context()
        quantum_tape = QuantumTape()
        outer_trace = ctx.trace
        with frame_tracing_context(ctx) as (inner_frame,inner_trace):

            in_classical_tracers = args # FIXME: save kwargs and the tree structure
            wffa, in_avals = deduce_avals(callee, args, kwargs)
            in_tracers = _input_type_to_tracers(inner_trace.new_arg, in_avals)
            with QueuingManager.stop_recording(), quantum_tape:
                res_classical_tracers = wffa.call_wrapped(*in_tracers)
            res_avals = list(map(shaped_abstractify, res_classical_tracers))

        out_classical_tracers = [DynamicJaxprTracer(outer_trace, a, jax_current()) for a in res_avals]
        for t, aval in zip(out_classical_tracers, res_avals):
            outer_trace.frame.tracers.append(t)
            outer_trace.frame.tracer_to_var[id(t)] = outer_trace.frame.newvar(aval)

        HybridOp(quantum_tape,
                 in_classical_tracers,
                 out_classical_tracers,
                 res_classical_tracers,
                 inner_frame,
                 inner_trace)

        return out_classical_tracers[0] # FIXME: generalize to the arbitrary number of retvals

    return _call


class ForLoop(HybridOp):
    pass

def for_loop(lower_bound, upper_bound, step):
    def _body_query(callee):
        def _call(init_state):
            ctx = get_main_tracing_context()
            quantum_tape = QuantumTape()
            outer_trace = ctx.trace
            with frame_tracing_context(ctx) as (inner_frame,inner_trace):

                in_classical_tracers = [lower_bound, upper_bound, step]
                wffa, in_avals = deduce_avals(callee, [lower_bound, init_state], {})
                arg_classical_tracers = _input_type_to_tracers(inner_trace.new_arg, in_avals)
                with QueuingManager.stop_recording(), quantum_tape:
                    res_classical_tracers = wffa.call_wrapped(*arg_classical_tracers)
                res_avals = list(map(shaped_abstractify, res_classical_tracers))

            out_classical_tracers = [DynamicJaxprTracer(outer_trace, a, jax_current()) for a in res_avals]
            for t, aval in zip(out_classical_tracers, res_avals):
                outer_trace.frame.tracers.append(t)
                outer_trace.frame.tracer_to_var[id(t)] = outer_trace.frame.newvar(aval)

            ForLoop(quantum_tape,
                    in_classical_tracers,
                    out_classical_tracers,
                    arg_classical_tracers,
                    res_classical_tracers,
                    inner_frame,
                    inner_trace)

            return out_classical_tracers[0]
        return _call

    return _body_query


hybrid_p = jax.core.Primitive("hybrid")
hybrid_p.multiple_results = True

forloop_p = jax.core.Primitive("forloop")
forloop_p.multiple_results = True
# mlir.register_lowering(forloop_p, _qfor_lowering)

# HACK: At the bind time we already know classical output tracers so we will add the variables
# to the primitive explicitly. Here we only return quantum value to make quantum output tracer
# appear.

@hybrid_p.def_abstract_eval
def _abstract_eval(*args, body_jaxpr, **kwargs):
    return [AbstractQreg()]

@forloop_p.def_abstract_eval
def _abstract_eval(*args, body_jaxpr, **kwargs):
    return [AbstractQreg()]


def trace_quantum_tape(quantum_tape:QuantumTape,
                       ctx:MainTracingContex,
                       frame:JaxprStackFrame,
                       trace:DynamicJaxprTrace) -> List[JaxTracer]:
    """ Recursively trace the nested `quantum_tape` and produce the quantum tracers. With quantum
    tracers we can complete the set of tracers and finally emit the JAXPR of the whole quantum
    program. """
    qreg = _input_type_to_tracers(trace.new_arg, [AbstractQreg()])[0]
    for op in quantum_tape.operations:
        qreg2 = None
        if isinstance(op, HybridOp):
            with frame_tracing_context(ctx, op.frame, op.trace):
                qreg_out = trace_quantum_tape(op.quantum_tape, ctx, op.frame, op.trace)[0]
                jaxpr, typ, consts = op.frame.to_jaxpr2([qreg_out] + op.result_classical_tracers)

            # FIXME: generalize and remove the [-1] item querying
            if isinstance(op, ForLoop):
                qreg2 = forloop_p.bind(op.in_classical_tracers[0],
                                       op.in_classical_tracers[1],
                                       op.in_classical_tracers[2],
                                       *(consts + op.arg_classical_tracers + [qreg]),
                                       body_jaxpr=ClosedJaxpr(jaxpr, consts),
                                       body_nconcsts=len(consts),
                                       apply_reverse_transform=False)[0]
            else:
                qreg2 = hybrid_p.bind(qreg,
                                      *chain(op.in_classical_tracers, consts),
                                      body_jaxpr=ClosedJaxpr(jaxpr, consts))[0]

            # HACK: we add variables for the already existing tracers
            for t in op.out_classical_tracers:
                frame.eqns[-1].outvars.append(trace.getvar(t))
        else:
            # Handle custom operation. We support 1-qubit ops only for now.
            qubit = qextract(qreg, op.wires[0])
            qubit2 = qinst(op.name, 1, qubit)[0]
            qreg2 = qinsert(qreg, op.wires[0], qubit2)
        assert qreg2 is not None
        qreg = qreg2

    return [qreg]


def trace_quantum_function(
    f:Callable, args, kwargs,
    transform:Optional[Callable[[QuantumTape],QuantumTape]]=None) -> ClosedJaxpr:
    """ The tracing function supporting user-defined quantum tape transformation. The tracing is
    done in parts: 1) Classical tracing, producing classical JAX tracers and the nested quantum tape
    2) Calling user-defined tape transformations 3) Quantum tape tracing, producing the quantum JAX
    tracers. Finally, print the resulting jaxpr. """
    with main_tracing_context() as ctx:

        quantum_tape = QuantumTape()
        # print("1. Tracing classical part")
        with frame_tracing_context(ctx) as (frame, trace):

            wffa, in_avals = deduce_avals(f, args, kwargs)
            in_tracers = _input_type_to_tracers(trace.new_arg, in_avals)
            with QueuingManager.stop_recording(), quantum_tape:
                ans = wffa.call_wrapped(*in_tracers)
            out_classical_tracers = list(map(trace.full_raise, ans))

        # print("2. Presenting the full quantum tape to users")
        # print(f"\n{quantum_tape.operations=}\n")
        transformed_tape = transform(quantum_tape) if transform else quantum_tape

        # print("3. Tracing the quantum tape")
        with frame_tracing_context(ctx, frame, trace):
            ans2 = trace_quantum_tape(transformed_tape, ctx, frame, trace)
            out_quantum_tracers = list(map(trace.full_raise, ans2))
            jaxpr, out_type, consts = frame.to_jaxpr2(out_classical_tracers + out_quantum_tracers)

    closed_jaxpr = ClosedJaxpr(jaxpr, consts)
    return closed_jaxpr


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

