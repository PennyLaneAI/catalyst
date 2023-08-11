import jax
from jax._src.core import (ClosedJaxpr, MainTrace as JaxMainTrace, new_main, cur_sublevel, get_aval,
                           Tracer as JaxTracer)
from jax._src.interpreters.partial_eval import (DynamicJaxprTrace, JaxprStackFrame,
                                                trace_to_subjaxpr_dynamic2, extend_jaxpr_stack,
                                                _input_type_to_tracers)
from jax._src.source_info_util import reset_name_stack
from jax._src.lax.lax import _abstractify
from jax._src import linear_util as lu
from jax._src.tree_util import (tree_flatten, tree_unflatten)
from jax._src.api_util import (
    flatten_fun, apply_flat_fun, flatten_fun_nokwargs, flatten_fun_nokwargs2,
    argnums_partial, argnums_partial_except, flatten_axes, donation_vector,
    rebase_donate_argnums, _ensure_index, _ensure_index_tuple,
    shaped_abstractify, _ensure_str_tuple, apply_flat_fun_nokwargs,
    check_callable, debug_info, result_paths, flat_out_axes, debug_info_final)
from catalyst.jax_primitives import Qreg, AbstractQreg, qinst, qextract, qinsert
from typing import Optional, Callable, List
from pennylane import QueuingManager
from pennylane.operation import AnyWires, Operation, Wires
from pennylane.tape import QuantumTape


from dataclasses import dataclass


@dataclass
class ToplevelTracingContex:
    main: JaxMainTrace
    quantum_tape: QuantumTape

TRACING_CONTEXT : Optional[ToplevelTracingContex] = None


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

    def __init__(self, quantum_tape, out_classical_tracers, frame, trace, args, kwargs):
        self.quantum_tape = quantum_tape
        self.frame = frame
        self.trace = trace
        self.out_classical_tracers = out_classical_tracers
        # kwargs["wires"] = Wires(HybridOp.num_wires)
        super().__init__(wires = Wires(HybridOp.num_wires))

    def __repr__(self):
        """Constructor-call-like representation."""
        return f"{self.name}(tape={self.quantum_tape.operations})"


def hybrid(callee:Callable):
    """ A model frontend extension function. Simplified analog of for_loop, while_loop, adjoint,
    etc. """
    def _call(*args, **kwargs):
        global TRACING_CONTEXT
        ctx = TRACING_CONTEXT

        quantum_tape = QuantumTape()
        frame = JaxprStackFrame()
        with extend_jaxpr_stack(ctx.main, frame), reset_name_stack():
            trace = DynamicJaxprTrace(ctx.main, cur_sublevel())
            wffa, in_avals = deduce_avals(callee, args, kwargs)
            in_tracers = _input_type_to_tracers(trace.new_arg, in_avals)
            with QueuingManager.stop_recording(), quantum_tape:
                ans = wffa.call_wrapped(*in_tracers)
            out_classical_tracers = list(map(trace.full_raise, ans))
        #print(f"Nested tape: {quantum_tape=}")
        HybridOp(quantum_tape, out_classical_tracers, frame, trace, args, kwargs)
        return out_classical_tracers[0] # FIXME: generalize to the arbitrary number of retvals

    return _call

hybrid_p = jax.core.Primitive("hybrid")


@hybrid_p.def_abstract_eval
def _abstract_eval(*args, jaxpr_body):
    return AbstractQreg()


def trace_quantum_tape(quantum_tape:QuantumTape,
                       main:JaxMainTrace,
                       frame:JaxprStackFrame,
                       trace:DynamicJaxprTrace) -> List[JaxTracer]:
    with extend_jaxpr_stack(main, frame), reset_name_stack():
        qreg = _input_type_to_tracers(trace.new_arg, [AbstractQreg()])[0]
        for op in quantum_tape.operations:
            qreg2 = None
            if isinstance(op, HybridOp):
                # Handle nested quantum operation
                qreg_out = trace_quantum_tape(op.quantum_tape, main, op.frame, op.trace)[0]
                jaxpr, _, _ = op.frame.to_jaxpr2(op.out_classical_tracers + [qreg_out])
                #print(f"Nested jaxpr:\n{jaxpr=}")
                qreg2 = hybrid_p.bind(qreg, jaxpr_body=jaxpr)
            else:
                # We assume 1-qubit quantum operation for now
                qubit = qextract(qreg, op.wires[0])
                qubit2 = qinst(op.name, 1, qubit)[0]
                qreg2 = qinsert(qreg, op.wires[0], qubit2)
            assert qreg2 is not None
            qreg = qreg2

    return [qreg]


def trace_quantum_function(
    f:Callable, args, kwargs,
    transform:Optional[Callable[[QuantumTape],QuantumTape]]=None) -> ClosedJaxpr:
    """ JAX tracer supporting quantum functions and user-defined quantum tape transformation. """
    global TRACING_CONTEXT
    with new_main(DynamicJaxprTrace, dynamic=True) as main:
        main.jaxpr_stack = ()
        TRACING_CONTEXT = ctx = ToplevelTracingContex(main, QuantumTape())

        frame = JaxprStackFrame()
        with extend_jaxpr_stack(ctx.main, frame), reset_name_stack():
            trace = DynamicJaxprTrace(ctx.main, cur_sublevel())
            wffa, in_avals = deduce_avals(f, args, kwargs)
            in_tracers = _input_type_to_tracers(trace.new_arg, in_avals)
            with QueuingManager.stop_recording(), ctx.quantum_tape:
                ans = wffa.call_wrapped(*in_tracers)
            out_classical_tracers = list(map(trace.full_raise, ans))

        print(f"\n***\n{ctx.quantum_tape.operations=}\n***\n")
        transformed_tape = transform(ctx.quantum_tape) if transform else ctx.quantum_tape

        with extend_jaxpr_stack(ctx.main, frame), reset_name_stack():
            qreg_tracer = _input_type_to_tracers(trace.new_arg, [AbstractQreg()])[0]
            ans2 = trace_quantum_tape(transformed_tape, main, frame, trace)
            out_quantum_tracers = list(map(trace.full_raise, ans2))

    jaxpr, out_type, consts = frame.to_jaxpr2(out_classical_tracers + out_quantum_tracers)
    return jaxpr

