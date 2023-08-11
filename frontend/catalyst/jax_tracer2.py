import jax
from jax._src.core import (ClosedJaxpr, MainTrace as JaxMainTrace, new_main, cur_sublevel, get_aval,
                           Tracer as JaxTracer)
from jax._src.interpreters.partial_eval import (DynamicJaxprTrace, DynamicJaxprTracer,
                                                JaxprStackFrame, trace_to_subjaxpr_dynamic2,
                                                extend_jaxpr_stack, _input_type_to_tracers,
                                                new_jaxpr_eqn)
from jax._src.source_info_util import reset_name_stack, current as jax_current
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
from itertools import chain

from dataclasses import dataclass


@dataclass
class ToplevelTracingContex:
    main: JaxMainTrace
    trace: DynamicJaxprTrace
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

    def __init__(self, quantum_tape, in_classical_tracers, out_classical_tracers, result_classical_tracers, frame, trace, args, kwargs):
        self.quantum_tape = quantum_tape
        self.frame = frame
        self.trace = trace
        self.in_classical_tracers = in_classical_tracers
        self.out_classical_tracers = out_classical_tracers
        self.result_classical_tracers = result_classical_tracers
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
        parent_trace = ctx.trace

        quantum_tape = QuantumTape()
        frame = JaxprStackFrame()
        with extend_jaxpr_stack(ctx.main, frame), reset_name_stack():
            trace = DynamicJaxprTrace(ctx.main, cur_sublevel())
            ctx.trace = trace
            wffa, in_avals = deduce_avals(callee, args, kwargs)
            in_tracers = _input_type_to_tracers(trace.new_arg, in_avals)
            with QueuingManager.stop_recording(), quantum_tape:
                res_tracers = wffa.call_wrapped(*in_tracers)

        res_avals = list(map(shaped_abstractify, res_tracers))
        out_classical_tracers = [DynamicJaxprTracer(trace, a, jax_current()) for a in res_avals]
        for t, aval in zip(out_classical_tracers, res_avals):
            trace.frame.tracers.append(t)
            trace.frame.tracer_to_var[id(t)] = trace.frame.newvar(aval)

        HybridOp(quantum_tape, args, out_classical_tracers, res_tracers, frame, trace, args, kwargs)

        ctx.trace = parent_trace
        return out_classical_tracers[0] # FIXME: generalize to the arbitrary number of retvals

    return _call

hybrid_p = jax.core.Primitive("hybrid")
hybrid_p.multiple_results = True


@hybrid_p.def_abstract_eval
def _abstract_eval(*args, jaxpr_body):
    # HACK: At the bind time we already know classical output tracers so we will add the variables
    # to the primitive explicitly. Here we only return quantum value to make quantum output tracer
    # appear.
    return [AbstractQreg()]


def trace_quantum_tape(quantum_tape:QuantumTape,
                       main:JaxMainTrace,
                       frame:JaxprStackFrame,
                       trace:DynamicJaxprTrace) -> List[JaxTracer]:
    """ Recursively trace the nested `quantum_tape` and produce the quantum tracers. With quantum
    tracers we can complete the set of tracers and finally emit the JAXPR of the whole quantum
    program. """
    qreg = _input_type_to_tracers(trace.new_arg, [AbstractQreg()])[0]
    for op in quantum_tape.operations:
        qreg2 = None
        if isinstance(op, HybridOp):
            # Handle nested quantum operation
            with extend_jaxpr_stack(main, op.frame), reset_name_stack():
                qreg_out = trace_quantum_tape(op.quantum_tape, main, op.frame, op.trace)[0]
                jaxpr, typ, consts = op.frame.to_jaxpr2([qreg_out] + op.result_classical_tracers)

            qreg2 = hybrid_p.bind(qreg, *chain(op.in_classical_tracers, consts), jaxpr_body=jaxpr)[0]
            # HACK: we add variables for already existing tracers
            frame.eqns[-1].outvars.append(trace.getvar(op.out_classical_tracers[0]))
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
    global TRACING_CONTEXT
    with new_main(DynamicJaxprTrace, dynamic=True) as main:
        main.jaxpr_stack = ()

        print("1. Tracing classical part")
        frame = JaxprStackFrame()
        with extend_jaxpr_stack(main, frame), reset_name_stack():
            trace = DynamicJaxprTrace(main, cur_sublevel())
            TRACING_CONTEXT = ctx = ToplevelTracingContex(main, trace, QuantumTape())

            wffa, in_avals = deduce_avals(f, args, kwargs)
            in_tracers = _input_type_to_tracers(trace.new_arg, in_avals)
            with QueuingManager.stop_recording(), ctx.quantum_tape:
                ans = wffa.call_wrapped(*in_tracers)
            out_classical_tracers = list(map(trace.full_raise, ans))

        print("2. Presenting the full quantum tape to users")
        print(f"\n{ctx.quantum_tape.operations=}\n")
        transformed_tape = transform(ctx.quantum_tape) if transform else ctx.quantum_tape

        print("3. Tracing the quantum tape")
        with extend_jaxpr_stack(ctx.main, frame), reset_name_stack():
            ans2 = trace_quantum_tape(transformed_tape, main, frame, trace)
            out_quantum_tracers = list(map(trace.full_raise, ans2))

            jaxpr, out_type, consts = frame.to_jaxpr2(out_classical_tracers + out_quantum_tracers)
    return jaxpr

