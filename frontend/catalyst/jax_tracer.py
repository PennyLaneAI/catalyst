# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains functions tracing and lowering JAX code to MLIR.
"""

import jax
import pennylane as qml
from jax._src import source_info_util
from jax._src.dispatch import jaxpr_replicas
from jax._src.interpreters.mlir import _module_name_regex
from jax._src.lax.lax import xb, xla
from jax._src.util import wrap_name
from jax.interpreters.mlir import (
    AxisContext,
    ModuleContext,
    ReplicaAxisContext,
    ir,
    lower_jaxpr_to_fun,
    lowerable_effects,
)
from jax.interpreters.partial_eval import DynamicJaxprTracer
from jax.tree_util import tree_flatten, tree_structure, tree_unflatten
from pennylane.measurements import CountsMP, MeasurementProcess
from pennylane.operation import Wires

import catalyst.jax_primitives as jprim
# from catalyst.jax_tape import JaxTape
from catalyst.utils.tracing import TracingContext



from functools import update_wrapper

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
    qdevice_p,
    qextract,
    qextract_p,
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
# from catalyst.jax_tracer import (
#     KNOWN_NAMED_OBS,
# )
from catalyst.utils.exceptions import CompileError
from catalyst.utils.jax_extras import (
    initial_style_jaxprs_with_common_consts1,
    initial_style_jaxprs_with_common_consts2,
    new_main2,
    sort_eqns,
)
from catalyst.utils.tracing import TracingContext


from pennylane.measurements import MidMeasureMP
from catalyst.utils.patching import Patcher


from jax.linear_util import wrap_init




class Function:
    """An object that represents a compiled function.

    At the moment, it is only used to compute sensible names for higher order derivative
    functions in MLIR.

    Args:
        fn (Callable): the function boundary.

    Raises:
        AssertionError: Invalid function type.
    """

    def __init__(self, fn):
        self.fn = fn
        self.__name__ = fn.__name__

    def __call__(self, *args, **kwargs):
        jaxpr, shape = jax.make_jaxpr(self.fn, return_shape=True)(*args)
        shape_tree = tree_structure(shape)

        def _eval_jaxpr(*args):
            return jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)

        retval = jprim.func_p.bind(wrap_init(_eval_jaxpr), *args, fn=self)
        return tree_unflatten(shape_tree, retval)



KNOWN_NAMED_OBS = (qml.Identity, qml.PauliX, qml.PauliY, qml.PauliZ, qml.Hadamard)

FORCED_ORDER_PRIMITIVES = {qdevice_p, qextract_p}

@dataclass
class MainTracingContex:
    main: JaxMainTrace
    frames: Dict[DynamicJaxprTrace, JaxprStackFrame]
    mains: Dict[DynamicJaxprTrace, JaxMainTrace]
    trace: Optional[DynamicJaxprTrace]

    def __init__(self, main:JaxMainTrace):
        self.main, self.frames, self.mains, self.trace = main, {}, {}, None


TRACING_CONTEXT: Optional[MainTracingContex] = None


@contextmanager
def main_tracing_context() -> ContextManager[MainTracingContex]:
    global TRACING_CONTEXT
    with new_base_main(DynamicJaxprTrace, dynamic=True) as main:
        main.jaxpr_stack = ()
        TRACING_CONTEXT = ctx = MainTracingContex(main)
        try:
            yield ctx
        finally:
            TRACING_CONTEXT = None


def get_main_tracing_context(hint=None) -> MainTracingContex:
    """Checks a number of tracing conditions and return the MainTracingContex"""
    msg = f"{hint or 'catalyst functions'} can only be used from within @qjit decorated code."
    TracingContext.check_is_tracing(msg)
    if TRACING_CONTEXT is None:
        raise CompileError(f"{hint} can only be used from within a qml.qnode.")
    return TRACING_CONTEXT


@dataclass
class QRegPromise:
    base: DynamicJaxprTracer
    cache: Dict[int, DynamicJaxprTracer]

    def __init__(self, base):
        self.base = base
        self.cache = {}


def promise_qextract(qrp: QRegPromise,
                     wires:List[Any]) -> List[DynamicJaxprTracer]:
    cached_tracers = set([w for w in qrp.cache.keys() if not isinstance(w, int)])
    requested_tracers = set([w for w in wires if not isinstance(w, int)])
    if cached_tracers != requested_tracers:
        promise_actualize(qrp)
    qubits = []
    for w in wires:
        if w in qrp.cache:
            qubit = qrp.cache[w]
            assert qubit is not None, \
                f"Attempting to extract wire {w} from register {qrp.base} for the second time"
            qubits.append(qubit)
            qrp.cache[w] = None
        else:
            qubits.append(qextract(qrp.base, w))
    return qubits


def promise_qinsert(qrp: QRegPromise,
                    wires,
                    qubits) -> None:
    assert len(wires)==len(qubits)
    for w, qubit in zip(wires, qubits):
        assert (w not in qrp.cache) or (qrp.cache[w] is None), \
            f"Attempting to insert an already-inserted wire {w} into {qrp.base}"
        qrp.cache[w] = qubit


def promise_actualize(qrp: QRegPromise) -> DynamicJaxprTracer:
    qreg = qrp.base
    for w,qubit in qrp.cache.items():
        qreg = qinsert(qreg, w, qubit)
    qrp.cache = {}
    qrp.base = qreg
    return qreg


@contextmanager
def frame_tracing_context(
    ctx: MainTracingContex, trace: Optional[DynamicJaxprTrace] = None
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


def trace_to_mlir(func, *args, **kwargs):
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

    # The compilation cache must be clear for each translation unit.
    # Otherwise, MLIR functions which do not exist in the current translation unit will be assumed
    # to exist if an equivalent python function is seen in the cache. This happens during testing or
    # if we wanted to compile a single python function multiple times with different options.
    jprim.mlir_fn_cache.clear()

    with TracingContext():
        jaxpr, shape = jax.make_jaxpr(func, return_shape=True)(*args, **kwargs)

    nrep = jaxpr_replicas(jaxpr)
    effects = [eff for eff in jaxpr.effects if eff in jax.core.ordered_effects]
    axis_context = ReplicaAxisContext(xla.AxisEnv(nrep, (), ()))
    name_stack = source_info_util.new_name_stack(wrap_name("ok", "jit"))
    module, context = custom_lower_jaxpr_to_module(
        func_name="jit_" + func.__name__,
        module_name=func.__name__,
        jaxpr=jaxpr,
        effects=effects,
        platform="cpu",
        axis_context=axis_context,
        name_stack=name_stack,
        donated_args=[],
    )

    return module, context, jaxpr, tree_structure(shape)


def trace_quantum_tape(
    quantum_tape: QuantumTape,
    device: QubitDevice,
    qreg: DynamicJaxprTracer,
    ctx: MainTracingContex,
    trace: DynamicJaxprTrace,
) -> DynamicJaxprTracer:
    """Recursively trace the nested `quantum_tape` and produce the quantum tracers. With quantum
    tracers we can complete the set of tracers and finally emit the JAXPR of the whole quantum
    program."""
    # Notes:
    # [1] - We add alread existing classical tracers into the last JAX equation.
    # [2] - We are interested in a new quantum tracer, so we ignore all others.

    def bind_overwrite_classical_tracers(op: HybridOp, binder, *args, **kwargs):
        """Binds the primitive `prim` but override the returned classical tracers with the already
        existing output tracers of the operation `op`."""
        out_quantum_tracer = binder(*args, **kwargs)[-1]
        eqn = ctx.frames[trace].eqns[-1]
        assert (len(eqn.outvars) - 1) == len(op.out_classical_tracers)
        for i, t in zip(range(len(eqn.outvars) - 1), op.out_classical_tracers):  # [1]
            eqn.outvars[i] = trace.getvar(t)
        return out_quantum_tracer # [2]

    qrp = QRegPromise(qreg)
    for op in device.expand_fn(quantum_tape):
        qreg = None
        qrp2 = None
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
                qreg = promise_actualize(qrp)
                qrp2 = QRegPromise(bind_overwrite_classical_tracers(
                    op,
                    qfor_p.bind,
                    op.in_classical_tracers[0],
                    op.in_classical_tracers[1],
                    step,
                    *(consts + op.in_classical_tracers[3:] + [qreg]),
                    body_jaxpr=ClosedJaxpr(convert_constvars_jaxpr(jaxpr), ()),
                    body_nconsts=len(consts),
                    apply_reverse_transform=apply_reverse_transform,
                ))

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

                qreg = promise_actualize(qrp)
                qrp2 = QRegPromise(bind_overwrite_classical_tracers(
                    op,
                    qcond_p.bind,
                    *(op.in_classical_tracers + combined_consts + [qreg]),
                    branch_jaxprs=jaxprs2,
                ))

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

                qreg = promise_actualize(qrp)
                qrp2 = QRegPromise(bind_overwrite_classical_tracers(
                    op,
                    qwhile_p.bind,
                    *(cond_consts + body_consts + op.in_classical_tracers + [qreg]),
                    cond_jaxpr=ClosedJaxpr(convert_constvars_jaxpr(cond_jaxpr), ()),
                    body_jaxpr=ClosedJaxpr(convert_constvars_jaxpr(body_jaxpr), ()),
                    cond_nconsts=len(cond_consts),
                    body_nconsts=len(body_consts),
                ))

            elif isinstance(op, MidCircuitMeasure):
                wire = op.in_classical_tracers[0]
                qubit = promise_qextract(qrp, [wire])[0]
                qubit2 = bind_overwrite_classical_tracers(op, qmeasure_p.bind, qubit)
                promise_qinsert(qrp, [wire], [qubit2])
                qrp2 = qrp

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

                qreg = promise_actualize(qrp)
                args, args_tree = tree_flatten((body_consts, op.in_classical_tracers, [qreg]))
                op_results = adjoint_p.bind(
                    *args,
                    args_tree=args_tree,
                    jaxpr=ClosedJaxpr(convert_constvars_jaxpr(body_jaxpr), ()),
                )
                qrp2 = QRegPromise(op_results[0])
            else:
                raise NotImplementedError(f"{op=}")
        else:
            if isinstance(op, MeasurementProcess):
                qrp2 = qrp
            else:
                qubits = promise_qextract(qrp, op.wires)
                if isinstance(op, QubitUnitary):
                    qubits2 = qunitary(*[*op.parameters, *qubits])
                else:
                    qubits2 = qinst(op.name, len(qubits), *qubits, *op.parameters)
                promise_qinsert(qrp, op.wires, qubits2)
                qrp2 = qrp

        assert qrp2 is not None
        qrp = qrp2

    qreg = promise_actualize(qrp)
    ctx.frames[trace].eqns = sort_eqns(ctx.frames[trace].eqns, FORCED_ORDER_PRIMITIVES)
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


def trace_quantum_measurements(
    quantum_tape,
    device: QubitDevice,
    qreg: DynamicJaxprTracer,
    ctx: MainTracingContex,
    trace: DynamicJaxprTrace,
    outputs: List[Union[MeasurementProcess, DynamicJaxprTracer, Any]],
    out_tree,
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
    quantum algorithm.
    The tracing is done in parts as follows: (1) Classical tracing, classical JAX tracers and the
    quantum tape are produced (2) Quantum tape tracing, the remaining quantum JAX tracers and the
    final JAXPR are produced.
    Tape transformations could be applied in-between, allowing users to modify the algorithm
    before the final jaxpr is created."""

    with main_tracing_context() as ctx:
        # (1) - Classical tracing
        quantum_tape = QuantumTape()
        with frame_tracing_context(ctx) as trace:
            wffa, in_avals, out_tree_promise = deduce_avals(f, args, kwargs)
            in_classical_tracers = _input_type_to_tracers(trace.new_arg, in_avals)
            with QueuingManager.stop_recording(), quantum_tape:
                # Quantum tape transformations happen at the end of tracing
                ans = wffa.call_wrapped(*in_classical_tracers)
            out_classical_tracers_or_measurements = [
                (trace.full_raise(t) if isinstance(t, DynamicJaxprTracer) else t) for t in ans
            ]

        # (2) - Quantum tracing
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


# pylint: disable=too-many-arguments
def custom_lower_jaxpr_to_module(
    func_name: str,
    module_name: str,
    jaxpr: jax.core.ClosedJaxpr,
    effects,
    platform: str,
    axis_context: AxisContext,
    name_stack,
    donated_args,
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
    platform = xb.canonicalize_platform(platform)
    if not xb.is_known_platform(platform):
        raise ValueError(f"Unknown platform {platform}")
    in_avals = jaxpr.in_avals
    assert arg_shardings is None
    assert result_shardings is None
    platforms_with_donation = ("cuda", "rocm", "tpu")
    assert platform not in platforms_with_donation
    if any(eff not in lowerable_effects for eff in jaxpr.effects):
        raise ValueError(f"Cannot lower jaxpr with effects: {jaxpr.effects}")
    if any(donated_args):
        unused_donations = [str(a) for a, d in zip(in_avals, donated_args) if d]
        msg = "See an explanation at https://jax.readthedocs.io/en/latest/faq.html#buffer-donation."
        if platform not in platforms_with_donation:
            msg = f"Donation is not implemented for {platform}.\n{msg}"

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
        unlowerable_effects = {eff for eff in jaxpr.effects if eff not in lowerable_effects}
        if unlowerable_effects:
            raise ValueError(f"Cannot lower jaxpr with unlowerable effects: {unlowerable_effects}")
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



class QFunc:
    """A device specific quantum function.

    Args:
        qfunc (Callable): the quantum function
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values
        device (a derived class from QubitDevice): a device specification which determines
            the valid gate set for the quantum function
    """

    # The set of supported devices at runtime
    RUNTIME_DEVICES = (
        "lightning.qubit",
        "lightning.kokkos",
        "braket.aws.qubit",
        "braket.local.qubit",
    )

    def __init__(self, fn, device):
        self.func = fn
        self.device = device
        update_wrapper(self, fn)

    def __call__(self, *args, **kwargs):
        if isinstance(self, qml.QNode):
            if self.device.short_name not in QFunc.RUNTIME_DEVICES:
                raise CompileError(
                    f"The {self.device.short_name} device is not "
                    "supported for compilation at the moment."
                )

            backend_kwargs = {}
            if hasattr(self.device, "shots"):
                backend_kwargs["shots"] = self.device.shots if self.device.shots else 0
            if self.device.short_name == "braket.local.qubit":  # pragma: no cover
                backend_kwargs["backend"] = self.device._device._delegate.DEVICE_ID
            elif self.device.short_name == "braket.aws.qubit":  # pragma: no cover
                backend_kwargs["device_arn"] = self.device._device._arn
                if self.device._s3_folder:
                    backend_kwargs["s3_destination_folder"] = str(self.device._s3_folder)

            device = QJITDevice(
                self.device.shots, self.device.wires, self.device.short_name, backend_kwargs
            )
        else:
            # Allow QFunc to still be used by itself for internal testing.
            device = self.device

        jaxpr, shape = trace_quantum_function(self.func, device, args, kwargs)

        retval_tree = tree_structure(shape)

        def _eval_jaxpr(*args):
            return jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)

        args_data, _ = tree_flatten(args)

        wrapped = wrap_init(_eval_jaxpr)
        retval = jprim.func_p.bind(wrapped, *args_data, fn=self)

        return tree_unflatten(retval_tree, retval)



class QJITDevice(qml.QubitDevice):
    """QJIT device.

    A device that interfaces the compilation pipeline of Pennylane programs.

    Args:
        wires (int): the number of wires to initialize the device with
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified. Setting
            to ``None`` results in computing statistics like expectation values and
            variances analytically
        backend_name (str): name of the device from the list of supported and compiled backend
            devices by the runtime
        backend_kwargs (Dict(str, AnyType)): An optional dictionary of the device specifications
    """

    name = "QJIT device"
    short_name = "qjit.device"
    pennylane_requires = "0.1.0"
    version = "0.0.1"
    author = ""
    operations = [
        "MidCircuitMeasure",
        "Cond",
        "WhileLoop",
        "ForLoop",
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "Identity",
        "S",
        "T",
        "PhaseShift",
        "RX",
        "RY",
        "RZ",
        "CNOT",
        "CY",
        "CZ",
        "SWAP",
        "IsingXX",
        "IsingYY",
        "IsingXY",
        "IsingZZ",
        "ControlledPhaseShift",
        "CRX",
        "CRY",
        "CRZ",
        "CRot",
        "CSWAP",
        "MultiRZ",
        "QubitUnitary",
        "Adjoint",
    ]
    observables = [
        "Identity",
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "Hermitian",
        "Hamiltonian",
    ]

    def __init__(self, shots=None, wires=None, backend_name=None, backend_kwargs=None):
        self.backend_name = backend_name if backend_name else "default"
        self.backend_kwargs = backend_kwargs if backend_kwargs else {}
        super().__init__(wires=wires, shots=shots)

    def apply(self, operations, **kwargs):
        """
        Raises: RuntimeError
        """
        raise RuntimeError("QJIT devices cannot apply operations.")

    def default_expand_fn(self, circuit, max_expansion=10):
        """
        Most decomposition logic will be equivalent to PennyLane's decomposition.
        However, decomposition logic will differ in the following cases:

        1. All :class:`qml.QubitUnitary <pennylane.ops.op_math.Controlled>` operations
            will decompose to :class:`qml.QubitUnitary <pennylane.QubitUnitary>` operations.
        2. :class:`qml.ControlledQubitUnitary <pennylane.ControlledQubitUnitary>` operations
            will decompose to :class:`qml.QubitUnitary <pennylane.QubitUnitary>` operations.
        3. The list of device-supported gates employed by Catalyst is currently different than
            that of the ``lightning.qubit`` device, as defined by the
            :class:`~.pennylane_extensions.QJITDevice`.

        Args:
            circuit: circuit to expand
            max_expansion: the maximum number of expansion steps if no fixed-point is reached.
        """
        # Ensure catalyst.measure is used instead of qml.measure.
        if any(isinstance(op, MidMeasureMP) for op in circuit.operations):
            raise CompileError("Must use 'measure' from Catalyst instead of PennyLane.")

        # Fallback for controlled gates that won't decompose successfully.
        # Doing so before rather than after decomposition is generally a trade-off. For low
        # numbers of qubits, a unitary gate might be faster, while for large qubit numbers prior
        # decomposition is generally faster.
        # At the moment, bypassing decomposition for controlled gates will generally have a higher
        # success rate, as complex decomposition paths can fail to trace (c.f. PL #3521, #3522).

        def _decomp_controlled_unitary(self, *_args, **_kwargs):
            return qml.QubitUnitary(qml.matrix(self), wires=self.wires)

        def _decomp_controlled(self, *_args, **_kwargs):
            return qml.QubitUnitary(qml.matrix(self), wires=self.wires)

        with Patcher(
            (qml.ops.ControlledQubitUnitary, "compute_decomposition", _decomp_controlled_unitary),
            (qml.ops.Controlled, "has_decomposition", lambda self: True),
            (qml.ops.Controlled, "decomposition", _decomp_controlled),
        ):
            expanded_tape = super().default_expand_fn(circuit, max_expansion)

        self.check_validity(expanded_tape.operations, [])
        return expanded_tape

