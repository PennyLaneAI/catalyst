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

from contextlib import contextmanager
from dataclasses import dataclass
from functools import update_wrapper
from typing import Any, Callable, ContextManager, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import pennylane as qml
from jax._src.api import ShapeDtypeStruct
from jax._src.core import ClosedJaxpr, JaxprEqn, ShapedArray
from jax._src.interpreters.partial_eval import (
    DynamicJaxprTrace,
    DynamicJaxprTracer,
    _input_type_to_tracers,
    convert_constvars_jaxpr,
)
from jax._src.linear_util import wrap_init
from jax._src.source_info_util import current as jax_current
from jax._src.util import unzip2
from pennylane import QubitDevice, QubitUnitary, QueuingManager
from pennylane.measurements import MeasurementProcess, MidMeasureMP
from pennylane.operation import AnyWires, Operation, Wires
from pennylane.tape import QuantumTape

from catalyst.jax_primitives import (
    AbstractQreg,
    adjoint_p,
    compbasis_p,
    counts_p,
    expval_p,
    func_p,
    hamiltonian,
    hermitian,
    mlir_fn_cache,
    namedobs,
    probs_p,
    qalloc_p,
    qcond_p,
    qdealloc_p,
    qdevice_p,
    qextract_p,
    qfor_p,
    qinsert_p,
    qinst_p,
    qmeasure_p,
    qunitary_p,
    qwhile_p,
    sample_p,
    state_p,
    tensorobs,
    tensorobs_p,
    var_p,
)
from catalyst.utils.exceptions import CompileError
from catalyst.utils.jax_extras import (
    JaxprPrimitive,
    PyTreeDef,
    deduce_avals,
    initial_style_jaxprs_with_common_consts1,
    initial_style_jaxprs_with_common_consts2,
    jaxpr_to_mlir,
    sort_eqns,
    tree_flatten,
    tree_structure,
    tree_unflatten,
    treedef_is_leaf,
)
from catalyst.utils.patching import Patcher
from catalyst.utils.tracing import EvaluationContext, EvaluationMode, JaxTracingContext


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

        retval = func_p.bind(wrap_init(_eval_jaxpr), *args, fn=self)
        return tree_unflatten(shape_tree, retval)


KNOWN_NAMED_OBS = (qml.Identity, qml.PauliX, qml.PauliY, qml.PauliZ, qml.Hadamard)

FORCED_ORDER_PRIMITIVES = {qdevice_p, qextract_p}

PAULI_NAMED_MAP = {
    "I": "Identity",
    "X": "PauliX",
    "Y": "PauliY",
    "Z": "PauliZ",
}


@dataclass
class QRegPromise:
    base: DynamicJaxprTracer
    cache: Dict[int, DynamicJaxprTracer]

    def __init__(self, base):
        self.base = base
        self.cache = {}


def qrp_extract(qrp: QRegPromise, wires: List[Any], allow_reuse=False) -> List[DynamicJaxprTracer]:
    cached_tracers = set([w for w in qrp.cache.keys() if not isinstance(w, int)])
    requested_tracers = set([w for w in wires if not isinstance(w, int)])
    if cached_tracers != requested_tracers:
        qrp_actualize(qrp)
    qubits = []
    for w in wires:
        if w in qrp.cache:
            qubit = qrp.cache[w]
            assert (
                qubit is not None
            ), f"Attempting to extract wire {w} from register {qrp.base} for the second time"
            qubits.append(qubit)
            if not allow_reuse:
                qrp.cache[w] = None
        else:
            qubits.append(qextract_p.bind(qrp.base, w))
    return qubits


def qrp_insert(qrp: QRegPromise, wires, qubits) -> None:
    assert len(wires) == len(qubits)
    for w, qubit in zip(wires, qubits):
        assert (w not in qrp.cache) or (
            qrp.cache[w] is None
        ), f"Attempting to insert an already-inserted wire {w} into {qrp.base}"
        qrp.cache[w] = qubit


def qrp_actualize(qrp: QRegPromise) -> DynamicJaxprTracer:
    qreg = qrp.base
    for w, qubit in qrp.cache.items():
        qreg = qinsert_p.bind(qreg, w, qubit)
    qrp.cache = {}
    qrp.base = qreg
    return qreg


def new_inner_tracer(trace: DynamicJaxprTrace, aval) -> DynamicJaxprTracer:
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
    binder: JaxprPrimitive = None

    def __init__(self, in_classical_tracers, out_classical_tracers, regions: List[HybridOpRegion]):
        self.in_classical_tracers = in_classical_tracers
        self.out_classical_tracers = out_classical_tracers
        self.regions = regions
        super().__init__(wires=Wires(HybridOp.num_wires))

    def __repr__(self):
        """Constructor-call-like representation."""
        return f"{self.name}(tapes={[r.quantum_tape.operations for r in self.regions]})"

    def bind_overwrite_classical_tracers(
        self, ctx: JaxTracingContext, trace: DynamicJaxprTrace, *args, **kwargs
    ) -> DynamicJaxprTracer:
        """Binds the JAX primitive but override the returned classical tracers with the already
        existing output tracers, stored in the operations."""
        # Notes:
        # [1] - We add alread existing classical tracers into the last JAX equation.
        # [2] - We are interested in a new quantum tracer, so we ignore all others.
        assert self.binder is not None, "HybridOp should set a binder"
        out_quantum_tracer = self.binder(*args, **kwargs)[-1]
        eqn = ctx.frames[trace].eqns[-1]
        assert (len(eqn.outvars) - 1) == len(self.out_classical_tracers)
        for i, t in zip(range(len(eqn.outvars) - 1), self.out_classical_tracers):  # [1]
            eqn.outvars[i] = trace.getvar(t)
        return out_quantum_tracer  # [2]

    def trace(
        self,
        ctx: JaxTracingContext,
        device: QubitDevice,
        trace: DynamicJaxprTrace,
        qrp: QRegPromise,
    ) -> QRegPromise:
        raise NotImplementedError("HybridOp should implement trace")


def has_nested_tapes(op: Operation) -> bool:
    return (
        isinstance(op, HybridOp)
        and len(op.regions) > 0
        and any([r.quantum_tape is not None for r in op.regions])
    )


class ForLoop(HybridOp):
    binder = qfor_p.bind

    def trace(self, ctx, device, trace, qrp) -> QRegPromise:
        op = self
        inner_trace = op.regions[0].trace
        inner_tape = op.regions[0].quantum_tape
        res_classical_tracers = op.regions[0].res_classical_tracers

        with EvaluationContext.frame_tracing_context(ctx, inner_trace):
            qreg_in = _input_type_to_tracers(inner_trace.new_arg, [AbstractQreg()])[0]
            qrp_out = trace_quantum_tape(inner_tape, device, qreg_in, ctx, inner_trace)
            qreg_out = qrp_actualize(qrp_out)
            jaxpr, typ, consts = ctx.frames[inner_trace].to_jaxpr2(
                res_classical_tracers + [qreg_out]
            )

        step = op.in_classical_tracers[2]
        apply_reverse_transform = isinstance(step, int) and step < 0
        qreg = qrp_actualize(qrp)
        qrp2 = QRegPromise(
            op.bind_overwrite_classical_tracers(
                ctx,
                trace,
                op.in_classical_tracers[0],
                op.in_classical_tracers[1],
                step,
                *(consts + op.in_classical_tracers[3:] + [qreg]),
                body_jaxpr=ClosedJaxpr(convert_constvars_jaxpr(jaxpr), ()),
                body_nconsts=len(consts),
                apply_reverse_transform=apply_reverse_transform,
            )
        )
        return qrp2


class MidCircuitMeasure(HybridOp):
    binder = qmeasure_p.bind

    def trace(self, ctx, device, trace, qrp) -> QRegPromise:
        op = self
        wire = op.in_classical_tracers[0]
        qubit = qrp_extract(qrp, [wire])[0]
        qubit2 = op.bind_overwrite_classical_tracers(ctx, trace, qubit)
        qrp_insert(qrp, [wire], [qubit2])
        return qrp


class Cond(HybridOp):
    binder = qcond_p.bind

    def trace(self, ctx, device, trace, qrp) -> QRegPromise:
        jaxprs, consts = [], []
        op = self
        for region in op.regions:
            with EvaluationContext.frame_tracing_context(ctx, region.trace):
                qreg_in = _input_type_to_tracers(region.trace.new_arg, [AbstractQreg()])[0]
                qrp_out = trace_quantum_tape(
                    region.quantum_tape, device, qreg_in, ctx, region.trace
                )
                qreg_out = qrp_actualize(qrp_out)
                jaxpr, typ, const = ctx.frames[region.trace].to_jaxpr2(
                    region.res_classical_tracers + [qreg_out]
                )
                jaxprs.append(jaxpr)
                consts.append(const)

        jaxprs2, combined_consts = initial_style_jaxprs_with_common_consts2(jaxprs, consts)

        qreg = qrp_actualize(qrp)
        qrp2 = QRegPromise(
            op.bind_overwrite_classical_tracers(
                ctx,
                trace,
                *(op.in_classical_tracers + combined_consts + [qreg]),
                branch_jaxprs=jaxprs2,
            )
        )
        return qrp2


class WhileLoop(HybridOp):
    binder = qwhile_p.bind

    def trace(self, ctx, device, trace, qrp) -> QRegPromise:
        cond_trace = self.regions[0].trace
        res_classical_tracers = self.regions[0].res_classical_tracers
        with EvaluationContext.frame_tracing_context(ctx, cond_trace):
            _input_type_to_tracers(cond_trace.new_arg, [AbstractQreg()])[0]
            cond_jaxpr, _, cond_consts = ctx.frames[cond_trace].to_jaxpr2(res_classical_tracers)

        body_trace = self.regions[1].trace
        body_tape = self.regions[1].quantum_tape
        res_classical_tracers = self.regions[1].res_classical_tracers
        with EvaluationContext.frame_tracing_context(ctx, body_trace):
            qreg_in = _input_type_to_tracers(body_trace.new_arg, [AbstractQreg()])[0]
            qrp_out = trace_quantum_tape(body_tape, device, qreg_in, ctx, body_trace)
            qreg_out = qrp_actualize(qrp_out)
            body_jaxpr, _, body_consts = ctx.frames[body_trace].to_jaxpr2(
                res_classical_tracers + [qreg_out]
            )

        qreg = qrp_actualize(qrp)
        qrp2 = QRegPromise(
            self.bind_overwrite_classical_tracers(
                ctx,
                trace,
                *(cond_consts + body_consts + self.in_classical_tracers + [qreg]),
                cond_jaxpr=ClosedJaxpr(convert_constvars_jaxpr(cond_jaxpr), ()),
                body_jaxpr=ClosedJaxpr(convert_constvars_jaxpr(body_jaxpr), ()),
                cond_nconsts=len(cond_consts),
                body_nconsts=len(body_consts),
            )
        )
        return qrp2


class Adjoint(HybridOp):
    binder = adjoint_p.bind

    def trace(self, ctx, device, trace, qrp) -> QRegPromise:
        op = self
        body_trace = op.regions[0].trace
        body_tape = op.regions[0].quantum_tape
        res_classical_tracers = op.regions[0].res_classical_tracers
        with EvaluationContext.frame_tracing_context(ctx, body_trace):
            qreg_in = _input_type_to_tracers(body_trace.new_arg, [AbstractQreg()])[0]
            qrp_out = trace_quantum_tape(body_tape, device, qreg_in, ctx, body_trace)
            qreg_out = qrp_actualize(qrp_out)
            body_jaxpr, _, body_consts = ctx.frames[body_trace].to_jaxpr2(
                res_classical_tracers + [qreg_out]
            )

        qreg = qrp_actualize(qrp)
        args, args_tree = tree_flatten((body_consts, op.in_classical_tracers, [qreg]))
        op_results = adjoint_p.bind(
            *args,
            args_tree=args_tree,
            jaxpr=ClosedJaxpr(convert_constvars_jaxpr(body_jaxpr), ()),
        )
        qrp2 = QRegPromise(op_results[0])
        return qrp2


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
    mlir_fn_cache.clear()

    with EvaluationContext(EvaluationMode.CLASSICAL_COMPILATION):
        jaxpr, shape = jax.make_jaxpr(func, return_shape=True)(*args, **kwargs)

    return jaxpr_to_mlir(func.__name__, jaxpr, shape)


def trace_quantum_tape(
    quantum_tape: QuantumTape,
    device: QubitDevice,
    qreg: DynamicJaxprTracer,
    ctx: JaxTracingContext,
    trace: DynamicJaxprTrace,
) -> QRegPromise:
    """Recursively trace the nested `quantum_tape` and produce the quantum tracers. With quantum
    tracers we can complete the set of tracers and finally emit the JAXPR of the whole quantum
    program."""

    qrp = QRegPromise(qreg)
    for op in device.expand_fn(quantum_tape):
        qrp2 = None
        if isinstance(op, HybridOp):
            qrp2 = op.trace(ctx, device, trace, qrp)
        else:
            if isinstance(op, MeasurementProcess):
                qrp2 = qrp
            else:
                qubits = qrp_extract(qrp, op.wires)
                if isinstance(op, QubitUnitary):
                    qubits2 = qunitary_p.bind(*[*op.parameters, *qubits])
                else:
                    qubits2 = qinst_p.bind(
                        *qubits, *op.parameters, op=op.name, qubits_len=len(qubits)
                    )
                qrp_insert(qrp, op.wires, qubits2)
                qrp2 = qrp

        assert qrp2 is not None
        qrp = qrp2

    # qreg = qrp_actualize(qrp)
    ctx.frames[trace].eqns = sort_eqns(ctx.frames[trace].eqns, FORCED_ORDER_PRIMITIVES)
    return qrp


def trace_observables(
    obs: Operation, device, qrp: QRegPromise, m_wires
) -> Tuple[List[DynamicJaxprTracer], Optional[int]]:
    wires = obs.wires if (obs and len(obs.wires) > 0) else m_wires
    # qubits = [qextract_p.bind(qreg, w) for w in wires]
    qubits = None
    if obs is None:
        qubits = qrp_extract(qrp, wires, allow_reuse=True)
        obs_tracers = compbasis_p.bind(*qubits)
    elif isinstance(obs, KNOWN_NAMED_OBS):
        qubits = qrp_extract(qrp, wires, allow_reuse=True)
        obs_tracers = namedobs(type(obs).__name__, qubits[0])
    elif isinstance(obs, qml.Hermitian):
        # TODO: remove asarray once fixed upstream: https://github.com/PennyLaneAI/pennylane/issues/4263
        qubits = qrp_extract(qrp, wires, allow_reuse=True)
        obs_tracers = hermitian(jax.numpy.asarray(*obs.parameters), *qubits)
    elif isinstance(obs, qml.operation.Tensor):
        nested_obs = [trace_observables(o, device, qrp, m_wires)[0] for o in obs.obs]
        obs_tracers = tensorobs_p.bind(*nested_obs)
    elif isinstance(obs, qml.Hamiltonian):
        nested_obs = [trace_observables(o, device, qrp, m_wires)[0] for o in obs.ops]
        obs_tracers = hamiltonian(jax.numpy.asarray(obs.parameters), *nested_obs)
    elif isinstance(obs, qml.ops.op_math.Prod):
        nested_obs = [trace_observables(o, device, qrp, m_wires)[0] for o in obs]
        obs_tracers = tensorobs(*nested_obs)
    elif isinstance(obs, qml.ops.op_math.Sum):
        nested_obs = [trace_observables(o, device, qrp, m_wires)[0] for o in obs]
        obs_tracers = hamiltonian(jax.numpy.asarray(jnp.ones(len(obs))), *nested_obs)
    elif isinstance(obs, qml.ops.op_math.SProd):
        terms = obs.terms()
        coeffs = jax.numpy.array(terms[0])
        nested_obs = trace_observables(terms[1][0], device, qrp, m_wires)[0]
        obs_tracers = hamiltonian(coeffs, nested_obs)
    elif paulis := obs._pauli_rep:  # pylint: disable=protected-access
        # Use the pauli sentence representation of the observable, if applicable
        obs_tracers = pauli_sentence_to_hamiltonian_obs(paulis, qrp)
    else:
        raise NotImplementedError(f"Observable {obs} (of type {type(obs)}) is not impemented")
    return obs_tracers, (len(qubits) if qubits else None)


def pauli_sentence_to_hamiltonian_obs(paulis, qrp):
    """Convert a :class:`pennylane.pauli.PauliSentence` into a Hamiltonian.

    Args:
        paulis: a :class:`pennylane.pauli.PauliSentence`
        qubit_states: the statically known qubit state at this program point
        qreg: the quantum register with the state at this program point

    Returns:
        a Hamiltonian JAX primitive used for tracing
    """
    pwords, coeffs = zip(*paulis.items())
    nested_obs = [pauli_word_to_tensor_obs(pword, qrp) for pword in pwords]

    # No need to create a Hamiltonian for a single TensorObs
    if len(nested_obs) == 1 and coeffs[0] == 1.0:
        return nested_obs[0]

    coeffs = jax.numpy.asarray(coeffs)
    return hamiltonian(coeffs, *nested_obs)


def pauli_word_to_tensor_obs(obs, qrp):
    """Convert a :class:`pennylane.pauli.PauliWord` into a Named or Tensor observable.

    Args:
        obs: a :class:`pennylane.pauli.PauliWord`
        qubit_states: the statically known qubit state at this program point
        qreg: the quantum register with the state at this program point

    Returns:
        a NamedObs or TensorObs JAX primitive used for tracing
    """
    if len(obs) == 1:
        wire, pauli = list(obs.items())[0]
        # qubits = [qextract_p.bind(qreg, w) for w in [wire]]
        qubits = [qrp_extract(qrp, [wire], allow_reuse=True)]
        return namedobs(PAULI_NAMED_MAP[pauli], qubits[0])

    nested_obs = []
    for wire, pauli in obs.items():
        # qubits = get_qubits_from_wires([wire], qubit_states, qreg)
        # qubits = [qextract_p.bind(qreg, w) for w in [wire]]
        qubits = [qrp_extract(qrp, [wire], allow_reuse=True)]
        nested_obs.append(jprim.namedobs(PAULI_NAMED_MAP[pauli], qubits[0]))

    return tensorobs(*nested_obs)


def trace_quantum_measurements(
    quantum_tape,
    device: QubitDevice,
    qrp: QRegPromise,
    ctx: JaxTracingContext,
    trace: DynamicJaxprTrace,
    outputs: List[Union[MeasurementProcess, DynamicJaxprTracer, Any]],
    out_tree,
) -> List[DynamicJaxprTracer]:
    shots = device.shots
    out_classical_tracers = []

    for i, o in enumerate(outputs):
        if isinstance(o, MeasurementProcess):
            m_wires = o.wires if o.wires else range(device.num_wires)
            obs_tracers, nqubits = trace_observables(o.obs, device, qrp, m_wires)

            using_compbasis = obs_tracers.primitive == compbasis_p
            if o.return_type.value == "sample":
                shape = (shots, nqubits) if using_compbasis else (shots,)
                out_classical_tracers.append(sample_p.bind(obs_tracers, shots=shots, shape=shape))
            elif o.return_type.value == "expval":
                out_classical_tracers.append(expval_p.bind(obs_tracers, shots=shots))
            elif o.return_type.value == "var":
                out_classical_tracers.append(var_p.bind(obs_tracers, shots=shots))
            elif o.return_type.value == "probs":
                assert using_compbasis
                shape = (2**nqubits,)
                out_classical_tracers.append(probs_p.bind(obs_tracers, shape=shape))
            elif o.return_type.value == "counts":
                shape = (2**nqubits,) if using_compbasis else (2,)
                out_classical_tracers.extend(counts_p.bind(obs_tracers, shots=shots, shape=shape))
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
                shape = (2**nqubits,)
                out_classical_tracers.append(state_p.bind(obs_tracers, shape=shape))
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

    with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
        # (1) - Classical tracing
        quantum_tape = QuantumTape()
        with EvaluationContext.frame_tracing_context(ctx) as trace:
            wffa, in_avals, out_tree_promise = deduce_avals(f, args, kwargs)
            in_classical_tracers = _input_type_to_tracers(trace.new_arg, in_avals)
            with QueuingManager.stop_recording(), quantum_tape:
                # Quantum tape transformations happen at the end of tracing
                ans = wffa.call_wrapped(*in_classical_tracers)
            out_classical_tracers_or_measurements = [
                (trace.full_raise(t) if isinstance(t, DynamicJaxprTracer) else t) for t in ans
            ]

        # (2) - Quantum tracing
        with EvaluationContext.frame_tracing_context(ctx, trace):
            qdevice_p.bind(spec="kwargs", val=str(device.backend_kwargs))
            qdevice_p.bind(spec="backend", val=device.backend_name)
            qreg_in = qalloc_p.bind(len(device.wires))
            qrp_out = trace_quantum_tape(quantum_tape, device, qreg_in, ctx, trace)
            out_classical_tracers, out_classical_tree = trace_quantum_measurements(
                quantum_tape,
                device,
                qrp_out,
                ctx,
                trace,
                out_classical_tracers_or_measurements,
                out_tree_promise(),
            )
            # qreg_out = qrp_actualize(qrp_out)

            qdealloc_p.bind(qreg_in)

            out_classical_tracers = [trace.full_raise(t) for t in out_classical_tracers]
            # out_quantum_tracers = [qreg_out]
            out_quantum_tracers = []

            jaxpr, out_type, consts = ctx.frames[trace].to_jaxpr2(
                out_classical_tracers + out_quantum_tracers
            )
            # jaxpr._outvars = jaxpr._outvars[:-1]
            # out_type = out_type[:-1]
            # FIXME: `check_jaxpr` complains about the `AbstractQreg` type. Consider fixing.
            # check_jaxpr(jaxpr)

    closed_jaxpr = ClosedJaxpr(jaxpr, consts)
    out_avals, _ = unzip2(out_type)
    out_shape = tree_unflatten(
        out_classical_tree, [ShapeDtypeStruct(a.shape, a.dtype, a.named_shape) for a in out_avals]
    )
    return closed_jaxpr, out_shape


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

        with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION):
            jaxpr, shape = trace_quantum_function(self.func, device, args, kwargs)

        retval_tree = tree_structure(shape)

        def _eval_jaxpr(*args):
            return jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)

        args_data, _ = tree_flatten(args)

        wrapped = wrap_init(_eval_jaxpr)
        retval = func_p.bind(wrapped, *args_data, fn=self)

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
            # TODO: Remove once work_wires is no longer needed for decomposition.
            (qml.ops.MultiControlledX, "decomposition", _decomp_controlled),
        ):
            expanded_tape = super().default_expand_fn(circuit, max_expansion)

        self.check_validity(expanded_tape.operations, [])
        return expanded_tape
