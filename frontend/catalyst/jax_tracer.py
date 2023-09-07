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

from typing import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane import QubitDevice, QubitUnitary, QueuingManager
from pennylane.measurements import MeasurementProcess
from pennylane.operation import AnyWires, Operation, Wires
from pennylane.tape import QuantumTape

from catalyst.jax_primitives import (
    compbasis_p,
    counts_p,
    expval_p,
    func_p,
    hamiltonian_p,
    hermitian_p,
    mlir_fn_cache,
    namedobs_p,
    probs_p,
    qalloc_p,
    qdealloc_p,
    qdevice_p,
    qextract_p,
    qinsert_p,
    qinst_p,
    qunitary_p,
    sample_p,
    state_p,
    tensorobs_p,
    var_p,
)
from catalyst.utils.contexts import EvaluationContext, EvaluationMode, JaxTracingContext
from catalyst.utils.jax_extras import (
    ClosedJaxpr,
    DynamicJaxprTrace,
    DynamicJaxprTracer,
    PyTreeDef,
    ShapeDtypeStruct,
    _input_type_to_tracers,
    deduce_avals,
    jaxpr_to_mlir,
    pytree,
    sort_eqns,
    tree_structure,
    tree_unflatten,
    unzip2,
    wrap_init,
)


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

FORCED_ORDER_PRIMITIVES = {qdevice_p, qextract_p, qinst_p}

PAULI_NAMED_MAP = {
    "I": "Identity",
    "X": "PauliX",
    "Y": "PauliY",
    "Z": "PauliZ",
}


class QRegPromise:
    """QReg adaptor tracing the qubit extractions and insertions. The adaptor works by postponing
    the insertions in order to re-use qubits later thus skipping the extractions."""

    def __init__(self, qreg: DynamicJaxprTracer):
        self.base: DynamicJaxprTracer = qreg
        self.cache: Dict[Any, DynamicJaxprTracer] = {}

    def extract(self, wires: List[Any], allow_reuse=False) -> List[DynamicJaxprTracer]:
        """Extract qubits from the wrapped quantum register or get the already extracted qubits
        from cache"""
        # pylint: disable=consider-iterating-dictionary
        qrp = self
        cached_tracers = {w for w in qrp.cache.keys() if not isinstance(w, int)}
        requested_tracers = {w for w in wires if not isinstance(w, int)}
        if cached_tracers != requested_tracers:
            qrp.actualize()
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

    def insert(self, wires, qubits) -> None:
        """Insert qubits to the cache."""
        qrp = self
        assert len(wires) == len(qubits)
        for w, qubit in zip(wires, qubits):
            assert (w not in qrp.cache) or (
                qrp.cache[w] is None
            ), f"Attempting to insert an already-inserted wire {w} into {qrp.base}"
            qrp.cache[w] = qubit

    def actualize(self) -> DynamicJaxprTracer:
        """Prune the qubit cache by performing the postponed insertions."""
        qrp = self
        qreg = qrp.base
        for w, qubit in qrp.cache.items():
            qreg = qinsert_p.bind(qreg, w, qubit)
        qrp.cache = {}
        qrp.base = qreg
        return qreg


@dataclass
class HybridOpRegion:
    """A code region of a nested HybridOp operation containing a JAX trace manager, a quantum tape,
    input and output classical tracers.

    Args:
        trace: JAX tracing context holding the tracers and equations for this region.
        quantum_tape: PennyLane tape containing quantum operations of this region.
        arg_classical_tracers: JAX tracers or constants, available in this region as
                               arguments during the classical tracing.
        res_classical_tracers: JAX tracers or constants returned to the outer scope after the
                               classical tracing of this region.

    """

    trace: Optional[DynamicJaxprTrace]
    quantum_tape: Optional[QuantumTape]
    arg_classical_tracers: List[DynamicJaxprTracer]
    res_classical_tracers: List[DynamicJaxprTracer]


class HybridOp(Operation):
    """A base class for operations carrying nested regions. The class stores the information
    obtained in the process of classical tracing and required for the completion of the quantum
    tracing. The methods of this class describe various aspects of quantum tracing.

    Args:
        in_classical_tracers (List of JAX tracers or constants):
            Classical tracers captured in the beginning of the classical tracing.
        out_classical_tracers (List of JAX tracers or constants):
            Classical tracers released as results of the classical tracing of this operation.
        regions (List of HybridOpRegions):
            Inner regions (e.g. body of a for-loop), each with its arguments, results and quantum
            tape, captured during the classical tracing.
        binder (Callable):
            JAX primitive binder function to call when the quantum tracing is complete.
    """

    def _no_binder(self, *_):
        raise RuntimeError("{self} does not support JAX binding")  # pragma: no cover

    num_wires = AnyWires
    binder: Callable = _no_binder

    def __init__(self, in_classical_tracers, out_classical_tracers, regions: List[HybridOpRegion]):
        self.in_classical_tracers = in_classical_tracers
        self.out_classical_tracers = out_classical_tracers
        self.regions: List[HybridOpRegion] = regions
        super().__init__(wires=Wires(HybridOp.num_wires))

    def __repr__(self):
        """Constructor-call-like representation."""
        nested_ops = [r.quantum_tape.operations for r in self.regions if r.quantum_tape]
        return f"{self.name}(tapes={nested_ops})"

    def bind_overwrite_classical_tracers(
        self, ctx: JaxTracingContext, trace: DynamicJaxprTrace, *args, **kwargs
    ) -> DynamicJaxprTracer:
        """Binds the JAX primitive but override the returned classical tracers with the already
        existing output tracers, stored in the operations."""
        # Notes:
        # [1] - We are interested in a new quantum tracer only, so we ignore all other (classical)
        #       tracers returned by JAX.
        # [2] - We add the already existing classical tracers into the last JAX equation created by
        #       JAX bind handler of the ``trace`` object.
        assert self.binder is not None, "HybridOp should set a binder"
        out_quantum_tracer = self.binder(*args, **kwargs)[-1]  # [1]
        eqn = ctx.frames[trace].eqns[-1]
        assert (len(eqn.outvars) - 1) == len(self.out_classical_tracers)
        for i, t in zip(range(len(eqn.outvars) - 1), self.out_classical_tracers):  # [2]
            eqn.outvars[i] = trace.getvar(t)
        return out_quantum_tracer

    def trace_quantum(
        self,
        ctx: JaxTracingContext,
        device: QubitDevice,
        trace: DynamicJaxprTrace,
        qrp: QRegPromise,
    ) -> QRegPromise:
        """Perform the second, quantum part of the Hybrid operation tracing."""
        raise NotImplementedError("HybridOp should implement trace")  # pragma: no cover


def has_nested_tapes(op: Operation) -> bool:
    """Detects if the PennyLane operation holds nested quantum tapes or not."""
    return (
        isinstance(op, HybridOp)
        and len(op.regions) > 0
        and any(r.quantum_tape is not None for r in op.regions)
    )


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
    """Recursively trace ``quantum_tape`` containing both PennyLane original and Catalyst extension
    operations. Produce ``QRegPromise`` object holding the resulting quantum register tracer.

    Args:
        quantum_tape: PennyLane quantum tape to trace.
        device: PennyLane quantum device.
        qreg: JAX tracer for quantum register in its initial state.
        ctx: JAX tracing context object.
        trace: JAX frame to emit the Jaxpr quations into.

    Returns:
        qrp: QRegPromise object holding the JAX tracer representing the quantum register into its
             final state.
    """
    # Notes:
    # [1] - At this point JAX equation contains both equations added during the classical tracing
    #       and the equations added during the quantum tracing. The equations are linked by named
    #       variables which are in 1-to-1 correspondance with JAX tracers. Since we create
    #       classical tracers (e.g. for mid-circuit measurements) during the classical tracing, but
    #       emit the corresponding equations only now by ``bind``-ing primitives, we might get
    #       equatoins in a wrong order. The set of variables are always complete though, so we sort
    #       the equations to restore their correct order.

    qrp = QRegPromise(qreg)
    for op in device.expand_fn(quantum_tape):
        qrp2 = None
        if isinstance(op, HybridOp):
            qrp2 = op.trace_quantum(ctx, device, trace, qrp)
        else:
            if isinstance(op, MeasurementProcess):
                qrp2 = qrp
            else:
                qubits = qrp.extract(op.wires)
                if isinstance(op, QubitUnitary):
                    qubits2 = qunitary_p.bind(*[*op.parameters, *qubits])
                else:
                    qubits2 = qinst_p.bind(
                        *qubits, *op.parameters, op=op.name, qubits_len=len(qubits)
                    )
                qrp.insert(op.wires, qubits2)
                qrp2 = qrp

        assert qrp2 is not None
        qrp = qrp2

    ctx.frames[trace].eqns = sort_eqns(ctx.frames[trace].eqns, FORCED_ORDER_PRIMITIVES)  # [1]
    return qrp


def trace_observables(
    obs: Operation, qrp: QRegPromise, m_wires: int
) -> Tuple[List[DynamicJaxprTracer], Optional[int]]:
    """Trace observables.

    Args:
        obs (Operation): an observable operation
        qrp (QRegPromise): Quantum register tracer with cached qubits
        m_wires (int): the default number of wires to use for this measurement process

    Returns:
        out_classical_tracers: a list of classical tracers corresponding to the measured values.
        nqubits: number of actually measured qubits.
    """
    wires = obs.wires if (obs and len(obs.wires) > 0) else m_wires
    qubits = None
    if obs is None:
        qubits = qrp.extract(wires, allow_reuse=True)
        obs_tracers = compbasis_p.bind(*qubits)
    elif isinstance(obs, KNOWN_NAMED_OBS):
        qubits = qrp.extract(wires, allow_reuse=True)
        obs_tracers = namedobs_p.bind(qubits[0], kind=type(obs).__name__)
    elif isinstance(obs, qml.Hermitian):
        # TODO: remove once fixed upstream: https://github.com/PennyLaneAI/pennylane/issues/4263
        qubits = qrp.extract(wires, allow_reuse=True)
        obs_tracers = hermitian_p.bind(jax.numpy.asarray(*obs.parameters), *qubits)
    elif isinstance(obs, qml.operation.Tensor):
        nested_obs = [trace_observables(o, qrp, m_wires)[0] for o in obs.obs]
        obs_tracers = tensorobs_p.bind(*nested_obs)
    elif isinstance(obs, qml.Hamiltonian):
        nested_obs = [trace_observables(o, qrp, m_wires)[0] for o in obs.ops]
        obs_tracers = hamiltonian_p.bind(jax.numpy.asarray(obs.parameters), *nested_obs)
    elif paulis := obs._pauli_rep:  # pylint: disable=protected-access
        # Use the pauli sentence representation of the observable, if applicable
        obs_tracers = pauli_sentence_to_hamiltonian_obs(paulis, qrp)
    elif isinstance(obs, qml.ops.op_math.Prod):
        nested_obs = [trace_observables(o, qrp, m_wires)[0] for o in obs]
        obs_tracers = tensorobs_p.bind(*nested_obs)
    elif isinstance(obs, qml.ops.op_math.Sum):
        nested_obs = [trace_observables(o, qrp, m_wires)[0] for o in obs]
        obs_tracers = hamiltonian_p.bind(jax.numpy.asarray(jnp.ones(len(obs))), *nested_obs)
    elif isinstance(obs, qml.ops.op_math.SProd):
        terms = obs.terms()
        coeffs = jax.numpy.array(terms[0])
        nested_obs = trace_observables(terms[1][0], qrp, m_wires)[0]
        obs_tracers = hamiltonian_p.bind(coeffs, nested_obs)
    else:
        raise NotImplementedError(
            f"Observable {obs} (of type {type(obs)}) is not impemented"
        )  # pragma: no cover
    return obs_tracers, (len(qubits) if qubits else None)


def pauli_sentence_to_hamiltonian_obs(paulis, qrp: QRegPromise) -> List[DynamicJaxprTracer]:
    """Convert a :class:`pennylane.pauli.PauliSentence` into a Hamiltonian.

    Args:
        paulis: a :class:`pennylane.pauli.PauliSentence`
        qrp (QRegPromise): Quantum register tracer with cached qubits

    Returns:
        List of JAX tracers representing a Hamiltonian
    """
    pwords, coeffs = zip(*paulis.items())
    nested_obs = [pauli_word_to_tensor_obs(pword, qrp) for pword in pwords]

    # No need to create a Hamiltonian for a single TensorObs
    if len(nested_obs) == 1 and coeffs[0] == 1.0:
        return nested_obs[0]

    coeffs = jax.numpy.asarray(coeffs)
    return hamiltonian_p.bind(coeffs, *nested_obs)


def pauli_word_to_tensor_obs(obs, qrp: QRegPromise) -> List[DynamicJaxprTracer]:
    """Convert a :class:`pennylane.pauli.PauliWord` into a Named or Tensor observable.

    Args:
        obs: a :class:`pennylane.pauli.PauliWord`
        qrp (QRegPromise): Quantum register tracer with cached qubits

    Returns:
        List of JAX tracers representing NamedObs or TensorObs
    """
    if len(obs) == 1:
        wire, pauli = list(obs.items())[0]
        qubits = qrp.extract([wire], allow_reuse=True)
        return namedobs_p.bind(qubits[0], kind=PAULI_NAMED_MAP[pauli])

    nested_obs = []
    for wire, pauli in obs.items():
        qubits = qrp.extract([wire], allow_reuse=True)
        nested_obs.append(namedobs_p.bind(qubits[0], kind=PAULI_NAMED_MAP[pauli]))

    return tensorobs_p.bind(*nested_obs)


def trace_quantum_measurements(
    device: QubitDevice,
    qrp: QRegPromise,
    outputs: List[Union[MeasurementProcess, DynamicJaxprTracer, Any]],
    out_tree: PyTreeDef,
) -> Tuple[List[DynamicJaxprTracer], PyTreeDef]:
    """Trace quantum measurement. Accept a list of QNode ouptputs and its Pytree-shape. Process
    the quantum measurement outputs, leave other outputs as-is.

    Args:
        device (QubitDevice): PennyLane quantum device to use for quantum measurements.
        qrp (QRegPromise): Quantum register tracer with cached qubits
        outputs (List of quantum function results): List of qnode output JAX tracers to process.
        out_tree (PyTreeDef): PyTree-shape of the outputs.

    Returns:
        out_classical_tracers: modified list of JAX classical qnode ouput tracers.
        out_tree: modified PyTree-shape of the qnode output.
    """
    # pylint: disable=too-many-branches
    shots = device.shots
    out_classical_tracers = []

    for i, o in enumerate(outputs):
        if isinstance(o, MeasurementProcess):
            m_wires = o.wires if o.wires else range(device.num_wires)
            obs_tracers, nqubits = trace_observables(o.obs, qrp, m_wires)

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
                if len(meas_return_trees_children):
                    meas_return_trees_children[i] = counts_tree
                    out_tree = out_tree.make_from_node_data_and_children(
                        pytree.PyTreeRegistry(),
                        out_tree.node_data(),
                        meas_return_trees_children,
                    )
                else:
                    out_tree = counts_tree
            elif o.return_type.value == "state":
                assert using_compbasis
                shape = (2**nqubits,)
                out_classical_tracers.append(state_p.bind(obs_tracers, shape=shape))
            else:
                raise NotImplementedError(
                    f"Measurement {o.return_type.value} is not impemented"
                )  # pragma: no cover
        elif isinstance(o, DynamicJaxprTracer):
            out_classical_tracers.append(o)
        else:
            assert not isinstance(o, (list, dict)), f"Expected a tracer or a measurement, got {o}"
            out_classical_tracers.append(o)

    return out_classical_tracers, out_tree

def transform_callback(tape: QuantumTape) -> (Sequence[QuantumTape], Callable):
    # for transform registered
    #  transform(tape) ... (recursively?) I suppose that is what the execute and
    # transform dispatch are doing.
    #return [tape, tape], lambda res: jnp.add(res[0], res[1])
    return [tape], lambda res: res[0]


def trace_quantum_function(
    f: Callable, device: QubitDevice, args, kwargs
) -> Tuple[ClosedJaxpr, Any]:
    """Trace quantum function in a way that allows building a nested quantum tape describing the
    quantum algorithm.

    The tracing is done as follows: (1) Classical tracing, producing the classical JAX tracers and
    the quantum tape (2) Quantum tape tracing, producing the remaining quantum and classical JAX
    tracers. With all the tracers in hands, the final JAXPR is produced. Note that caller can apply
    tape transformations by using PennyLane's transformation API on the argument function.

    Args:
        f (Callable): a function to trace
        device (QubitDevice): Quantum device to use for quantum computations
        args: Positional arguments to pass to ``f``
        kwargs: Keyword arguments to pass to ``f``

    Returns:
        closed_jaxpr: JAXPR expression of the function ``f``.
        abstract_results: Output structure filled with abstract return values of ``f``.
    """

    with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
        # (1) - Classical tracing
        quantum_tape = QuantumTape()
        with EvaluationContext.frame_tracing_context(ctx) as trace:
            wffa, in_avals, out_tree_promise = deduce_avals(f, args, kwargs)
            in_classical_tracers = _input_type_to_tracers(trace.new_arg, in_avals)
            with QueuingManager.stop_recording(), quantum_tape:
                # Quantum tape transformations happen at the end of tracing
                ans = wffa.call_wrapped(*in_classical_tracers)

            # Ans contains the leaves of the pytree (empty for measurement without
            # data https://github.com/PennyLaneAI/pennylane/pull/4607)
            # Therefore we need to compute the tree with measurements as leaves and it comes
            # with an extra computational cost

            # 1. Recompute the original return
            with QueuingManager.stop_recording():
                ans = tree_unflatten(out_tree_promise(), ans)

            def is_leaf(obj):
                return isinstance(obj, qml.measurements.MeasurementProcess)

            # 2. Create a new tree that has measurements as leaves
            ans, out_tree = jax.tree_util.tree_flatten(ans, is_leaf=is_leaf)

            out_classical_tracers_or_measurements = [
                (trace.full_raise(t) if isinstance(t, DynamicJaxprTracer) else t) for t in ans
            ]

        # (2) - Quantum tracing
        tapes, callback = transform_callback(quantum_tape)
        del quantum_tape
        results = []
        results_tracers = []
        for tape in tapes:
            with EvaluationContext.frame_tracing_context(ctx, trace):

                qdevice_p.bind(spec="kwargs", val=str(device.backend_kwargs))
                qdevice_p.bind(spec="backend", val=device.backend_name)
                qreg_in = qalloc_p.bind(len(device.wires))
                qrp_out = trace_quantum_tape(tape, device, qreg_in, ctx, trace)
                out_classical_tracers, out_classical_tree = trace_quantum_measurements(
                    device,
                    qrp_out,
                    out_classical_tracers_or_measurements,
                    out_tree_promise(),
                )
                out_quantum_tracers = [qrp_out.actualize()]
                qdealloc_p.bind(qreg_in)

                out_classical_tracers = [trace.full_raise(t) for t in out_classical_tracers]
                results_tracers += out_classical_tracers

                jaxpr, out_type, consts = ctx.frames[trace].to_jaxpr2(
                    out_classical_tracers + out_quantum_tracers
                )
                jaxpr._outvars = jaxpr._outvars[:-1]  # pylint: disable=protected-access
                out_type = out_type[:-1]

                out_avals, _ = unzip2(out_type)
                abstract_results = tree_unflatten(
                    out_classical_tree, [ShapeDtypeStruct(a.shape, a.dtype, a.named_shape) for a in out_avals]
                )
                # TODO: `check_jaxpr` complains about the `AbstractQreg` type. Consider fixing.
                # check_jaxpr(jaxpr)

            results += results_tracers

        with EvaluationContext.frame_tracing_context(ctx, trace):
            results_tracers_flat, tracers_tree = tree_flatten(results_tracers)
            import pdb
            pdb.set_trace()
            wffa, in_avals, out_tree_promise = deduce_avals(callback, results_tracers, dict())
            wffa.call_wrapped(results_tracers)
            results_tracers = callback(results_tracers_flat)
            out_classical_tracers = [trace.full_raise(results_tracers)]
            jaxpr, out_type, consts = ctx.frames[trace].to_jaxpr2(out_classical_tracers)
            closed_jaxpr = ClosedJaxpr(jaxpr, consts)
            out_avals, _ = unzip2(out_type)
            abstract_results = tree_unflatten(out_tree_promise(), [ShapeDtypeStruct(a.shape, a.dtype, a.named_shape) for a in out_classical_tracers])

    return closed_jaxpr, abstract_results
