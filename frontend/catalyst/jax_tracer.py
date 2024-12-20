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

"""
This module contains functions tracing and lowering JAX code to MLIR.
"""

import logging
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial, reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane import QubitUnitary, QueuingManager
from pennylane.devices import QubitDevice
from pennylane.measurements import (
    CountsMP,
    ExpectationMP,
    MeasurementProcess,
    ProbabilityMP,
    StateMP,
    VarianceMP,
)
from pennylane.operation import AnyWires, Operation, Operator, Wires
from pennylane.ops import Adjoint, Controlled, ControlledOp
from pennylane.tape import QuantumTape
from pennylane.transforms.core import TransformProgram

import catalyst
from catalyst.api_extensions.callbacks import MemrefCallable
from catalyst.jax_extras import (
    ClosedJaxpr,
    DynamicJaxprTrace,
    DynamicJaxprTracer,
    ExpansionStrategy,
    InputSignature,
    OutputSignature,
    PyTreeDef,
    PyTreeRegistry,
    ShapedArray,
    _abstractify,
    _input_type_to_tracers,
    cond_expansion_strategy,
    convert_element_type,
    deduce_avals,
    deduce_signatures,
    eval_jaxpr,
    input_type_to_tracers,
    jaxpr_to_mlir,
    make_jaxpr2,
    sort_eqns,
    transient_jax_config,
    tree_flatten,
    tree_structure,
    tree_unflatten,
    wrap_init,
)
from catalyst.jax_extras.tracing import bind_flexible_primitive
from catalyst.jax_primitives import (
    AbstractQreg,
    compbasis_p,
    counts_p,
    expval_p,
    func_p,
    hamiltonian_p,
    hermitian_p,
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
    set_basis_state_p,
    set_state_p,
    state_p,
    tensorobs_p,
    var_p,
)
from catalyst.logging import debug_logger, debug_logger_init
from catalyst.tracing.contexts import (
    EvaluationContext,
    EvaluationMode,
    JaxTracingContext,
)
from catalyst.utils.exceptions import CompileError

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# TODO: refactor the tracer module
# pylint: disable=too-many-lines

# Global flag tracing wether the function that we trace might be used for gradients
TRACING_GRADIENTS: List[str] = []


def _in_gradient_tracing(qnode) -> Optional[str]:
    """If we are tracing gradient - return the current grad method."""
    if len(TRACING_GRADIENTS) == 0:
        return None

    method = TRACING_GRADIENTS[-1]
    return qnode.diff_method if method == "auto" else method


@contextmanager
def mark_gradient_tracing(method: str):
    """Wraps the inner flow with the gradient-tracing flag"""
    try:
        TRACING_GRADIENTS.append(method)
        yield
    finally:
        TRACING_GRADIENTS.pop()


def _make_execution_config(qnode):
    """Updates the execution_config object with information about execution. This is
    used in preprocess to determine what decomposition and validation is needed."""

    execution_config = qml.devices.ExecutionConfig()
    if qnode:
        execution_config.gradient_method = _in_gradient_tracing(qnode)

    return execution_config


def get_device_shots(dev):
    """Helper function to get device shots."""
    return dev.shots if isinstance(dev, qml.devices.LegacyDevice) else dev.shots.total_shots


def get_device_shot_vector(dev):
    """Helper function to get device shot vector."""
    return [(shot_copy.shots, shot_copy.copies) for shot_copy in dev.shots.shot_vector]


class Function:
    """An object that represents a compiled function.

    At the moment, it is only used to compute sensible names for higher order derivative
    functions in MLIR.

    Args:
        fn (Callable): the function boundary.

    Raises:
        AssertionError: Invalid function type.
    """

    CACHE = {}

    def __new__(cls, fn):
        if cached_instance := cls.CACHE.get(fn):
            return cached_instance
        new_instance = super().__new__(cls)
        cls.CACHE[fn] = new_instance
        return new_instance

    @debug_logger_init
    def __init__(self, fn):
        self.fn = fn
        if isinstance(fn, partial):
            self.__name__ = fn.func.__name__
        else:
            self.__name__ = fn.__name__

    @debug_logger
    def __call__(self, *args, **kwargs):
        jaxpr, _, out_tree = make_jaxpr2(self.fn)(*args, **kwargs)

        def _eval_jaxpr(*args, **kwargs):
            return jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args, **kwargs)

        args, _ = jax.tree_util.tree_flatten((args, kwargs))
        retval = func_p.bind(wrap_init(_eval_jaxpr), *args, fn=self.fn)
        return tree_unflatten(out_tree, retval)


KNOWN_NAMED_OBS = (qml.Identity, qml.PauliX, qml.PauliY, qml.PauliZ, qml.Hadamard)

# Take care when adding primitives to this set in order to avoid introducing a quadratic number of
# edges to the jaxpr equation graph in ``sort_eqns()``. Each equation with a primitive in this set
# is constrained to occur before all subsequent equations in the quantum operations trace.
FORCED_ORDER_PRIMITIVES = {qdevice_p}

PAULI_NAMED_MAP = {
    "I": "Identity",
    "X": "PauliX",
    "Y": "PauliY",
    "Z": "PauliZ",
}


@debug_logger
def retrace_with_result_types(jaxpr: ClosedJaxpr, target_types: List[ShapedArray]) -> ClosedJaxpr:
    """Return a JAXPR that is identical to the given one but with added type conversion operations
    to produce the provided type signature in its output."""
    # TODO: is eval expensive? or would it be better to modify the jaxpr in place?
    with_qreg = isinstance(target_types[-1], AbstractQreg)
    with EvaluationContext(EvaluationMode.CLASSICAL_COMPILATION) as ctx:
        with EvaluationContext.frame_tracing_context(ctx) as trace:
            in_tracers = _input_type_to_tracers(trace.new_arg, jaxpr.in_avals)
            out_tracers = [
                trace.full_raise(t) for t in eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *in_tracers)
            ]
            out_tracers_, target_types_ = (
                (out_tracers[:-1], target_types[:-1]) if with_qreg else (out_tracers, target_types)
            )
            out_promoted_tracers = [
                (convert_element_type(tr, ty) if _abstractify(tr).dtype != ty else tr)
                for tr, ty in zip(out_tracers_, target_types_)
            ]
            jaxpr2, _, consts = ctx.frames[trace].to_jaxpr2(
                out_promoted_tracers + ([out_tracers[-1]] if with_qreg else [])
            )
    return ClosedJaxpr(jaxpr2, consts)


def _apply_result_type_conversion(
    ctx: JaxTracingContext,
    jaxpr: ClosedJaxpr,
    consts: List[Any],
    target_types: List[ShapedArray],
    num_implicit_outputs: int,
) -> Tuple[List[Any], InputSignature, OutputSignature]:
    """Re-trace the ``jaxpr`` program and apply type conversion to its results. Return full
    information about the modified Jaxpr program. The jaxpr program is only allowed to take zero or
    one quantum register as an argument.

    Args:
        ctx: Jax tracing context object.
        jaxpr: The Jaxpr program to apply the conversion to.
        consts: List of constant values we need to know to trace this program.
        target_types: List of types we want to convert the outputs of the program to. The list must
                      match the number of outputs, except maybe the very last output if it is Qreg.
        num_implicit_outputs: Number of implicit outputs found in the Jaxpr program.

    Returns:
        List[TracerLike]: output tracers of the program
        InputSignature: new input signature of the function
        OutputSignature: new output signature of the function
    """
    with_qreg = len(target_types) > 0 and isinstance(target_types[-1], AbstractQreg)
    args = [AbstractQreg()] if with_qreg else []

    def _fun(*in_tracers):
        out_tracers = eval_jaxpr(jaxpr, consts, *in_tracers)
        out_tracers_, target_types_ = (
            (out_tracers[:-1], target_types[:-1]) if with_qreg else (out_tracers, target_types)
        )
        out_promoted_tracers = [
            (convert_element_type(tr, ty) if _abstractify(tr).dtype != ty else tr)
            for tr, ty in zip(out_tracers_, target_types_)
        ]
        return out_promoted_tracers[num_implicit_outputs:] + (
            [out_tracers[-1]] if with_qreg else []
        )

    expanded_tracers, in_sig, out_sig = trace_function(
        ctx, _fun, *args, expansion_strategy=cond_expansion_strategy()
    )

    return expanded_tracers, in_sig, out_sig


def _promote_jaxpr_types(types: List[List[Any]]) -> List[Any]:
    # TODO: Our custom AbstractQreg happened to be incompatible with jnp.promote_types, we suspect
    # we failed to match some expectation of Jax. We suggest to make our abstract values compatible
    # and hopefully remove the logic behind the condition [1]. Should we add AbstractQreg into the
    # `_weak_types` list of JAX?
    assert len(types) > 0, "Expected one or more set of types"
    assert all(len(t) == len(types[0]) for t in types), "Expected matching number of arguments"

    def _shapes(ts):
        return [t.shape for t in ts if isinstance(t, ShapedArray)]

    assert all(_shapes(t) == _shapes(types[0]) for t in types), "Expected matching shapes"
    all_ends_with_qreg = all(len(t) > 0 and isinstance(t[-1], AbstractQreg) for t in types)
    all_not_ends_with_qreg = all(len(t) == 0 or not isinstance(t[-1], AbstractQreg) for t in types)
    assert (
        all_ends_with_qreg or all_not_ends_with_qreg
    ), "We require either all-qregs or all-non-qregs as last items of the type lists"
    if all_ends_with_qreg:  # [1]
        types = [t[:-1] for t in types]
    results = list(map(partial(reduce, jnp.promote_types), zip(*types)))
    return results + ([AbstractQreg()] if all_ends_with_qreg else [])


@debug_logger
def unify_convert_result_types(ctx, jaxprs, consts, nimplouts):
    """Unify result types of the jaxpr programs given.
    Args:
        jaxprs (list of ClosedJaxpr): Source Jaxpr programs. The program results must have
                                      matching sizes and numpy array shapes but dtypes might be
                                      different.
        consts (list of Jaxpr constants): Constants of the sourece Jaxpr programs.
        nimplout (list of integers): Numbers of implicit outputs of Jaxpr programs.

    Returns (list of output signatures):
        Output jaxprs of the new programs
        Output type of the new programs
        Output tracers of the new programs
        Constants of the new programs

    Raises:
        TypePromotionError: Unification is not possible.

    """
    promoted_types = _promote_jaxpr_types([[v.aval for v in j.outvars] for j in jaxprs])
    jaxpr_acc, type_acc, tracers_acc, consts_acc = [], [], [], []
    for j, a, num_implicit_outputs in zip(jaxprs, consts, nimplouts):
        tracers, _, out_sig = _apply_result_type_conversion(
            ctx, j, a, promoted_types, num_implicit_outputs
        )
        jaxpr_acc.append(out_sig.out_initial_jaxpr())
        type_acc.append(out_sig.out_type())
        tracers_acc.append(tracers)
        consts_acc.append(out_sig.out_consts())
    return jaxpr_acc, type_acc[0], tracers_acc, consts_acc


class QRegPromise:
    """QReg adaptor tracing the qubit extractions and insertions. The adaptor works by postponing
    the insertions in order to re-use qubits later thus skipping the extractions."""

    @debug_logger_init
    def __init__(self, qreg: DynamicJaxprTracer):
        self.base: DynamicJaxprTracer = qreg
        self.cache: Dict[Any, DynamicJaxprTracer] = {}

    @debug_logger
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

    @debug_logger
    def insert(self, wires, qubits) -> None:
        """Insert qubits to the cache."""
        qrp = self
        assert len(wires) == len(qubits), f"len(wires)({len(wires)}) != len(qubits)({len(qubits)})"
        for w, qubit in zip(wires, qubits):
            assert (w not in qrp.cache) or (
                qrp.cache[w] is None
            ), f"Attempting to insert an already-inserted wire {w} into {qrp.base}"
            qrp.cache[w] = qubit

    @debug_logger
    def actualize(self) -> DynamicJaxprTracer:
        """Prune the qubit cache by performing the postponed insertions."""
        qrp = self
        qreg = qrp.base
        for w, qubit in qrp.cache.items():
            if qubit is not None:
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


class HybridOp(Operator):
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

    @debug_logger_init
    def __init__(
        self,
        in_classical_tracers,
        out_classical_tracers,
        regions: List[HybridOpRegion],
        apply_reverse_transform=False,
        expansion_strategy=None,
    ):  # pylint: disable=too-many-arguments
        self.in_classical_tracers = in_classical_tracers
        self.out_classical_tracers = out_classical_tracers
        self.regions: List[HybridOpRegion] = regions
        self.expansion_strategy = expansion_strategy
        self.apply_reverse_transform = apply_reverse_transform
        super().__init__(wires=Wires(HybridOp.num_wires))

    def __repr__(self):
        """Constructor-call-like representation."""
        nested_ops = [r.quantum_tape.operations for r in self.regions if r.quantum_tape]
        return f"{self.name}(tapes={nested_ops})"

    @debug_logger
    def bind_overwrite_classical_tracers(
        self,
        ctx: JaxTracingContext,
        trace: DynamicJaxprTrace,
        in_expanded_tracers,
        out_expanded_tracers,
        **kwargs,
    ) -> DynamicJaxprTracer:
        """Binds the JAX primitive but override the returned classical tracers with the already
        existing output tracers, stored in the operations since the classical tracing stage.
        User-defined transformations might have changed them by the time this function is called.
        The quantum tracer, namely the quantum register is not supposed to be changed so it is kept
        as-is.
        """
        assert self.binder is not None, "HybridOp should set a binder"
        out_quantum_tracer = self.binder(*in_expanded_tracers, **kwargs)[-1]
        eqn = ctx.frames[trace].eqns[-1]
        assert len(eqn.outvars[:-1]) == len(
            out_expanded_tracers
        ), f"{eqn.outvars=}\n{out_expanded_tracers=}"
        for i, t in zip(range(len(eqn.outvars[:-1])), out_expanded_tracers):
            if trace.getvar(t) in set(
                [
                    *sum([e.outvars for e in ctx.frames[trace].eqns[:-1]], []),
                    *ctx.frames[trace].invars,
                    *ctx.frames[trace].constvar_to_val.keys(),
                ]
            ):
                # Do not re-assign vars from other equations
                continue
            eqn.outvars[i] = trace.getvar(t)
        return out_quantum_tracer

    @debug_logger
    def trace_quantum(
        self,
        ctx: JaxTracingContext,
        device: QubitDevice,
        trace: DynamicJaxprTrace,
        qrp: QRegPromise,
    ) -> QRegPromise:
        """Perform the second, quantum part of the Hybrid operation tracing."""
        raise NotImplementedError("HybridOp should implement trace")  # pragma: no cover


def has_nested_tapes(op: Operator) -> bool:
    """Detects if the PennyLane operation holds nested quantum tapes or not."""
    return (
        isinstance(op, HybridOp)
        and len(op.regions) > 0
        and any(r.quantum_tape is not None for r in op.regions)
    )


def nested_quantum_regions(op: Operation) -> List[HybridOpRegion]:
    """Returns the list of nested quantum regions."""
    return (
        [region for region in op.regions if region.quantum_tape is not None]
        if isinstance(op, HybridOp)
        else []
    )


@debug_logger
def trace_to_jaxpr(func, static_argnums, abstracted_axes, args, kwargs):
    """Trace a Python function to JAXPR.

    Args:
        func: python function to be traced
        static_argnums: indices of static arguments.
        abstracted_axes: abstracted axes specification. Necessary for JAX to use dynamic tensor
            sizes.
        args: arguments to ``func``
        kwargs: keyword arguments to ``func``

    Returns:
        ClosedJaxpr: the Jaxpr program corresponding to ``func``
        PyTreeDef: PyTree-shape of the return values in ``PyTreeDef``
    """

    with transient_jax_config({"jax_dynamic_shapes": True}):
        make_jaxpr_kwargs = {
            "static_argnums": static_argnums,
            "abstracted_axes": abstracted_axes,
        }
        with EvaluationContext(EvaluationMode.CLASSICAL_COMPILATION):
            jaxpr, out_type, out_treedef = make_jaxpr2(func, **make_jaxpr_kwargs)(*args, **kwargs)
            plugins = EvaluationContext.get_plugins()

    return jaxpr, out_type, out_treedef, plugins


@debug_logger
def lower_jaxpr_to_mlir(jaxpr, func_name):
    """Lower a JAXPR to MLIR.

    Args:
        ClosedJaxpr: the JAXPR to lower to MLIR
        func_name: a name to use for the MLIR function

    Returns:
        ir.Module: the MLIR module coontaining the JAX program
        ir.Context: the MLIR context
    """

    MemrefCallable.clearcache()

    with transient_jax_config({"jax_dynamic_shapes": True}):
        mlir_module, ctx = jaxpr_to_mlir(func_name, jaxpr)

    return mlir_module, ctx


def trace_state_prep(op, qrp):
    """Trace qml.StatePrep

    Args:
        op: StatePrep op being traced
        qrp: QRegPromise object holding the JAX tracer representing the quantum register's state

    Postcondition:
        qrp is updated to hold the output qubits from qml.StatePrep
    """
    assert isinstance(op, qml.StatePrep), "qml.StatePrep expected"

    qubits = qrp.extract(op.wires)
    partial_sv = op.parameters[0]
    # jnp.complex128 is the top element in the type promotion lattice
    # so it is ok to do this.
    # https://jax.readthedocs.io/en/latest/type_promotion.html
    partial_sv = jax.lax.convert_element_type(partial_sv, jnp.dtype(jnp.complex128))
    # The frontend guarantees that partial_sv.shape == (2**wires,)
    # We have a test for that, and just if in the future this changes:
    err_msg = "State vector must have shape (2**wires,)"
    assert partial_sv.shape == (2 ** len(qubits),), err_msg
    qubits2 = set_state_p.bind(*qubits, partial_sv)
    qrp.insert(op.wires, qubits2)


def trace_basis_state(op, qrp):
    """Trace qml.BasisState
    Args:
        op: qml.BasisState op being traced
        qrp: QRegPromise object holding the JAX tracer representing the quantum register's state
    Postcondition:
        qrp is updated to hold the output qubits from qml.StatePrep
    """
    assert isinstance(op, qml.BasisState), "qml.BasisState expected"

    qubits = qrp.extract(op.wires)
    basis_state = jax.lax.convert_element_type(op.parameters[0], jnp.dtype(jnp.bool))
    qubits2 = set_basis_state_p.bind(*qubits, basis_state)
    qrp.insert(op.wires, qubits2)


# pylint: disable=too-many-arguments
@debug_logger
def trace_quantum_operations(
    quantum_tape: QuantumTape,
    device: QubitDevice,
    qreg: DynamicJaxprTracer,
    ctx: JaxTracingContext,
    trace: DynamicJaxprTrace,
    mcm_config: qml.devices.MCMConfig = qml.devices.MCMConfig(),
) -> QRegPromise:
    """Recursively trace ``quantum_tape``'s operations containing both PennyLane original and
    Catalyst extension operations. Produce ``QRegPromise`` object holding the resulting quantum
    register tracer.

    Args:
        quantum_tape: PennyLane quantum tape to trace.
        device: PennyLane quantum device.
        qreg: JAX tracer for quantum register in its initial state.
        ctx: JAX tracing context object.
        trace: JAX frame to emit the Jaxpr equations into.

    Returns:
        qrp: QRegPromise object holding the JAX tracer representing the quantum register into its
             final state.
    """
    # Notes:
    # [1] - At this point JAX equation contains both equations added during the classical tracing
    #       and the equations added during the quantum tracing. The equations are linked by named
    #       variables which are in 1-to-1 correspondence with JAX tracers. Since we create
    #       classical tracers (e.g. for mid-circuit measurements) during the classical tracing, but
    #       emit the corresponding equations only now by ``bind``-ing primitives, we might get
    #       equations in a wrong order. The set of variables are always complete though, so we sort
    #       the equations to restore their correct order.

    def bind_native_operation(qrp, op, controlled_wires, controlled_values, adjoint=False):
        # For named-controlled operations (e.g. CNOT, CY, CZ) - bind directly by name. For
        # Controlled(OP) bind OP with native quantum control syntax, and similarly for Adjoint(OP).
        if type(op) in (Controlled, ControlledOp):
            return bind_native_operation(
                qrp,
                op.base,
                controlled_wires + op.control_wires,
                controlled_values + op.control_values,
                adjoint,
            )
        elif isinstance(op, Adjoint):
            return bind_native_operation(
                qrp, op.base, controlled_wires, controlled_values, not adjoint
            )
        elif isinstance(op, QubitUnitary):
            qubits = qrp.extract(op.wires)
            controlled_qubits = qrp.extract(controlled_wires)
            qubits2 = qunitary_p.bind(
                *[*op.parameters, *qubits, *controlled_qubits, *controlled_values],
                qubits_len=len(qubits),
                ctrl_len=len(controlled_qubits),
                adjoint=adjoint,
            )
            qrp.insert(op.wires, qubits2[: len(qubits)])
            qrp.insert(controlled_wires, qubits2[len(qubits) :])
        elif isinstance(op, qml.GlobalPhase):
            controlled_qubits = qrp.extract(controlled_wires)
            qubits2 = bind_flexible_primitive(
                qinst_p,
                {"static_params": op.parameters},
                *[*controlled_qubits, *controlled_values],
                op=op.name,
                ctrl_len=len(controlled_qubits),
                ctrl_value_len=len(controlled_values),
                adjoint=adjoint,
            )
            qrp.insert(controlled_wires, qubits2)
        elif isinstance(op, qml.StatePrep):
            trace_state_prep(op, qrp)
        elif isinstance(op, qml.BasisState):
            trace_basis_state(op, qrp)
        else:
            qubits = qrp.extract(op.wires)
            controlled_qubits = qrp.extract(controlled_wires)
            qubits2 = bind_flexible_primitive(
                qinst_p,
                {"static_params": op.parameters},
                *[*qubits, *controlled_qubits, *controlled_values],
                op=op.name,
                qubits_len=len(qubits),
                ctrl_len=len(controlled_qubits),
                ctrl_value_len=len(controlled_values),
                adjoint=adjoint,
            )
            qrp.insert(op.wires, qubits2[: len(qubits)])
            qrp.insert(controlled_wires, qubits2[len(qubits) :])
        return qrp

    qrp = QRegPromise(qreg)

    if isinstance(device, qml.devices.LegacyDevice):
        # Old device API expands tapes here. Note: this way some ops might bypass the verification.
        # We decided to ignore this since we are aiming new device API.
        ops = device.expand_fn(quantum_tape).operations
    else:
        ops = quantum_tape.operations

    for op in ops:
        qrp2 = None
        if isinstance(op, HybridOp):
            kwargs = (
                {"postselect_mode": mcm_config.postselect_mode}
                if isinstance(op, catalyst.api_extensions.quantum_operators.MidCircuitMeasure)
                else {}
            )
            qrp2 = op.trace_quantum(ctx, device, trace, qrp, **kwargs)
        elif isinstance(op, MeasurementProcess):
            qrp2 = qrp
        else:
            qrp2 = bind_native_operation(qrp, op, [], [])

        assert qrp2 is not None
        qrp = qrp2

    ctx.frames[trace].eqns = sort_eqns(ctx.frames[trace].eqns, FORCED_ORDER_PRIMITIVES)  # [1]
    return qrp


@debug_logger
def trace_observables(
    obs: Operator, qrp: QRegPromise, m_wires: int
) -> Tuple[List[DynamicJaxprTracer], Optional[int]]:
    """Trace observables.

    Args:
        obs (Operator): an observable operator
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
    elif isinstance(obs, qml.ops.op_math.Prod):
        nested_obs = [trace_observables(o, qrp, m_wires)[0] for o in obs]
        obs_tracers = tensorobs_p.bind(*nested_obs)
    elif isinstance(obs, qml.ops.LinearCombination):
        coeffs, observables = obs.terms()
        nested_obs = [trace_observables(o, qrp, m_wires)[0] for o in observables]
        obs_tracers = hamiltonian_p.bind(jax.numpy.asarray(coeffs), *nested_obs)
    elif isinstance(obs, qml.ops.op_math.Sum):
        nested_obs = [trace_observables(o, qrp, m_wires)[0] for o in obs]
        obs_tracers = hamiltonian_p.bind(jax.numpy.asarray(jnp.ones(len(obs))), *nested_obs)
    elif isinstance(obs, qml.ops.op_math.SProd):
        coeffs, terms = obs.terms()
        coeffs = jax.numpy.array(coeffs)
        nested_obs = []
        for term in terms:
            obs = trace_observables(term, qrp, m_wires)[0]
            nested_obs.append(obs)
        obs_tracers = hamiltonian_p.bind(coeffs, *nested_obs)
    else:
        raise NotImplementedError(
            f"Observable {obs} (of type {type(obs)}) is not implemented"
        )  # pragma: no cover
    return obs_tracers, (len(qubits) if qubits else 0)


@debug_logger
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


@debug_logger
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


def identity_qnode_transform(tape: QuantumTape) -> (Sequence[QuantumTape], Callable):
    """Identity transform"""
    return [tape], lambda res: res[0]


# pylint: disable=too-many-statements,too-many-branches
@debug_logger
def trace_quantum_measurements(
    device: QubitDevice,
    qrp: QRegPromise,
    outputs: List[Union[MeasurementProcess, DynamicJaxprTracer, Any]],
    out_tree: PyTreeDef,
) -> Tuple[List[DynamicJaxprTracer], PyTreeDef]:
    """Trace quantum measurements. Accept a list of QNode ouptputs and its Pytree-shape. Process
    the quantum measurement outputs, leave other outputs as-is.

    Args:
        device (QubitDevice): PennyLane quantum device to use for quantum measurements.
        qrp (QRegPromise): Quantum register tracer with cached qubits
        outputs (List of quantum function results): List of qnode output JAX tracers to process.
        out_tree (PyTreeDef): PyTree-shape of the outputs.
        quantum_tape: PennyLane quantum tape.

    Returns:
        out_classical_tracers: modified list of JAX classical qnode ouput tracers.
        out_tree: modified PyTree-shape of the qnode output.
    """
    shots = get_device_shots(device)
    out_classical_tracers = []

    for i, o in enumerate(outputs):
        if isinstance(o, MeasurementProcess):

            # Check if the measurement is supported shot-vector where num_of_total_copies > 1
            if device.shots.num_copies > 1 and not isinstance(o, qml.measurements.SampleMP):
                raise NotImplementedError(
                    f"Measurement {type(o).__name__} is not supported a shot-vector. "
                    "Use qml.sample() instead."
                )

            m_wires = o.wires if o.wires else range(len(device.wires))

            obs_tracers, nqubits = trace_observables(o.obs, qrp, m_wires)

            using_compbasis = obs_tracers.primitive == compbasis_p

            if isinstance(o, qml.measurements.SampleMP):

                if shots is None:  # needed for old device API only
                    raise ValueError(
                        "qml.sample cannot work with shots=None. "
                        "Please specify a finite number of shots."
                    )
                if o.mv is not None:  # qml.sample(m)
                    out_classical_tracers.append(o.mv)
                else:
                    shape = (shots, nqubits) if using_compbasis else (shots,)
                    result = bind_flexible_primitive(
                        sample_p, {"shots": shots}, obs_tracers, num_qubits=nqubits
                    )
                    if using_compbasis:
                        result = jnp.astype(result, jnp.int64)

                    reshaped_result = ()
                    shot_vector = get_device_shot_vector(device)
                    start_idx = 0  # Start index for slicing
                    for shot, copies in shot_vector:
                        for _ in range(copies):
                            sliced_result = result[start_idx : start_idx + shot]
                            reshaped_result += (sliced_result.reshape(shot, nqubits),)
                            start_idx += shot

                    if len(reshaped_result) == 1:
                        reshaped_result = reshaped_result[0]

                    out_classical_tracers.append(reshaped_result)

            elif type(o) is ExpectationMP:
                out_classical_tracers.append(expval_p.bind(obs_tracers))
            elif type(o) is VarianceMP:
                out_classical_tracers.append(var_p.bind(obs_tracers))
            elif type(o) is ProbabilityMP:
                assert using_compbasis
                shape = (2**nqubits,)
                out_classical_tracers.append(probs_p.bind(obs_tracers, shape=shape))
            elif type(o) is CountsMP:
                if shots is None:  # needed for old device API only
                    raise ValueError(
                        "qml.sample cannot work with shots=None. "
                        "Please specify a finite number of shots."
                    )
                shape = (2**nqubits,) if using_compbasis else (2,)
                results = bind_flexible_primitive(
                    counts_p, {"shots": shots}, obs_tracers, shape=shape
                )
                if using_compbasis:
                    results = (jnp.asarray(results[0], jnp.int64), results[1])
                out_classical_tracers.extend(results)
                counts_tree = tree_structure(("keys", "counts"))
                meas_return_trees_children = out_tree.children()
                if len(meas_return_trees_children):
                    meas_return_trees_children[i] = counts_tree
                    out_tree = out_tree.make_from_node_data_and_children(
                        PyTreeRegistry(),
                        out_tree.node_data(),
                        meas_return_trees_children,
                    )
                else:
                    out_tree = counts_tree
            elif type(o) is StateMP:
                assert using_compbasis
                shape = (2**nqubits,)
                out_classical_tracers.append(state_p.bind(obs_tracers, shape=shape))
            else:
                raise NotImplementedError(
                    f"Measurement {type(o)} is not implemented"
                )  # pragma: no cover
        elif isinstance(o, DynamicJaxprTracer):
            out_classical_tracers.append(o)
        else:
            assert not isinstance(o, (list, dict)), f"Expected a tracer or a measurement, got {o}"
            out_classical_tracers.append(o)

    return out_classical_tracers, out_tree


@debug_logger
def is_transform_valid_for_batch_transforms(tape, flat_results):
    """Not all transforms are valid for batch transforms.
    Batch transforms will increase the number of tapes from 1 to N.
    However, if the wave function collapses or there is any other non-deterministic behaviour
    such as noise present, then each tape would have different results.

    Also, MidCircuitMeasure is a HybridOp, which PL does not handle at the moment.
    Let's wait until mid-circuit measurements are better integrated into both PL
    and Catalyst and discussed more as well."""
    class_tracers, meas_tracers = split_tracers_and_measurements(flat_results)

    # Can transforms be applied?
    # Since transforms are a PL feature and PL does not support the same things as
    # Catalyst, transforms may have invariants that rely on PL invariants.
    # For example:
    #   * mid-circuit measurements (for batch-transforms)
    #   * that the output will be only a sequence of `MeasurementProcess`es.
    def is_measurement(op):
        """Only to avoid 100 character per line limit."""
        return isinstance(op, MeasurementProcess)

    is_out_measurements = map(is_measurement, meas_tracers)
    is_all_out_measurements = all(is_out_measurements) and not class_tracers
    is_out_measurement_sequence = is_all_out_measurements and isinstance(meas_tracers, Sequence)
    is_out_single_measurement = is_all_out_measurements and is_measurement(meas_tracers)

    def is_midcircuit_measurement(op):
        """Only to avoid 100 character per line limit."""
        return isinstance(op, catalyst.api_extensions.MidCircuitMeasure)

    is_valid_output = is_out_measurement_sequence or is_out_single_measurement
    if not is_valid_output:
        msg = (
            "A transformed quantum function must return either a single measurement, "
            "or a nonempty sequence of measurements."
        )
        raise CompileError(msg)

    is_wave_function_collapsed = any(map(is_midcircuit_measurement, tape.operations))
    are_batch_transforms_valid = is_valid_output and not is_wave_function_collapsed
    return are_batch_transforms_valid


@debug_logger
def apply_transform(
    qnode_program,
    device_program,
    device_modify_measurements,
    tape,
    flat_results,
):
    """Apply transform."""
    # Some transforms use trainability as a basis for transforming.
    # See batch_params
    params = tape.get_parameters(trainable_only=False)
    tape.trainable_params = qml.math.get_trainable_indices(params)

    if qnode_program.is_informative:
        msg = "Catalyst does not support informative transforms."
        raise CompileError(msg)

    if qnode_program or device_modify_measurements:
        is_valid_for_batch = is_transform_valid_for_batch_transforms(tape, flat_results)
        total_program = qnode_program + device_program
    else:
        is_valid_for_batch = True
        # Apply the identity transform in order to keep generalization
        total_program = device_program

    tapes, post_processing = total_program([tape])
    if not is_valid_for_batch and len(tapes) > 1:
        msg = "Multiple tapes are generated, but each run might produce different results."
        raise CompileError(msg)
    return tapes, post_processing


@debug_logger
def split_tracers_and_measurements(flat_values):
    """Return classical tracers and measurements"""
    classical = []
    measurements = []
    for flat_value in flat_values:
        if isinstance(flat_value, DynamicJaxprTracer):
            # classical should remain empty for all valid cases at the moment.
            # This is because split_tracers_and_measurements is only called
            # when checking the validity of transforms. And transforms cannot
            # return any tracers.
            classical.append(flat_value)  # pragma: no cover
        else:
            measurements.append(flat_value)

    return classical, measurements


@debug_logger
def trace_post_processing(ctx, trace, post_processing: Callable, pp_args):
    """Trace post processing function.

    Args:
        ctx (EvaluationContext): context
        trace (DynamicJaxprTrace): trace
        post_processing(Callable): PennyLane transform post_processing function
        pp_args(structured Jax tracers): List of results returned from the PennyLane quantum
                                         transform function

    Returns:
        closed_jaxpr(ClosedJaxpr): JAXPR expression for the whole frame
        out_type(jax.OutputType) : List of abstract values with explicitness flag
        out_tree(PyTreeDef): PyTree shape of the qnode result
    """

    with EvaluationContext.frame_tracing_context(ctx, trace):
        # What is the input to the post_processing function?
        # The input to the post_processing function is going to be a list of values One for each
        # tape. The tracers are all flat in pp_args.

        # We need to deduce the type/shape/tree of the post_processing.
        wffa, _, _, out_tree_promise = deduce_avals(post_processing, (pp_args,), {})

        # wffa will take as an input a flatten tracers.
        # After wffa is called, then the shape becomes available in out_tree_promise.
        in_tracers = [trace.full_raise(t) for t in tree_flatten(pp_args)[0]]
        out_tracers = [trace.full_raise(t) for t in wffa.call_wrapped(*in_tracers)]
        jaxpr, out_type, consts = ctx.frames[trace].to_jaxpr2(out_tracers)
        closed_jaxpr = ClosedJaxpr(jaxpr, consts)
        return closed_jaxpr, out_type, out_tree_promise()


@debug_logger
def trace_function(
    ctx: JaxTracingContext, fun: Callable, *args, expansion_strategy: ExpansionStrategy, **kwargs
) -> Tuple[List[Any], InputSignature, OutputSignature]:
    """Trace classical Python function containing no quantum computations. Arguments and results of
    the function are allowed to contain dynamic dimensions. Depending on the expansion strategy, the
    resulting Jaxpr program might or might not preserve sharing among the dynamic dimension
    variables. The support for expansion options makes this function different from
    `jax_extras.make_jaxpr2`.

    Args:
        ctx: Jax tracing context helper.
        fun: Callable python function.
        expansion_strategy: dynamic dimension expansion options.
        *args: Sample positional arguments of the function.
        **kwargs: Sample keyword arguments of the function.

    Result:
        Expanded list of output Jax tracers
        InputSignature of the resulting Jaxpr program
        OutputSignature of the resulting Jaxpr program
    """
    wfun, in_sig, out_sig = deduce_signatures(
        fun, args, kwargs, expansion_strategy=expansion_strategy
    )

    with EvaluationContext.frame_tracing_context(ctx) as trace:
        arg_expanded_tracers = input_type_to_tracers(
            in_sig.in_type, trace.new_arg, trace.full_raise
        )
        res_expanded_tracers = wfun.call_wrapped(*arg_expanded_tracers)

        return res_expanded_tracers, in_sig, out_sig


@debug_logger
def trace_quantum_function(
    f: Callable, device: QubitDevice, args, kwargs, qnode, static_argnums
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
        qnode: The quantum node to be traced, it contains user transforms.

    Returns:
        closed_jaxpr: JAXPR expression of the function ``f``.
        out_type: JAXPR output type (list of abstract values with explicitness flags).
        out_tree: PyTree shapen of the result
    """

    with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
        # (1) - Classical tracing
        quantum_tape = QuantumTape(shots=device.shots)
        with EvaluationContext.frame_tracing_context(ctx) as trace:
            wffa, in_avals, keep_inputs, out_tree_promise = deduce_avals(
                f, args, kwargs, static_argnums
            )
            in_classical_tracers = _input_type_to_tracers(trace.new_arg, in_avals)
            with QueuingManager.stop_recording(), quantum_tape:
                # Quantum tape transformations happen at the end of tracing
                in_classical_tracers = [t for t, k in zip(in_classical_tracers, keep_inputs) if k]
                return_values_flat = wffa.call_wrapped(*in_classical_tracers)
            # Ans contains the leaves of the pytree (empty for measurement without
            # data https://github.com/PennyLaneAI/pennylane/pull/4607)
            # Therefore we need to compute the tree with measurements as leaves and it comes
            # with an extra computational cost

            # 1. Recompute the original return
            with QueuingManager.stop_recording():
                return_values = tree_unflatten(out_tree_promise(), return_values_flat)

            def is_leaf(obj):
                return isinstance(obj, qml.measurements.MeasurementProcess)

            # 2. Create a new tree that has measurements as leaves
            return_values_flat, return_values_tree = jax.tree_util.tree_flatten(
                return_values, is_leaf=is_leaf
            )
            if isinstance(device, qml.devices.Device):
                config = _make_execution_config(qnode)
                device_program, config = device.preprocess(ctx, config)
                device_modify_measurements = config.device_options["transforms_modify_measurements"]
            else:
                device_program = TransformProgram()
                device_modify_measurements = False  # this is only for the new API transform program

            qnode_program = qnode.transform_program if qnode else TransformProgram()

            tapes, post_processing = apply_transform(
                qnode_program,
                device_program,
                device_modify_measurements,
                quantum_tape,
                return_values_flat,
            )

        # (2) - Quantum tracing
        transformed_results = []

        with EvaluationContext.frame_tracing_context(ctx, trace):
            qnode_transformed = len(qnode_program) > 0
            for tape in tapes:
                # Set up quantum register for the current tape.
                # We just need to ensure there is a tape cut in between each.
                # Each tape will be outlined into its own function with mlir pass
                # -split-multiple-tapes

                # TODO: device shots is now always a concrete integer or None
                # When PennyLane allows dynamic shots, update tracing to accept dynamic shots too
                device_shots = get_device_shots(device) or 0
                qdevice_p.bind(
                    device_shots,
                    rtd_lib=device.backend_lib,
                    rtd_name=device.backend_name,
                    rtd_kwargs=str(device.backend_kwargs),
                )
                qreg_in = qalloc_p.bind(len(device.wires))

                # If the program is batched, that means that it was transformed.
                # If it was transformed, that means that the program might have
                # changed the output. See `split_non_commuting`
                if qnode_transformed or device_modify_measurements:
                    # TODO: In the future support arbitrary output from the user function.
                    output = tape.measurements
                    _, trees = jax.tree_util.tree_flatten(output, is_leaf=is_leaf)
                else:
                    output = return_values_flat
                    trees = return_values_tree

                mcm_config = qnode.execute_kwargs["mcm_config"]
                qrp_out = trace_quantum_operations(tape, device, qreg_in, ctx, trace, mcm_config)
                meas, meas_trees = trace_quantum_measurements(device, qrp_out, output, trees)
                qreg_out = qrp_out.actualize()

                # Check if the measurements are nested then apply the full_raise
                def check_full_raise(arr, func):
                    if isinstance(arr, (list, tuple)):
                        return type(arr)(check_full_raise(x, func) for x in arr)
                    else:
                        return func(arr)

                meas_tracers = check_full_raise(meas, trace.full_raise)
                meas_results = tree_unflatten(meas_trees, meas_tracers)

                # TODO: Allow the user to return whatever types they specify.
                if qnode_transformed or device_modify_measurements:
                    assert isinstance(meas_results, list)
                    if len(meas_results) == 1:
                        transformed_results.append(meas_results[0])
                    else:
                        transformed_results.append(tuple(meas_results))
                else:
                    transformed_results.append(meas_results)

                # Deallocate the register after the current tape is finished
                # This dealloc primitive also serves as the tape cut when splitting tapes
                qdealloc_p.bind(qreg_out)

        closed_jaxpr, out_type, out_tree = trace_post_processing(
            ctx, trace, post_processing, transformed_results
        )
        # TODO: `check_jaxpr` complains about the `AbstractQreg` type. Consider fixing.
        # check_jaxpr(jaxpr)
    return closed_jaxpr, out_type, out_tree, return_values_tree
