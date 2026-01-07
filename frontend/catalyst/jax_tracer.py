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

import itertools
import logging
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from functools import partial, reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax._src.interpreters.partial_eval as pe
import jax.numpy as jnp
import pennylane as qml
from jax._src.lax import lax
from jax._src.lax.lax import _extract_tracers_dyn_shape
from jax._src.pjit import jit_p
from jax._src.source_info_util import current as current_source_info
from jax.api_util import debug_info as jdb
from jax.core import Tracer, get_aval
from pennylane import QubitUnitary, QueuingManager
from pennylane.devices import QubitDevice
from pennylane.measurements import (
    CountsMP,
    ExpectationMP,
    MeasurementProcess,
    ProbabilityMP,
    SampleMP,
    StateMP,
    VarianceMP,
)
from pennylane.operation import Operation, Operator, Wires
from pennylane.ops import Adjoint, Controlled, ControlledOp
from pennylane.tape import QuantumTape

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
    ShapedArray,
    _input_type_to_tracers,
    cond_expansion_strategy,
    convert_element_type,
    deduce_avals,
    deduce_signatures,
    eval_jaxpr,
    input_type_to_tracers,
    jaxpr_to_mlir,
    make_from_node_data_and_children,
    make_jaxpr2,
    sort_eqns,
    transient_jax_config,
    tree_flatten,
    tree_structure,
    tree_unflatten,
    wrap_init,
)
from catalyst.jax_extras.patches import (
    patched_drop_unused_vars,
    patched_dyn_shape_staging_rule,
    patched_make_eqn,
    patched_pjit_staging_rule,
)
from catalyst.jax_extras.tracing import uses_transform
from catalyst.jax_primitives import (
    AbstractQreg,
    compbasis_p,
    counts_p,
    device_init_p,
    device_release_p,
    expval_p,
    func_p,
    gphase_p,
    hamiltonian_p,
    hermitian_p,
    namedobs_p,
    num_qubits_p,
    probs_p,
    qalloc_p,
    qdealloc_p,
    qextract_p,
    qinsert_p,
    qinst_p,
    sample_p,
    set_basis_state_p,
    set_state_p,
    state_p,
    tensorobs_p,
    unitary_p,
    var_p,
)
from catalyst.logging import debug_logger, debug_logger_init
from catalyst.tracing.contexts import EvaluationContext, EvaluationMode
from catalyst.utils.exceptions import CompileError
from catalyst.utils.patching import DictPatchWrapper, Patcher

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


def _get_eqn_from_tracing_eqn(eqn_or_callable):
    """Helper function to extract the actual equation from JAX 0.7's TracingEqn wrapper."""
    if callable(eqn_or_callable):
        actual_eqn = eqn_or_callable()
        if actual_eqn is None:  # pragma: no cover
            raise RuntimeError("TracingEqn weakref was garbage collected")
        return actual_eqn
    else:  # pragma: no cover
        return eqn_or_callable


def _make_execution_config(qnode):
    """Updates the execution_config object with information about execution. This is
    used in preprocess to determine what decomposition and validation is needed."""

    execution_config = qml.devices.ExecutionConfig()
    if qnode:
        execution_config = replace(execution_config, gradient_method=_in_gradient_tracing(qnode))

    return execution_config


@dataclass
class ClassicalTraceResult:
    """Result from classical tracing phase.

    Args:
        trace: The JAX trace context
        tapes: List of quantum tapes after transformations
        post_processing: Post-processing function for results
        tracing_mode: The tracing mode (TRANSFORM or normal)
        return_values_flat: Flattened return values
        return_values_tree: PyTree structure of return values
        is_leaf: Function to identify measurement leaves
    """

    trace: Any
    tapes: List[Any]
    post_processing: Callable
    tracing_mode: Any
    return_values_flat: List[Any]
    return_values_tree: Any
    is_leaf: Callable


@dataclass
class TraceResult:
    """Result from tracing a quantum function.

    This class groups related return values from trace_quantum_function

    Args:
        closed_jaxpr: JAXPR expression of the traced function
        out_type: JAXPR output type (list of abstract values with explicitness flags)
        out_tree: PyTree shape of the result
        return_values_tree: PyTree structure of return values with measurements as leaves
        classical_return_indices: Indices of classical return values
        num_mcm: Number of mid-circuit measurements
    """

    closed_jaxpr: Any
    out_type: Any
    out_tree: Any
    return_values_tree: Any
    classical_return_indices: List[int]
    num_mcm: int


class Function:
    """An object that represents a compiled function.

    At the moment, it is only used to compute sensible names for higher order derivative
    functions in MLIR.

    Args:
        fn (Callable): the function boundary.

    Raises:
        AssertionError: Invalid function type.
    """

    @debug_logger_init
    def __init__(self, fn):
        self.fn = fn
        if isinstance(fn, partial):
            self.__name__ = fn.func.__name__
        else:
            self.__name__ = fn.__name__

    @debug_logger
    def __call__(self, *args, **kwargs):
        jaxpr, _, out_tree = make_jaxpr2(
            self.fn,
            static_argnums=kwargs.pop("static_argnums", ()),
            debug_info=kwargs.pop("debug_info", jdb("Function", self.fn, args, kwargs)),
        )(*args, **kwargs)

        def _eval_jaxpr(*args, **kwargs):
            return jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args, **kwargs)

        args, _ = jax.tree_util.tree_flatten((args, kwargs))
        retval = func_p.bind(
            wrap_init(_eval_jaxpr, debug_info=jaxpr.jaxpr.debug_info), *args, fn=self.fn
        )
        return tree_unflatten(out_tree, retval)


KNOWN_NAMED_OBS = (qml.Identity, qml.PauliX, qml.PauliY, qml.PauliZ, qml.Hadamard)

# Take care when adding primitives to this set in order to avoid introducing a quadratic number of
# edges to the jaxpr equation graph in ``sort_eqns()``. Each equation with a primitive in this set
# is constrained to occur before all subsequent equations in the quantum operations trace.
FORCED_ORDER_PRIMITIVES = {device_init_p, gphase_p, set_basis_state_p, set_state_p}

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
        with EvaluationContext.frame_tracing_context() as trace:
            in_tracers = _input_type_to_tracers(
                partial(trace.new_arg, source_info=current_source_info()), jaxpr.in_avals
            )
            out_tracers = [
                trace.to_jaxpr_tracer(t, source_info=current_source_info())
                for t in eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *in_tracers)
            ]
            out_tracers_, target_types_ = (
                (out_tracers[:-1], target_types[:-1]) if with_qreg else (out_tracers, target_types)
            )
            out_promoted_tracers = [
                (convert_element_type(tr, ty) if get_aval(tr).dtype != ty else tr)
                for tr, ty in zip(out_tracers_, target_types_)
            ]
            jaxpr2, _, consts = ctx.frames[trace].to_jaxpr2(
                out_promoted_tracers + ([out_tracers[-1]] if with_qreg else [])
            )
    return ClosedJaxpr(jaxpr2, consts)


def _apply_result_type_conversion(
    jaxpr: ClosedJaxpr,
    consts: List[Any],
    target_types: List[ShapedArray],
    num_implicit_outputs: int,
) -> Tuple[List[Any], InputSignature, OutputSignature]:
    """Re-trace the ``jaxpr`` program and apply type conversion to its results. Return full
    information about the modified Jaxpr program. The jaxpr program is only allowed to take zero or
    one quantum register as an argument.

    Args:
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
            (convert_element_type(tr, ty) if get_aval(tr).dtype != ty else tr)
            for tr, ty in zip(out_tracers_, target_types_)
        ]
        return out_promoted_tracers[num_implicit_outputs:] + (
            [out_tracers[-1]] if with_qreg else []
        )

    expanded_tracers, in_sig, out_sig = trace_function(
        _fun, *args, expansion_strategy=cond_expansion_strategy(), debug_info=jaxpr.debug_info
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
def unify_convert_result_types(jaxprs, consts, nimplouts):
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
            j, a, promoted_types, num_implicit_outputs
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
        cached_tracers = {w for w in qrp.cache.keys() if isinstance(w, Tracer)}
        requested_tracers = {w for w in wires if isinstance(w, Tracer)}
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


cached_vars = weakref.WeakKeyDictionary()


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

    binder: Callable = _no_binder

    @debug_logger_init
    def __init__(
        self,
        in_classical_tracers,
        out_classical_tracers,
        regions: List[HybridOpRegion],
        apply_reverse_transform=False,
        expansion_strategy=None,
        debug_info=None,
    ):  # pylint: disable=too-many-arguments
        self.in_classical_tracers = in_classical_tracers
        self.out_classical_tracers = out_classical_tracers
        self.regions: List[HybridOpRegion] = regions
        self.expansion_strategy = expansion_strategy
        self.apply_reverse_transform = apply_reverse_transform
        self.debug_info = debug_info
        super().__init__(wires=Wires(()))

    def __repr__(self):
        """Constructor-call-like representation."""
        nested_ops = [r.quantum_tape.operations for r in self.regions if r.quantum_tape]
        return f"{self.name}(tapes={nested_ops})"

    @debug_logger
    def bind_overwrite_classical_tracers(
        self,
        _trace: DynamicJaxprTrace,
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

        # Here, we are binding any of the possible hybrid ops.
        # which includes: for_loop, while_loop, cond, measure.
        # This will place an equation at the very end of the current list of equations.
        out_quantum_tracer = self.binder(*in_expanded_tracers, **kwargs)[-1]

        trace = EvaluationContext.get_current_trace()
        eqn = _get_eqn_from_tracing_eqn(trace.frame.tracing_eqns[-1])
        frame = trace.frame

        assert len(eqn.outvars[:-1]) == len(
            out_expanded_tracers
        ), f"{eqn.outvars=}\n{out_expanded_tracers=}"

        jaxpr_variables = cached_vars.get(frame, set())
        if not jaxpr_variables:
            # We get all variables in the current frame
            outvars = itertools.chain.from_iterable(
                [_get_eqn_from_tracing_eqn(e).outvars for e in frame.tracing_eqns]
            )
            jaxpr_variables = set(outvars)
            jaxpr_variables.update(frame.invars)
            jaxpr_variables.update(frame.constvar_to_val.keys())
            cached_vars[frame] = jaxpr_variables

        last_eqn = _get_eqn_from_tracing_eqn(frame.tracing_eqns[-1])
        for outvar in last_eqn.outvars:
            # With the exception of the output variables from the current equation.
            jaxpr_variables.discard(outvar)

        for i, t in enumerate(out_expanded_tracers):
            # We look for what were the previous output tracers.
            # If they haven't changed, then we leave them unchanged.
            if t.val in jaxpr_variables:
                continue

            # For other hybrid ops, use the original logic
            # If the variable cannot be found in the current frame
            # it is because we have created it via new_inner_tracer
            # which uses JAX internals to create a tracer without associating
            # a particular variable nor binding it to an equation.
            # So at this moment, we create a new variable for it
            # and set it as the eqn.outvars.
            # For example catalyst.measure always returns a tracer associated with
            # the value true but without having it being bound to any primitive
            #
            #    m = new_inner_tracer(ctx.trace, get_aval(True))
            #
            # Then we would land here and bound that inner_trace to this primitive
            #
            #    a: bool, b: qubit = measure_p.bind()
            #
            # At the moment, it is a little bit unclear why we need this.
            # I think it may be due to the separation between tracing the quantum
            # part from the classical part.
            #
            # First we would have traced the classical part (which includes hybrid ops)
            # and we need tracers for those values.
            # Here we would be tracing the quantum part, and it is where we would finally
            # bind it with the correct primitive.
            #
            # This seems to be the case since the measure function does not bound.
            # But it returns this inner tracer.
            #
            # Then the topological sorter would sort the operations that depended
            # on the original inner_trace accordingly.
            #
            # This method only gets called in a for loop for quantum operations
            #
            #     for op in ops:
            #       qrp2 = None
            #         if isinstance(op, HybridOp):
            #            // ... snip...
            #            qrp2 = op.trace_quantum(ctx, device, trace, qrp, **kwargs)
            #
            # So it should be safe to cache the tracers as we are doing it.
            eqn.outvars[i] = t.val

        # Now, the output variables can be considered as part of the current frame.
        # This allows us to avoid importing all equations again next time.
        jaxpr_variables.update(eqn.outvars)
        return out_quantum_tracer

    @debug_logger
    def trace_quantum(
        self,
        ctx,
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


# pylint: disable=too-many-arguments
@debug_logger
def trace_to_jaxpr(func, static_argnums, abstracted_axes, args, kwargs, debug_info=None):
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

    with transient_jax_config(
        {"jax_dynamic_shapes": True, "jax_use_shardy_partitioner": False}
    ), Patcher(
        (pe, "_drop_unused_vars", patched_drop_unused_vars),
        (DynamicJaxprTrace, "make_eqn", patched_make_eqn),
        (lax, "_dyn_shape_staging_rule", patched_dyn_shape_staging_rule),
        (
            jax._src.pjit,  # pylint: disable=protected-access
            "pjit_staging_rule",
            patched_pjit_staging_rule,
        ),
        (DictPatchWrapper(pe.custom_staging_rules, jit_p), "value", patched_pjit_staging_rule),
    ):
        make_jaxpr_kwargs = {
            "static_argnums": static_argnums,
            "abstracted_axes": abstracted_axes,
            "debug_info": debug_info,
        }
        with EvaluationContext(EvaluationMode.CLASSICAL_COMPILATION):
            jaxpr, out_type, out_treedef = make_jaxpr2(func, **make_jaxpr_kwargs)(*args, **kwargs)
            plugins = EvaluationContext.get_plugins()

    return jaxpr, out_type, out_treedef, plugins


@debug_logger
def lower_jaxpr_to_mlir(jaxpr, func_name, arg_names):
    """Lower a JAXPR to MLIR.

    Args:
        ClosedJaxpr: the JAXPR to lower to MLIR
        func_name: a name to use for the MLIR function
        arg_names: list of parameter names for the MLIR function

    Returns:
        ir.Module: the MLIR module coontaining the JAX program
        ir.Context: the MLIR context
    """

    MemrefCallable.clearcache()

    # Apply JAX 0.7.0 compatibility patches during MLIR lowering.
    # JAX internally calls trace_to_jaxpr_dynamic2 during lowering of nested @jit primitives
    # (e.g., in jax.scipy.linalg.expm and jax.scipy.linalg.solve), which triggers two bugs:
    # 1. make_eqn signature changed to include out_tracers parameter
    # 2. pjit_staging_rule creates JaxprEqn instead of TracingEqn
    #   (AssertionError at partial_eval.py:1790)
    with transient_jax_config(
        {"jax_dynamic_shapes": True, "jax_use_shardy_partitioner": False}
    ), Patcher(
        # Fix make_eqn signature change (handles both old/new JAX versions)
        (DynamicJaxprTrace, "make_eqn", patched_make_eqn),
        # Fix pjit_staging_rule creating wrong equation type
        (DictPatchWrapper(pe.custom_staging_rules, jit_p), "value", patched_pjit_staging_rule),
    ):
        mlir_module, ctx = jaxpr_to_mlir(jaxpr, func_name, arg_names)

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


def trace_snapshot_op(
    op: Operation,
    device: QubitDevice,
    qrp: QRegPromise,
    out_snapshot_tracer: List[DynamicJaxprTracer],
) -> None:
    """Trace StateMP passed inside qml.Snapshot
    Args:
        op: qml.Snapshot being traced.
        device: PennyLane quantum device.
        qrp: QRegPromise object holding the JAX tracer representing the quantum register's state.
        out_snapshot_tracer: list to store JAX classical qnode snapshot results.

    """
    if type(op.hyperparameters["measurement"]) == StateMP:
        nqubits = (
            device.wires[0]
            if catalyst.device.qjit_device.is_dynamic_wires(device.wires)
            else len(device.wires)
        )
        if isinstance(nqubits, DynamicJaxprTracer):
            shape = (jnp.left_shift(1, nqubits),)
        else:
            shape = (2**nqubits,)
        qreg_out = qrp.actualize()
        obs_tracers = compbasis_p.bind(qreg_out, qreg_available=True)
        dyn_dims, static_shape = _extract_tracers_dyn_shape(shape)
        result = state_p.bind(obs_tracers, *dyn_dims, static_shape=tuple(static_shape))
        out_snapshot_tracer.append(result)
    else:
        raise NotImplementedError(
            "qml.Snapshot() only supports qml.state() when used from within Catalyst,"
            f" but encountered {type(op.hyperparameters['measurement'])}"
        )


# pylint: disable=too-many-arguments,too-many-statements
@debug_logger
def trace_quantum_operations(
    quantum_tape: QuantumTape,
    device: QubitDevice,
    qreg: DynamicJaxprTracer,
    ctx,
    trace: DynamicJaxprTrace,
    mcm_config: qml.devices.MCMConfig = qml.devices.MCMConfig(),
    out_snapshot_tracer=None,
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
        out_snapshot_tracer: modified list to store JAX classical qnode snapshot results.

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
            qubits2 = unitary_p.bind(
                *[*op.parameters, *qubits, *controlled_qubits, *controlled_values],
                qubits_len=len(qubits),
                ctrl_len=len(controlled_qubits),
                adjoint=adjoint,
            )
            qrp.insert(op.wires, qubits2[: len(qubits)])
            qrp.insert(controlled_wires, qubits2[len(qubits) :])
        elif isinstance(op, qml.GlobalPhase):
            controlled_qubits = qrp.extract(controlled_wires)
            qubits2 = gphase_p.bind(
                *[*op.parameters, *controlled_qubits, *controlled_values],
                ctrl_len=len(controlled_qubits),
                adjoint=adjoint,
            )
            qrp.insert(controlled_wires, qubits2)
        elif isinstance(op, qml.StatePrep):
            trace_state_prep(op, qrp)
        elif isinstance(op, qml.BasisState):
            trace_basis_state(op, qrp)
        elif isinstance(op, qml.Snapshot):
            trace_snapshot_op(op, device, qrp, out_snapshot_tracer)
        else:
            qubits = qrp.extract(op.wires)
            controlled_qubits = qrp.extract(controlled_wires)

            # This is a temporary workaround for the PCPhase operation
            # which does not follow the same pattern as `qinst_p`.
            # We will revisit this once we have a better solution for
            # supporting general PL operations in Catalyst.
            params = (
                op.parameters + [op.hyperparameters["dimension"][0]]
                if isinstance(op, qml.PCPhase)
                else op.parameters
            )

            qubits2 = qinst_p.bind(
                *[*qubits, *params, *controlled_qubits, *controlled_values],
                op=op.name,
                qubits_len=len(qubits),
                params_len=len(params),
                ctrl_len=len(controlled_qubits),
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
    trace = EvaluationContext.get_current_trace()
    trace.frame.tracing_eqns = sort_eqns(trace.frame.tracing_eqns, FORCED_ORDER_PRIMITIVES)  # [1]
    return qrp


@debug_logger
def trace_observables(
    obs: Optional[Operator],
    qrp: QRegPromise,
    m_wires: Optional[qml.wires.Wires],
) -> Tuple[List[DynamicJaxprTracer], Optional[int]]:
    """Trace observables.

    Args:
        obs (Operator): an observable operator
        qrp (QRegPromise): Quantum register tracer with cached qubits
        m_wires (Optional[qml.wires.Wires]): the default wires to use for this measurement process

    Returns:
        out_classical_tracers: a list of classical tracers corresponding to the measured values.
        nqubits: number of actually measured qubits.
    """
    wires = obs.wires if (obs and len(obs.wires) > 0) else m_wires
    qubits = None
    if obs is None:
        if wires is None:
            # If measuring all wires on the device, pass in the qreg to compbasis op
            # TODO: "all wires on the device" is None when number of wires is static,
            # but a tracer when dynamic. Update to handle dynamic case.
            qreg_out = qrp.actualize()
            obs_tracers = compbasis_p.bind(qreg_out, qreg_available=True)
        else:
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
    return obs_tracers, (len(qubits) if qubits else None)


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


# pylint: disable=too-many-statements,too-many-branches, too-many-positional-arguments
@debug_logger
def trace_quantum_measurements(
    shots_obj,
    device: QubitDevice,
    qrp: QRegPromise,
    outputs: List[Union[MeasurementProcess, DynamicJaxprTracer, Any]],
    out_tree: PyTreeDef,
    mcm_config: qml.devices.MCMConfig,
) -> Tuple[List[DynamicJaxprTracer], PyTreeDef]:
    """Trace quantum measurements. Accept a list of QNode ouptputs and its Pytree-shape. Process
    the quantum measurement outputs, leave other outputs as-is.

    Args:
        shots_obj: Shots object containing shots information (total_shots, shot_vector, num_copies).
                  The total_shots can be an int or JAX tracer for dynamic shots.
        device (QubitDevice): PennyLane quantum device to use for quantum measurements.
        qrp (QRegPromise): Quantum register tracer with cached qubits
        outputs (List of quantum function results): List of qnode output JAX tracers to process.
        out_tree (PyTreeDef): PyTree-shape of the outputs.

    Returns:
        out_classical_tracers: modified list of JAX classical qnode ouput tracers.
        out_tree: modified PyTree-shape of the qnode output.
    """
    # Extract total shots and detect if dynamic
    shots = shots_obj.total_shots
    out_classical_tracers = []

    for i, output in enumerate(outputs):
        if isinstance(output, MeasurementProcess):

            # Check if the measurement is supported shot-vector where num_of_total_copies > 1
            if shots_obj.has_partitioned_shots and not isinstance(output, SampleMP):
                raise NotImplementedError(
                    f"Measurement {type(output).__name__} does not support shot-vectors. "
                    "Use qml.sample() instead."
                )

            if device.wires is None:
                d_wires = num_qubits_p.bind()
            elif catalyst.device.qjit_device.is_dynamic_wires(device.wires):
                d_wires = num_qubits_p.bind()
            else:
                d_wires = len(device.wires)

            m_wires = output.wires if output.wires else None
            obs_tracers, nqubits = trace_observables(output.obs, qrp, m_wires)
            nqubits = d_wires if nqubits is None else nqubits

            using_compbasis = obs_tracers.primitive == compbasis_p

            if (
                mcm_config.mcm_method == "single-branch-statistics"
                and output.mv is not None
                and type(output) in [ExpectationMP, VarianceMP, ProbabilityMP, CountsMP]
            ):
                raise NotImplementedError(
                    "single-branch-statistics does not support measurement processes "
                    "(expval, var, probs, counts) on mid circuit measurements."
                )

            if isinstance(output, SampleMP):

                if shots is None:  # needed for old device API only
                    raise ValueError(
                        "qml.sample cannot work with shots=None. "
                        "Please specify a finite number of shots."
                    )
                if output.mv is not None:  # qml.sample(m)
                    out_classical_tracers.append(output.mv)
                else:
                    shape = (shots, nqubits)
                    dyn_dims, static_shape = _extract_tracers_dyn_shape(shape)
                    result = sample_p.bind(obs_tracers, *dyn_dims, static_shape=tuple(static_shape))
                    if using_compbasis:
                        result = jnp.astype(result, jnp.int64)

                    reshaped_result = ()
                    shot_vector = shots_obj.shot_vector
                    start_idx = 0  # Start index for slicing
                    has_shot_vector = len(shot_vector) > 1 or any(
                        copies > 1 for _, copies in shot_vector
                    )
                    if has_shot_vector:
                        for shot, copies in shot_vector:
                            for _ in range(copies):
                                sliced_result = jax.lax.dynamic_slice(
                                    result, (start_idx, 0), (shot, nqubits)
                                )
                                reshaped_result += (sliced_result.reshape(shot, nqubits),)
                                start_idx += shot
                    else:
                        reshaped_result += (result,)

                    if len(reshaped_result) == 1:
                        reshaped_result = reshaped_result[0]

                    out_classical_tracers.append(reshaped_result)

            elif type(output) is ExpectationMP:
                out_classical_tracers.append(expval_p.bind(obs_tracers))
            elif type(output) is VarianceMP:
                out_classical_tracers.append(var_p.bind(obs_tracers))
            elif type(output) is ProbabilityMP:
                assert using_compbasis
                if isinstance(nqubits, DynamicJaxprTracer):
                    shape = (jnp.left_shift(1, nqubits),)
                else:
                    shape = (2**nqubits,)
                dyn_dims, static_shape = _extract_tracers_dyn_shape(shape)
                result = probs_p.bind(obs_tracers, *dyn_dims, static_shape=tuple(static_shape))
                out_classical_tracers.append(result)
            elif type(output) is CountsMP:
                if shots is None:  # needed for old device API only
                    raise ValueError(
                        "qml.counts cannot work with shots=None. "
                        "Please specify a finite number of shots."
                    )

                if using_compbasis:
                    if isinstance(nqubits, DynamicJaxprTracer):
                        shape = (jnp.left_shift(1, nqubits),)
                    else:
                        shape = (2**nqubits,)
                else:
                    shape = (2,)

                dyn_dims, static_shape = _extract_tracers_dyn_shape(shape)
                results = counts_p.bind(obs_tracers, *dyn_dims, static_shape=tuple(static_shape))

                if using_compbasis:
                    results = (jnp.asarray(results[0], jnp.int64), results[1])
                out_classical_tracers.extend(results)
                counts_tree = tree_structure(("keys", "counts"))
                meas_return_trees_children = out_tree.children()
                if len(meas_return_trees_children):
                    meas_return_trees_children[i] = counts_tree

                    out_tree = make_from_node_data_and_children(
                        out_tree.node_data(), meas_return_trees_children
                    )

                else:
                    out_tree = counts_tree
            elif type(output) is StateMP:
                assert using_compbasis
                if isinstance(nqubits, DynamicJaxprTracer):
                    shape = (jnp.left_shift(1, nqubits),)
                else:
                    shape = (2**nqubits,)
                dyn_dims, static_shape = _extract_tracers_dyn_shape(shape)
                result = state_p.bind(obs_tracers, *dyn_dims, static_shape=tuple(static_shape))
                out_classical_tracers.append(result)
            else:
                raise NotImplementedError(
                    f"Measurement {type(output)} is not implemented"
                )  # pragma: no cover
        elif isinstance(output, DynamicJaxprTracer):
            out_classical_tracers.append(output)
        else:
            assert not isinstance(
                output, (list, dict)
            ), f"Expected a tracer or a measurement, got {output}"
            out_classical_tracers.append(output)

    return out_classical_tracers, out_tree


@debug_logger
def has_classical_outputs(flat_results):
    """Checks if the quantum function outputs classical values.

    Returns:
        bool
    """
    for result in flat_results:
        if not isinstance(result, MeasurementProcess):
            return True

    return False


@debug_logger
def has_midcircuit_measurement(tape):
    """Check if the tape contains any mid-circuit measurements."""

    return any(
        map(lambda op: isinstance(op, catalyst.api_extensions.MidCircuitMeasure), tape.operations)
    )


@debug_logger
def num_midcircuit_measurement(tape):
    """Number of mid-circuit measurements."""

    return sum(
        map(lambda op: isinstance(op, catalyst.api_extensions.MidCircuitMeasure), tape.operations)
    )


class TracingMode(Enum):
    """Enumerate the tracing modes supported by the quantum function tracer:

    DEFAULT - Allows mid-circuit measurements and returns function's results directly.
              Used when no transform is applied or when transform doesn't modify
              measurements or produce multiple tapes.

    TRANSFORM - Uses tape measurements instead of original function results.
                Prohibits mid-circuit measurements with multiple tapes and requires
                function to return only measurements (no classical results).
                Used when transform produces multiple tapes or modifies measurements.
    """

    DEFAULT = 0
    TRANSFORM = 1


@debug_logger
def have_measurements_changed(original_tape, modified_tape):
    """Check if the measurement has changed."""

    if len(original_tape.measurements) != len(modified_tape.measurements):
        return True

    # Copying tapes preserves object identity in the operation and measurement lists, due to
    # immutability. So we can compare identity directly. Equality comparisons are problematic
    # when the measurements contain tracers.
    for original_meas, modified_meas in zip(original_tape.measurements, modified_tape.measurements):
        if original_meas is not modified_meas:
            return True

    return False


@debug_logger
def apply_transforms(
    qnode_program,
    device_program,
    tape,
    flat_results,
):
    """Apply device and qnode transform programs."""
    # Some transforms use trainability as a basis for transforming.
    # See batch_params
    params = tape.get_parameters(trainable_only=False)
    tape.trainable_params = qml.math.get_trainable_indices(params)

    if qnode_program.is_informative:
        msg = "Catalyst does not support informative transforms."
        raise CompileError(msg)

    # Apply the transform
    total_program = qnode_program + device_program
    tapes, post_processing = total_program([tape])

    if len(tapes) > 1:
        if has_classical_outputs(flat_results) or has_midcircuit_measurement(tape):
            msg = (
                "Batch transforms are unsupported with MCMs or non-MeasurementProcess QNode "
                "outputs. The selected device, options, or applied QNode transforms, may be "
                "attempting to produce multiple tapes."
            )
            raise CompileError(msg)
        tracing_mode = TracingMode.TRANSFORM
    elif len(qnode_program) or have_measurements_changed(tape, tapes[0]):
        with_measurement_from_counts_or_samples = any(
            "measurements_from_counts" in (transform_str := str(getattr(qnode, "transform", "")))
            or "measurements_from_samples" in transform_str
            for qnode in qnode_program
        )

        if has_classical_outputs(flat_results) and with_measurement_from_counts_or_samples:
            msg = (
                "Transforming MeasurementProcesses is unsupported with non-MeasurementProcess "
                "QNode outputs. The selected device, options, or applied QNode transforms, may be "
                "attempting to transform MeasurementProcesses from the tape."
            )
            raise CompileError(msg)
        tracing_mode = TracingMode.TRANSFORM
    else:
        tracing_mode = TracingMode.DEFAULT

    return tapes, post_processing, tracing_mode


@debug_logger
def split_tracers_and_measurements(flat_values):
    """Return classical tracers and measurements"""
    classical = []
    measurements = []
    for flat_value in flat_values:
        if isinstance(flat_value, MeasurementProcess):
            measurements.append(flat_value)  # pragma: no cover
        else:
            # classical should remain empty for all valid cases at the moment.
            # This is because split_tracers_and_measurements is only called
            # when checking the validity of transforms. And transforms cannot
            # return any tracers.
            classical.append(flat_value)

    return classical, measurements


@debug_logger
def trace_post_processing(trace, post_processing: Callable, pp_args, debug_info=None):
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
    with EvaluationContext.frame_tracing_context(trace):
        # What is the input to the post_processing function?
        # The input to the post_processing function is going to be a list of values One for each
        # tape. The tracers are all flat in pp_args.

        # We need to deduce the type/shape/tree of the post_processing.
        wffa, _, _, out_tree_promise = deduce_avals(
            post_processing, (pp_args,), {}, debug_info=debug_info
        )

        # wffa will take as an input a flatten tracers.
        # After wffa is called, then the shape becomes available in out_tree_promise.
        in_tracers = [
            trace.to_jaxpr_tracer(t, source_info=current_source_info())
            for t in tree_flatten(pp_args)[0]
        ]
        out_tracers = [
            trace.to_jaxpr_tracer(t, source_info=current_source_info())
            for t in wffa.call_wrapped(*in_tracers)
        ]
        cur_trace = EvaluationContext.get_current_trace()
        jaxpr, out_type, consts = cur_trace.frame.to_jaxpr2(out_tracers, wffa.debug_info)
        closed_jaxpr = ClosedJaxpr(jaxpr, consts)
        return closed_jaxpr, out_type, out_tree_promise()


@debug_logger
def trace_function(
    fun: Callable, *args, expansion_strategy: ExpansionStrategy, **kwargs
) -> Tuple[List[Any], InputSignature, OutputSignature]:
    """Trace classical Python function containing no quantum computations. Arguments and results of
    the function are allowed to contain dynamic dimensions. Depending on the expansion strategy, the
    resulting Jaxpr program might or might not preserve sharing among the dynamic dimension
    variables. The support for expansion options makes this function different from
    `jax_extras.make_jaxpr2`.

    Args:
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
        fun,
        args,
        kwargs,
        expansion_strategy=expansion_strategy,
        debug_info=kwargs.pop("debug_info", None),
    )

    with EvaluationContext.frame_tracing_context(debug_info=wfun.debug_info) as trace:
        arg_expanded_tracers = input_type_to_tracers(
            in_sig.in_type,
            partial(trace.new_arg, source_info=current_source_info()),
            partial(trace.to_jaxpr_tracer, source_info=current_source_info()),
        )
        res_expanded_tracers = wfun.call_wrapped(*arg_expanded_tracers)

        return res_expanded_tracers, in_sig, out_sig


def _get_total_shots(qnode):
    """
    Extract total shots from qnode.
    If shots is None on the qnode, this method returns 0 (static).
    This method allows the qnode shots to be either static (python int
    literals) or dynamic (tracers).
    """
    # due to possibility of tracer, we cannot use a simple `or` here to simplify
    shots_value = qnode._shots.total_shots  # pylint: disable=protected-access
    if shots_value is None:
        shots = 0
    else:
        shots = shots_value
    return shots


def _construct_output_with_classical_values(
    tape: QuantumTape, return_values_flat: List[Any]
) -> Tuple[List[Any], List[int], int]:
    classical_values = []
    classical_return_indices = []
    num_mcm = num_midcircuit_measurement(tape)
    for i, value in enumerate(return_values_flat):
        if not isinstance(value, qml.measurements.MeasurementProcess):
            classical_values.append(value)
            classical_return_indices.append(i)
    output = classical_values + tape.measurements

    return output, classical_return_indices, num_mcm


def _trace_classical_phase(
    f: Callable,
    device: QubitDevice,
    args,
    kwargs,
    qnode,
    *,
    static_argnums,
    debug_info,
    ctx,
) -> ClassicalTraceResult:
    """Perform classical tracing phase of quantum function compilation.

    Args:
        f: Function to trace
        device: Quantum device
        args: Positional arguments
        kwargs: Keyword arguments
        qnode: Quantum node containing transforms
        static_argnums: Static argument numbers
        debug_info: Debug information
        ctx: Evaluation context

    Returns:
        ClassicalTraceResult: Results for quantum tracing phase
    """
    shots = qnode._shots  # pylint: disable=protected-access
    quantum_tape = QuantumTape(shots=shots)  # pylint: disable=protected-access
    with EvaluationContext.frame_tracing_context(debug_info=debug_info) as trace:
        wffa, in_avals, keep_inputs, out_tree_promise = deduce_avals(
            f, args, kwargs, static_argnums, debug_info
        )
        in_classical_tracers = _input_type_to_tracers(
            partial(trace.new_arg, source_info=current_source_info()), in_avals
        )
        with QueuingManager.stop_recording(), quantum_tape:
            # Quantum tape transformations happen at the end of tracing
            in_classical_tracers = [t for t, k in zip(in_classical_tracers, keep_inputs) if k]
            return_values_flat = wffa.call_wrapped(*in_classical_tracers)
        # Ans contains the leaves of the pytree (empty for measurement without
        # data https://github.com/PennyLaneAI/pennylane/pull/4607)
        # Therefore we need to compute the tree with measurements as leaves and it comes
        # with an extra computational cost

        if any(isinstance(wire, qml.wires.DynamicWire) for wire in quantum_tape.wires):
            msg = "qml.allocate() is only supported with program capture enabled."
            raise CompileError(msg)

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
            device_program, config = device.preprocess(ctx, execution_config=config, shots=shots)
        else:
            device_program = qml.CompilePipeline()

        qnode_program = qnode.transform_program if qnode else qml.CompilePipeline()

        tapes, post_processing, tracing_mode = apply_transforms(
            qnode_program,
            device_program,
            quantum_tape,
            return_values_flat,
        )

        return ClassicalTraceResult(
            trace=trace,
            tapes=tapes,
            post_processing=post_processing,
            tracing_mode=tracing_mode,
            return_values_flat=return_values_flat,
            return_values_tree=return_values_tree,
            is_leaf=is_leaf,
        )


def _get_qreg_in(qnode, device):
    total_shots = _get_total_shots(qnode)
    device_init_p.bind(
        total_shots,
        auto_qubit_management=(device.wires is None),
        rtd_lib=device.backend_lib,
        rtd_name=device.backend_name,
        rtd_kwargs=str(device.backend_kwargs),
    )
    if device.wires is None:
        # Automatic qubit management mode
        # We start with 0 wires and allocate new wires in runtime as we encounter them.
        qreg_in = qalloc_p.bind(0)
    elif catalyst.device.qjit_device.is_dynamic_wires(device.wires):
        # When device has dynamic wires, the device.wires iterable object
        # has a single value, which is the tracer for the number of wires
        qreg_in = qalloc_p.bind(device.wires[0])
    else:
        qreg_in = qalloc_p.bind(len(device.wires))
    return qreg_in


def _get_meas_results(trace, meas, meas_trees, snapshot_results):
    # Check if the measurements are nested then apply the to_jaxpr_tracer
    def check_full_raise(arr, func):
        if isinstance(arr, (list, tuple)):
            return type(arr)(check_full_raise(x, func) for x in arr)
        else:
            return func(arr)

    meas_tracers = check_full_raise(
        meas,
        partial(trace.to_jaxpr_tracer, source_info=current_source_info()),
    )
    if len(snapshot_results) > 0:
        # redefine meas_trees PyTree to have same PyTreeRegistry as Snapshot PyTree
        meas_trees = jax.tree_util.tree_structure(
            jax.tree_util.tree_unflatten(meas_trees, meas_tracers)
        )
        meas_trees = jax.tree_util.treedef_tuple([tree_structure(snapshot_results), meas_trees])
        meas_tracers = snapshot_results + meas_tracers
    return tree_unflatten(meas_trees, meas_tracers)


def _trace_quantum_step(
    device: QubitDevice,
    qnode,
    ctx,
    cls_result: ClassicalTraceResult,
    debug_info,  # pylint: disable=unused-argument
) -> Tuple[List[Any], List[int], int]:
    """Perform quantum tracing phase of quantum function compilation.

    This function handles the quantum tracing part, which includes:
    - Processing each quantum tape
    - Setting up quantum registers
    - Tracing quantum operations and measurements
    - Collecting results and handling snapshots

    Args:
        device: Quantum device
        qnode: Quantum node
        ctx: Evaluation context
        cls_result: Results from classical tracing phase
        debug_info: Debug information

    Returns:
        Tuple containing:
            - transformed_results: List of transformed measurement results
            - classical_return_indices: Indices of classical return values
            - num_mcm: Number of mid-circuit measurements
    """
    transformed_results = []
    classical_return_indices = []
    num_mcm = 0

    trace = cls_result.trace
    tapes = cls_result.tapes
    tracing_mode = cls_result.tracing_mode
    return_values_flat = cls_result.return_values_flat
    return_values_tree = cls_result.return_values_tree
    is_leaf = cls_result.is_leaf

    def process_tape_output(tape, return_values_flat, return_values_tree):
        cls_ret_indices = []
        num_mcm = 0
        if tracing_mode == TracingMode.TRANSFORM:
            # TODO: In the future support arbitrary output from the user function.
            if uses_transform(qnode, "dynamic_one_shot_partial"):
                output, cls_ret_indices, num_mcm = _construct_output_with_classical_values(
                    tape, return_values_flat
                )
            else:
                output = tape.measurements

            _, trees = jax.tree_util.tree_flatten(output, is_leaf=is_leaf)
        else:
            output = return_values_flat
            trees = return_values_tree

        return output, trees, cls_ret_indices, num_mcm

    with EvaluationContext.frame_tracing_context(trace):
        for tape in tapes:
            # Set up quantum register for the current tape.
            # We just need to ensure there is a tape cut in between each.
            # Each tape will be outlined into its own function with mlir passㄍ
            # -split-multiple-tapes
            qreg_in = _get_qreg_in(qnode, device)

            # If the program is batched, that means that it was transformed.
            # If it was transformed, that means that the program might have
            # changed the output. See `split_non_commuting`
            output, trees, cls_ret_indices, num_mcm = process_tape_output(
                tape, return_values_flat, return_values_tree
            )
            classical_return_indices.extend(cls_ret_indices)

            mcm_config = qml.devices.MCMConfig(
                postselect_mode=qnode.execute_kwargs["postselect_mode"],
                mcm_method=qnode.execute_kwargs["mcm_method"],
            )
            snapshot_results = []
            qrp_out = trace_quantum_operations(
                tape, device, qreg_in, ctx, trace, mcm_config, snapshot_results
            )
            shots = qnode._shots  # pylint: disable=protected-access
            meas, meas_trees = trace_quantum_measurements(
                shots, device, qrp_out, output, trees, mcm_config
            )
            qreg_out = qrp_out.actualize()

            # Get the measurement results
            meas_results = _get_meas_results(trace, meas, meas_trees, snapshot_results)

            # TODO: Allow the user to return whatever types they specify.
            if tracing_mode == TracingMode.TRANSFORM and isinstance(meas_results, list):
                if len(meas_results) == 1:
                    transformed_results.append(meas_results[0])
                else:
                    transformed_results.append(tuple(meas_results))
            else:
                transformed_results.append(meas_results)

            # Deallocate the register and release the device after the current tape is finished.
            qdealloc_p.bind(qreg_out)
            device_release_p.bind()

    return transformed_results, classical_return_indices, num_mcm


@debug_logger
def trace_quantum_function(
    f: Callable, device: QubitDevice, args, kwargs, qnode, static_argnums, debug_info
) -> TraceResult:
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
        static_argnums: Static argument numbers
        debug_info: Debug information

    Returns:
        TraceResult: A dataclass containing:
            - closed_jaxpr: JAXPR expression of the function ``f``
            - out_type: JAXPR output type (list of abstract values with explicitness flags)
            - out_tree: PyTree shape of the result
            - return_values_tree: PyTree structure of return values
            - classical_return_indices: Indices of classical return values
            - num_mcm: Number of mid-circuit measurements
    """
    with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
        # (1) - Classical tracing
        cls_result = _trace_classical_phase(
            f,
            device,
            args,
            kwargs,
            qnode,
            static_argnums=static_argnums,
            debug_info=debug_info,
            ctx=ctx,
        )

        # (2) - Quantum tracing
        transformed_results, classical_return_indices, num_mcm = _trace_quantum_step(
            device, qnode, ctx, cls_result, debug_info
        )

        # (3) - Post-processing
        closed_jaxpr, out_type, out_tree = trace_post_processing(
            cls_result.trace,
            cls_result.post_processing,
            transformed_results,
            debug_info,
        )
        # TODO: `check_jaxpr` complains about the `AbstractQreg` type. Consider fixing.
        # check_jaxpr(jaxpr)

    return TraceResult(
        closed_jaxpr=closed_jaxpr,
        out_type=out_type,
        out_tree=out_tree,
        return_values_tree=cls_result.return_values_tree,
        classical_return_indices=classical_return_indices,
        num_mcm=num_mcm,
    )
