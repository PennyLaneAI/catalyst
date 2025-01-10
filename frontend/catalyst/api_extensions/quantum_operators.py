# Copyright 2022-2024 Xanadu Quantum Technologies Inc.

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
This module contains public API functions for quantum operators which are not
included in PennyLane, or whose behaviour needs to be adapted for Catalyst.
"""

import copy
import sys
from collections.abc import Sized
from contextlib import nullcontext
from typing import Any, Callable, List, Optional, Union

import jax
import pennylane as qml
from jax._src.tree_util import tree_flatten
from jax.core import get_aval
from pennylane import QueuingManager
from pennylane.operation import Operator
from pennylane.ops.op_math.adjoint import create_adjoint_op
from pennylane.ops.op_math.controlled import create_controlled_op
from pennylane.tape import QuantumTape

from catalyst.api_extensions.control_flow import cond
from catalyst.jax_extras import (
    ClosedJaxpr,
    DynamicJaxprTracer,
    _input_type_to_tracers,
    convert_constvars_jaxpr,
    deduce_avals,
    new_inner_tracer,
)
from catalyst.jax_primitives import AbstractQreg, adjoint_p, qmeasure_p
from catalyst.jax_tracer import (
    HybridOp,
    HybridOpRegion,
    QRegPromise,
    has_nested_tapes,
    trace_quantum_operations,
)
from catalyst.tracing.contexts import EvaluationContext

pl_adjoint_module = sys.modules["pennylane.ops.op_math.adjoint"]
pl_ctrl_module = sys.modules["pennylane.ops.op_math.controlled"]


## API ##
def measure(
    wires, reset: Optional[bool] = False, postselect: Optional[int] = None
) -> DynamicJaxprTracer:
    r"""A :func:`qjit` compatible mid-circuit measurement on 1 qubit for PennyLane/Catalyst.

    .. important::

        The :func:`qml.measure() <pennylane.measure>` function is **not** QJIT
        compatible and :func:`catalyst.measure` from Catalyst should be used instead.

    Args:
        wires (int): The wire the projective measurement applies to.
        reset (Optional[bool]): Whether to reset the wire to the :math:`|0\rangle`
            state after measurement.
        postselect (Optional[int]): Which basis state to postselect after a mid-circuit measurement.

    Returns:
        A JAX tracer for the mid-circuit measurement.

    Raises:
        ValueError: Called outside the tape context.

    .. note::

        To configure the mid-circuit measurement simulation method and postselection
        mode, please see the relevant keyword arguments of the :func:`~.QNode`.

        For more details, check out our
        :doc:`dynamic quantum circuit guide <introduction/dynamic_quantum_circuits>`.

    **Example**

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=2)

        @qjit
        @qml.qnode(dev)
        def circuit(x: float):
            qml.RX(x, wires=0)
            m1 = measure(wires=0)

            qml.RX(m1 * jnp.pi, wires=1)
            m2 = measure(wires=1)

            qml.RZ(m2 * jnp.pi / 2, wires=0)
            return qml.expval(qml.PauliZ(0)), m2

    >>> circuit(0.43)
    [Array(1., dtype=float64), Array(False, dtype=bool)]
    >>> circuit(0.43)
    [Array(-1., dtype=float64), Array(True, dtype=bool)]

    **Example with post-selection**

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=1)

        @qjit
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            m = measure(0, postselect=1)
            return qml.expval(qml.PauliZ(0))

    >>> circuit()
    Array(-1., dtype=float64)

    **Example with reset**

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=1)

        @qjit
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            m = measure(0, reset=True)
            return qml.expval(qml.PauliZ(0))

    >>> circuit()
    Array(1., dtype=float64)
    """
    EvaluationContext.check_is_tracing("catalyst.measure can only be used from within @qjit.")
    EvaluationContext.check_is_quantum_tracing(
        "catalyst.measure can only be used from within a qml.qnode."
    )
    ctx = EvaluationContext.get_main_tracing_context()
    wires = list(wires) if isinstance(wires, (list, tuple)) else [wires]
    if len(wires) != 1:
        raise TypeError(f"Only one element is supported for the 'wires' parameter, got {wires}.")
    if isinstance(wires[0], jax.Array) and wires[0].shape not in ((), (1,)):
        raise TypeError(
            f"Measure is only supported on 1 qubit, got array of shape {wires[0].shape}."
        )

    # Copy, so wires remain unmodified
    in_classical_tracers = wires.copy()

    if postselect is not None and postselect not in [0, 1]:
        raise TypeError(f"postselect must be '0' or '1', got {postselect}")

    m = new_inner_tracer(ctx.trace, get_aval(True))
    MidCircuitMeasure(
        in_classical_tracers=in_classical_tracers,
        out_classical_tracers=[m],
        regions=[],
        reset=reset,
        postselect=postselect,
    )

    # If reset was requested, reset qubit only if the measurement result was 1
    if reset:

        @cond(m)
        def reset_fn():
            qml.PauliX(wires=wires)

        reset_fn()

    return m


def adjoint(f: Union[Callable, Operator], lazy=True) -> Union[Callable, Operator]:
    """A :func:`~.qjit` compatible adjoint transformer for PennyLane/Catalyst.

    Returns a quantum function or operator that applies the adjoint of the
    provided function or operator.

    .. warning::

        This function does not support performing the adjoint
        of quantum functions that contain mid-circuit measurements.

    Args:
        f (Callable or Operator): A PennyLane operation or a Python function
                                  containing PennyLane quantum operations.
        lazy (bool): Whether to delay the computation of the Hermitian conjugate until a later time
                     (typically during decomposition or compilation). The default is ``True``,
                     whereas ``False`` will immediately produce a new operator implementing the
                     adjoint. Note that ``False`` is only supported when the adjoint is applied to
                     a single Operator, rather than a quantum function.

    Returns:
        If an Operator is provided, returns an Operator that is the adjoint. If
        a function is provided, returns a function with the same call signature
        that returns the Adjoint of the provided function.

    Raises:
        ValueError: invalid parameter values

    **Example 1 (basic usage)**

    .. code-block:: python

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def workflow(theta, wires):
            catalyst.adjoint(qml.RZ)(theta, wires=wires)
            catalyst.adjoint(qml.RZ(theta, wires=wires))
            def func():
                qml.RX(theta, wires=wires)
                qml.RY(theta, wires=wires)
            catalyst.adjoint(func)()
            return qml.probs()

    >>> workflow(jnp.pi/2, wires=0)
    Array([0.5, 0.5], dtype=float64)

    **Example 2 (with Catalyst control flow)**

    .. code-block:: python

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def workflow(theta, n, wires):
            def func():
                @catalyst.for_loop(0, n, 1)
                def loop_fn(i):
                    qml.RX(theta, wires=wires)

                loop_fn()
            catalyst.adjoint(func)()
            return qml.probs()

    >>> workflow(jnp.pi/2, 3, 0)
    [1.00000000e+00 7.39557099e-32]
    """

    adj = AdjointCallable(f, lazy=lazy)

    # Return an instantiated version of the Adjoint class if we receive an operator instance.
    return adj() if isinstance(f, Operator) else adj


def ctrl(
    f: Union[Callable, Operator],
    control: List[Any],
    control_values: Optional[List[Any]] = None,
    work_wires: Optional[List[Any]] = None,
) -> Callable:
    """Create a method that applies a controlled version of the provided op. This function is the
    Catalyst version of the ``qml.ctrl`` that supports Catalyst hybrid operations such as loops and
    conditionals.

    Args:
        f (Callable or Operator): A PennyLane operation or a Python function
                                  containing PennyLane quantum operations.
        control (Wires): The control wire(s).
        control_values (List[bool], optional): The value(s) the control wire(s) should take.
            Integers other than 0 or 1 will be treated as ``int(bool(x))``.
        work_wires (Any): Any auxiliary wires that can be used in the decomposition

    Returns:
        (function or :class:`~.operation.Operator`): If an Operator is provided, returns a
        Controlled version of the Operator.  If a function is provided, returns a function with the
        same call signature that creates a controlled version of the provided function.

    Raises:
        ValueError: invalid parameter values, measurements are among the controlled operations.

    **Example**

    .. code-block:: python

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def workflow(theta, w, cw):
            qml.Hadamard(wires=[0])
            qml.Hadamard(wires=[1])

            def func(arg):
              qml.RX(theta, wires=arg)

            @cond(theta > 0.0)
            def cond_fn():
              qml.RY(theta, wires=w)

            catalyst.ctrl(func, control=[cw])(w)
            catalyst.ctrl(cond_fn, control=[cw])()
            catalyst.ctrl(qml.RZ, control=[cw])(theta, wires=w)
            catalyst.ctrl(qml.RY(theta, wires=w), control=[cw])
            return qml.probs()

    >>> workflow(jnp.pi/4, 1, 0)
    Array([0.25, 0.25, 0.03661165, 0.46338835], dtype=float64)
    """
    if control_values is not None and (
        (len(control) if isinstance(control, Sized) else 1)
        != (len(control_values) if isinstance(control_values, Sized) else 1)
    ):
        raise ValueError(
            f"Length of the control_values ({len(control_values)}) must be None or equal "
            f"to the lenght of control ({len(control)})"
        )

    res = CtrlCallable(f, control, control_values=control_values, work_wires=work_wires)
    return res() if isinstance(f, Operator) else res


## IMPL ##
class MidCircuitMeasure(HybridOp):
    """Operation representing a mid-circuit measurement."""

    binder = qmeasure_p.bind

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        in_classical_tracers,
        out_classical_tracers,
        regions: List[HybridOpRegion],
        reset: bool = None,
        postselect: int = None,
    ):
        HybridOp.__init__(self, in_classical_tracers, out_classical_tracers, regions)
        self._wires = qml.wires.Wires(in_classical_tracers)
        self.reset = reset
        self.postselect = postselect

    # pylint: disable=too-many-arguments
    def trace_quantum(self, ctx, device, trace, qrp, postselect_mode=None) -> QRegPromise:
        qubit = qrp.extract(self.wires)[0]
        if postselect_mode == "hw-like":
            qubit2 = self.bind_overwrite_classical_tracers(
                ctx,
                trace,
                in_expanded_tracers=[qubit],
                out_expanded_tracers=self.out_classical_tracers,
            )
        else:
            qubit2 = self.bind_overwrite_classical_tracers(
                ctx,
                trace,
                in_expanded_tracers=[qubit],
                out_expanded_tracers=self.out_classical_tracers,
                postselect=self.postselect,
            )
        qrp.insert(self.wires, [qubit2])
        return qrp

    def __hash__(self):
        hsh = super().__hash__()
        return hash(hsh + hash(self.out_classical_tracers[0]))


class AdjointCallable:
    """Callable wrapper to produce an adjoint instance."""

    def __init__(self, target, lazy):
        self.target = target
        self.lazy = lazy

        if isinstance(target, Operator):
            # Case 1: User passed an already instantiated operation, e.g. adjoint(qml.Hadamard(0))
            self.single_op = True
            self.instantiated = True
        elif isinstance(target, type) and issubclass(target, Operator):
            # Case 2: User passed the constructor of an operation, e.g. adjoint(qml.Hadamard)(0)
            self.single_op = True
            self.instantiated = False
        elif isinstance(target, Callable):
            # Case 3: User passed an arbitrary callable that will instantiate operations.
            # We want to create a callable that will generate an "opaque" Adjoint object.
            # This object differs from the Adjoint in PennyLane because that one can only be
            # instantiated on single operations.
            self.single_op = False
            self.instantiated = False
        else:
            raise ValueError(f"Expected a callable or a qml.Operator, not {target}")

        if not self.lazy and not self.single_op:
            # Supporting this for a qfunc is technically possible by invoking the decomposition on
            # HybridAdjoint with laza=False, however one would need to outline the classical jaxpr
            # into the outer scope, and possibly re-mapping tracers in the quantum operators,
            # in order to avoid escaped tracers which indicate an invalid program.
            raise ValueError(
                "Eagerly computing the adjoint (lazy=False) is only supported on single operators."
            )

    def __call__(self, *args, **kwargs):
        if self.single_op:
            base_op = self.target if self.instantiated else self.target(*args, **kwargs)
            return create_adjoint_op(base_op, self.lazy)

        tracing_artifacts = self.trace_body(args, kwargs)

        return HybridAdjoint(*tracing_artifacts)

    def trace_body(self, args, kwargs):
        """Generate a HybridOpRegion for use by Catalyst."""

        # Allow the creation of HybridAdjoint instances outside of any contexts.
        # Don't create a JAX context here as otherwise we could be dealing with escaped tracers.
        if not EvaluationContext.is_tracing():
            # QuantumTapes can themselves appear in queuing records.
            with QueuingManager.stop_recording(), QuantumTape() as quantum_tape:
                self.target(*args, **kwargs)

            adjoint_region = HybridOpRegion(None, quantum_tape, [], [])

            return [], [], [adjoint_region]

        # Create a nested jaxpr scope for the body of the adjoint.
        ctx = EvaluationContext.get_main_tracing_context()
        with EvaluationContext.frame_tracing_context(ctx) as inner_trace:
            in_classical_tracers, _ = tree_flatten((args, kwargs))
            wffa, in_avals, _, _ = deduce_avals(self.target, args, kwargs)
            arg_classical_tracers = _input_type_to_tracers(inner_trace.new_arg, in_avals)
            with QueuingManager.stop_recording(), QuantumTape() as quantum_tape:
                # FIXME: move all full_raise calls into a separate function
                res_classical_tracers = [
                    inner_trace.full_raise(t)
                    for t in wffa.call_wrapped(*arg_classical_tracers)
                    if isinstance(t, DynamicJaxprTracer)
                ]

            _check_no_measurements(quantum_tape)

            adjoint_region = HybridOpRegion(
                inner_trace, quantum_tape, arg_classical_tracers, res_classical_tracers
            )

        return in_classical_tracers, [], [adjoint_region]


class HybridAdjoint(HybridOp):
    """This class provides Catalyst-specific adjoint functionality, including tracing to a JAX
    primitive and managing the nested scope/tape, while also being a PennyLane operation itself
    than can be queued in a quantum context."""

    binder = adjoint_p.bind

    def trace_quantum(self, ctx, device, _trace, qrp) -> QRegPromise:
        op = self
        body_trace = op.regions[0].trace
        body_tape = op.regions[0].quantum_tape
        res_classical_tracers = op.regions[0].res_classical_tracers

        # Handle ops that were instantiated outside of a tracing context.
        if body_trace is None:
            frame_ctx = EvaluationContext.frame_tracing_context(ctx)
        else:
            frame_ctx = EvaluationContext.frame_tracing_context(ctx, body_trace)

        with frame_ctx as body_trace:
            qreg_in = _input_type_to_tracers(body_trace.new_arg, [AbstractQreg()])[0]
            qrp_out = trace_quantum_operations(body_tape, device, qreg_in, ctx, body_trace)
            qreg_out = qrp_out.actualize()
            body_jaxpr, _, body_consts = ctx.frames[body_trace].to_jaxpr2(
                res_classical_tracers + [qreg_out]
            )

        qreg = qrp.actualize()
        args, args_tree = tree_flatten((body_consts, op.in_classical_tracers, [qreg]))
        op_results = adjoint_p.bind(
            *args,
            args_tree=args_tree,
            jaxpr=ClosedJaxpr(convert_constvars_jaxpr(body_jaxpr), ()),
        )
        qrp2 = QRegPromise(op_results[-1])
        return qrp2

    @property
    def wires(self):
        """The list of all static wires."""

        assert len(self.regions) == 1, "Adjoint is expected to have one region"
        total_wires = sum((op.wires for op in self.regions[0].quantum_tape.operations), [])
        return total_wires

    def decomposition(self):
        """Resolve the Adjoint region by propagating the adjoint modifier to nested operations
        in reverse. Note that decomposition of control flow ops like for loops are only supported
        in the compiler."""
        assert len(self.regions) == 1, "Expected a single nested region for HybridAdjoint"

        return [
            create_adjoint_op(op, lazy=True)
            for op in reversed(self.regions[0].quantum_tape.operations)
        ]


class CtrlCallable:
    """Callable wrapper to produce a ctrl instance."""

    def __init__(self, target, control, control_values, work_wires):
        self.target = target
        self.control_wires = control
        self.control_values = control_values
        self.work_wires = work_wires

        if isinstance(target, Operator):
            # Case 1. Support an initialized operation as the base target
            self.single_op = True
            self.instantiated = True
        elif isinstance(target, type) and issubclass(target, Operator):
            # Case 2: Support an operation constructor as the base op
            self.single_op = True
            self.instantiated = False
        elif isinstance(target, Callable):
            # Case 3: Support a callable as the base op
            self.single_op = False
            self.instantiated = False
        else:
            raise ValueError(f"Expected a callable or a qml.Operator, not {target}")

    def __call__(self, *args, **kwargs):
        if self.single_op:
            base_op = self.target if self.instantiated else self.target(*args, **kwargs)
            return create_controlled_op(
                base_op, self.control_wires, self.control_values, self.work_wires
            )

        tracing_artifacts = self.trace_body(args, kwargs)

        return HybridCtrl(
            *tracing_artifacts,
            control_wires=self.control_wires,
            control_values=self.control_values,
            work_wires=self.work_wires,
        )

    def trace_body(self, args, kwargs):
        """Generate a HybridOpRegion for `catalyst.ctrl` to be used by the tracer."""

        # Allow the creation of HybridCtrl instances outside of any contexts.
        # Don't create a JAX context here as otherwise we could be dealing with escaped tracers.
        if not EvaluationContext.is_tracing():
            # QuantumTapes can themselves appear in queuing records.
            with QueuingManager.stop_recording(), QuantumTape() as quantum_tape:
                self.target(*args, **kwargs)

            ctrl_region = HybridOpRegion(None, quantum_tape, [], [])

            return [], [], [ctrl_region]

        # Create a nested jaxpr scope for the body of the adjoint.
        in_classical_tracers, _ = tree_flatten((args, kwargs))
        with QueuingManager.stop_recording(), QuantumTape() as quantum_tape:
            res = self.target(*args, **kwargs)
        out_classical_tracers, _ = tree_flatten(res)

        _check_no_measurements(quantum_tape)
        ctrl_region = HybridOpRegion(None, quantum_tape, [], [])

        return in_classical_tracers, out_classical_tracers, [ctrl_region]


class HybridCtrl(HybridOp):
    """Catalyst quantum ctrl operation support for both operations and callables"""

    def __init__(self, *tracing_artifacts, control_wires, control_values=None, work_wires=None):
        self._control_wires = qml.wires.Wires(control_wires)
        self._work_wires = qml.wires.Wires([] if work_wires is None else work_wires)
        if control_values is None:
            self._control_values = [True] * len(self._control_wires)
        elif isinstance(control_values, (int, bool)):
            self._control_values = [control_values]
        else:
            self._control_values = control_values
        super().__init__(*tracing_artifacts)

    def trace_quantum(self, ctx, device, trace, qrp) -> QRegPromise:
        raise NotImplementedError(
            "HybridCtrl does not support JAX quantum tracing"
        )  # pragma: no cover

    def decomposition(self):
        """Compute quantum decomposition of the gate by recursively scanning the nested tape and
        distributing the quantum control operaiton over the tape operations."""
        assert len(self.regions) == 1, "HybridCtrl is expected to have one region"

        _check_no_measurements(self.regions[0].quantum_tape)

        return ctrl_distribute(
            self.regions[0].quantum_tape,
            self._control_wires,
            self._control_values,
            self._work_wires,
        )

    @property
    def wires(self):
        """The list of all control-wires and active-wires."""
        assert len(self.regions) == 1, "HybridCtrl is expected to have one region"

        total_wires = sum(
            (op.wires for op in self.regions[0].quantum_tape.operations),
            self._control_wires,
        )
        return total_wires

    @property
    def control_wires(self):
        """Wires used in quantum conditioning."""
        return self._control_wires

    @property
    def control_values(self):
        """(Boolean) Values upon which to condition on."""
        return self._control_values

    @property
    def work_wires(self):
        """Optional wires that can be used in the expansion of this op."""
        return self._work_wires

    def map_wires(self, wire_map):
        """Map wires to new wires according to wire_map"""
        new_ops = []
        for op in self.regions[0].quantum_tape.operations:
            new_ops.append(op.map_wires(wire_map))
        self.regions[0].quantum_tape = QuantumTape(new_ops, [])
        self._control_wires = [wire_map.get(wire, wire) for wire in self._control_wires]
        self._work_wires = [wire_map.get(wire, wire) for wire in self._work_wires]
        return self


def ctrl_distribute(
    tape: QuantumTape,
    control_wires: List[Any],
    control_values: List[Any],
    work_wires: Optional[List[Any]] = None,
) -> QuantumTape:
    """Distribute the quantum control operation, described by ``control_wires`` and
    ``control_values``, over all the operations on the nested quantum tape.
    """
    # Note: The transformation modifies operations in the source quantum tape, so we must not use it
    # after we called this function.
    assert len(control_wires) > 0, "This transformation expects a non-empty list of control_wires"
    assert len(control_wires) == len(control_values), (
        f"Length of the control_values ({len(control_values)}) must be equal "
        f"to the lenght of control_wires ({len(control_wires)})"
    )

    # Allow decompositions outside of a Catalyst context.
    if EvaluationContext.is_tracing():
        ctx = EvaluationContext.get_main_tracing_context()
    else:
        ctx = None

    new_ops = []
    for op in tape.operations:
        if has_nested_tapes(op):
            if isinstance(op, HybridCtrl):
                nested_ops = ctrl_distribute(
                    op.regions[0].quantum_tape,
                    control_wires + op.control_wires,
                    control_values + op.control_values,
                    work_wires + op.work_wires,
                )
                new_ops.extend(nested_ops)
            else:
                for region in [region for region in op.regions if region.quantum_tape is not None]:
                    # Re-enter a JAXPR frame but do not create a new one is none exists.
                    if ctx and region.trace:
                        trace_manager = EvaluationContext.frame_tracing_context(ctx, region.trace)
                    else:
                        trace_manager = nullcontext

                    with trace_manager:
                        nested_ops = ctrl_distribute(
                            region.quantum_tape, control_wires, control_values, work_wires
                        )
                        region.quantum_tape = QuantumTape(
                            nested_ops, region.quantum_tape.measurements
                        )
                new_ops.append(op)
        elif isinstance(op, qml.ops.Adjoint):
            # ctrl resolves faster for nested hybrid controls than create_controlled_op
            ctrl_op = ctrl(
                copy.copy(op.base),
                control=control_wires,
                control_values=control_values,
                work_wires=work_wires,
            )
            new_ops.append(create_adjoint_op(ctrl_op, lazy=True))
        else:
            ctrl_op = create_controlled_op(
                copy.copy(op),
                control=control_wires,
                control_values=control_values,
                work_wires=work_wires,
            )
            new_ops.append(ctrl_op)
    return new_ops


## PRIVATE ##
def _check_no_measurements(tape: QuantumTape) -> None:
    """Check the nested quantum tape for the absense of quantum measurements of any kind"""

    if len(tape.measurements) > 0:
        raise ValueError("Quantum measurements are not allowed")
    for op in tape.operations:
        if has_nested_tapes(op):
            for r in [r for r in op.regions if r.quantum_tape is not None]:
                _check_no_measurements(r.quantum_tape)
        else:
            if isinstance(op, MidCircuitMeasure):
                raise ValueError(
                    "Measurements cannot be used within an adjoint() or ctrl() region."
                )
