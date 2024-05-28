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
from collections.abc import Sized
from typing import Any, Callable, List, Optional, Union

import jax
import pennylane as qml
from jax._src.tree_util import tree_flatten
from jax.core import get_aval
from pennylane import QueuingManager
from pennylane.operation import Operator
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
    trace_quantum_tape,
)
from catalyst.tracing.contexts import EvaluationContext


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
    [array(1.), array(False)]
    >>> circuit(0.43)
    [array(-1.), array(True)]

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
    -1.0

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
    1.0
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
    in_classical_tracers.append(postselect)

    m = new_inner_tracer(ctx.trace, get_aval(True))
    MidCircuitMeasure(
        in_classical_tracers=in_classical_tracers,
        out_classical_tracers=[m],
        regions=[],
    )

    # If reset was requested, reset qubit only if the measurement result was 1
    if reset:

        @cond(m)
        def reset_fn():
            qml.PauliX(wires=wires)

        reset_fn()

    return m


def adjoint(f: Union[Callable, Operator]) -> Union[Callable, Operator]:
    """A :func:`~.qjit` compatible adjoint transformer for PennyLane/Catalyst.

    Returns a quantum function or operator that applies the adjoint of the
    provided function or operator.

    .. warning::

        This function does not support performing the adjoint
        of quantum functions that contain mid-circuit measurements.

    Args:
        f (Callable or Operator): A PennyLane operation or a Python function
                                  containing PennyLane quantum operations.

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
    array([0.5, 0.5])

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

    if not EvaluationContext.is_tracing():
        return qml.adjoint(f)

    def _call_handler(*args, _callee: Callable, **kwargs):
        EvaluationContext.check_is_quantum_tracing(
            "catalyst.adjoint can only be used from within a qml.qnode."
        )
        ctx = EvaluationContext.get_main_tracing_context()
        with EvaluationContext.frame_tracing_context(ctx) as inner_trace:
            in_classical_tracers, _ = tree_flatten((args, kwargs))
            wffa, in_avals, _, _ = deduce_avals(_callee, args, kwargs)
            arg_classical_tracers = _input_type_to_tracers(inner_trace.new_arg, in_avals)
            quantum_tape = QuantumTape()
            with QueuingManager.stop_recording(), quantum_tape:
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

        return Adjoint(
            in_classical_tracers=in_classical_tracers,
            out_classical_tracers=[],
            regions=[adjoint_region],
        )

    if isinstance(f, Callable):

        def _callable(*args, **kwargs):
            return _call_handler(*args, _callee=f, **kwargs)

        return _callable
    elif isinstance(f, Operator):
        QueuingManager.remove(f)

        def _callee():
            QueuingManager.append(f)

        return _call_handler(_callee=_callee)
    else:
        raise ValueError(f"Expected a callable or a qml.Operator, not {f}")


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
    array([0.25, 0.25, 0.03661165, 0.46338835])
    """

    if not EvaluationContext.is_tracing():
        return qml.ctrl(f, control, control_values, work_wires)

    if control_values is not None and (
        (len(control) if isinstance(control, Sized) else 1)
        != (len(control_values) if isinstance(control_values, Sized) else 1)
    ):
        raise ValueError(
            f"Length of the control_values ({len(control_values)}) must be None or equal "
            f"to the lenght of control ({len(control)})"
        )

    def _call_handler(*args, _callee: Callable, **kwargs):
        EvaluationContext.check_is_quantum_tracing(
            "catalyst.ctrl can only be used from within a qml.qnode."
        )
        in_classical_tracers, _ = tree_flatten((args, kwargs))
        quantum_tape = QuantumTape()
        with QueuingManager.stop_recording(), quantum_tape:
            res = _callee(*args, **kwargs)
        out_classical_tracers, _ = tree_flatten(res)

        _check_no_measurements(quantum_tape)

        region = HybridOpRegion(None, quantum_tape, [], [])

        # Return the operation instance since PL expects this for qml.ctrl(op).
        return QCtrl(
            control_wires=control,
            control_values=control_values,
            work_wires=work_wires,
            in_classical_tracers=in_classical_tracers,
            out_classical_tracers=out_classical_tracers,
            regions=[region],
        )

    if isinstance(f, Callable):

        def _callable(*args, **kwargs):
            return _call_handler(*args, _callee=f, **kwargs)

        return _callable

    elif isinstance(f, Operator):
        QueuingManager.remove(f)

        def _callee():
            QueuingManager.append(f)

        return _call_handler(_callee=_callee)

    else:
        raise ValueError(f"Expected a callable or a qml.Operator, not {f}")  # pragma: no cover


## IMPL ##
class MidCircuitMeasure(HybridOp):
    """Operation representing a mid-circuit measurement."""

    binder = qmeasure_p.bind

    def __init__(self, *args, **kwargs):
        self.postselect = kwargs.pop("postselect", None)
        self.reset = kwargs.pop("reset", False)
        super().__init__(*args, **kwargs)

    def trace_quantum(self, ctx, device, trace, qrp) -> QRegPromise:
        op = self
        wire = op.in_classical_tracers[0]
        qubit = qrp.extract([wire])[0]
        postselect = op.in_classical_tracers[1]

        qubit2 = op.bind_overwrite_classical_tracers(ctx, trace, qubit, postselect=postselect)
        qrp.insert([wire], [qubit2])
        return qrp


class Adjoint(HybridOp):
    """PennyLane's adjoint operation"""

    binder = adjoint_p.bind

    def trace_quantum(self, ctx, device, trace, qrp) -> QRegPromise:
        op = self
        body_trace = op.regions[0].trace
        body_tape = op.regions[0].quantum_tape
        res_classical_tracers = op.regions[0].res_classical_tracers
        with EvaluationContext.frame_tracing_context(ctx, body_trace):
            qreg_in = _input_type_to_tracers(body_trace.new_arg, [AbstractQreg()])[0]
            qrp_out = trace_quantum_tape(body_tape, device, qreg_in, ctx, body_trace)
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


# TODO: This class needs to be made interoperable with qml.Controlled since qml.ctrl dispatches
#       to this class whenever a qjit context is active.
class QCtrl(HybridOp):
    """Catalyst quantum ctrl operation"""

    def __init__(self, *args, control_wires, control_values=None, work_wires=None, **kwargs):
        self._control_wires = qml.wires.Wires(control_wires)
        self._work_wires = qml.wires.Wires([] if work_wires is None else work_wires)
        if control_values is None:
            self._control_values = [True] * len(self._control_wires)
        elif isinstance(control_values, (int, bool)):
            self._control_values = [control_values]
        else:
            self._control_values = control_values

        super().__init__(*args, **kwargs)

    def trace_quantum(self, ctx, device, trace, qrp) -> QRegPromise:
        raise NotImplementedError("QCtrl does not support JAX quantum tracing")  # pragma: no cover

    def decomposition(self):
        """Compute quantum decomposition of the gate by recursively scanning the nested tape and
        distributing the quantum control operaiton over the tape operations."""
        assert len(self.regions) == 1, "Qctrl is expected to have one region"

        _check_no_measurements(self.regions[0].quantum_tape)
        new_tape = qctrl_distribute(
            self.regions[0].quantum_tape,
            self._control_wires,
            self._control_values,
            self._work_wires,
        )
        return new_tape.operations

    @property
    def wires(self):
        """The list of all control-wires, work-wires, and active-wires."""
        assert len(self.regions) == 1, "Qctrl is expected to have one region"

        total_wires = sum(
            (op.wires for op in self.regions[0].quantum_tape.operations),
            self._control_wires,
        )
        total_wires += self._work_wires
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


def qctrl_distribute(
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
    ctx = EvaluationContext.get_main_tracing_context()
    ops2 = []
    for op in tape.operations:
        if has_nested_tapes(op):
            if isinstance(op, QCtrl):
                for region in [region for region in op.regions if region.quantum_tape is not None]:
                    tape2 = qctrl_distribute(
                        region.quantum_tape,
                        control_wires + op.control_wires,
                        control_values + op.control_values,
                        work_wires + op.work_wires,
                    )
                    ops2.extend(tape2.operations)
            else:
                for region in [region for region in op.regions if region.quantum_tape is not None]:
                    with EvaluationContext.frame_tracing_context(ctx, region.trace):
                        region.quantum_tape = qctrl_distribute(
                            region.quantum_tape, control_wires, control_values, work_wires
                        )
                ops2.append(op)
        else:
            ops2.append(
                create_controlled_op(
                    copy.copy(op),
                    control=control_wires,
                    control_values=control_values,
                    work_wires=work_wires,
                )
            )
    return QuantumTape(ops2, tape.measurements)


## PRIVATE ##
def _check_no_measurements(tape: QuantumTape) -> None:
    """Check the nested quantum tape for the absense of quantum measurements of any kind"""

    msg = "Quantum measurements are not allowed"

    if len(tape.measurements) > 0:
        raise ValueError(msg)
    for op in tape.operations:
        if has_nested_tapes(op):
            for r in [r for r in op.regions if r.quantum_tape is not None]:
                _check_no_measurements(r.quantum_tape)
        else:
            if isinstance(op, MidCircuitMeasure):
                raise ValueError(msg)
