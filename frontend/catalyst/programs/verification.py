# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains for program verification.
"""

from typing import Any, Callable, Sequence

from pennylane import transform
from pennylane.measurements import (
    MeasurementProcess,
    MutualInfoMP,
    StateMP,
    VarianceMP,
    VnEntropyMP,
)
from pennylane.operation import Operation, Tensor
from pennylane.ops import (
    Adjoint,
    CompositeOp,
    Controlled,
    ControlledOp,
    ControlledQubitUnitary,
    Hamiltonian,
    SymbolicOp,
)
from pennylane.tape import QuantumTape

from catalyst.api_extensions import MidCircuitMeasure
from catalyst.jax_tracer import HybridOp, has_nested_tapes, nested_quantum_regions
from catalyst.tracing.contexts import EvaluationContext
from catalyst.utils.exceptions import CompileError, DifferentiableCompileError
from catalyst.utils.toml import OperationProperties


def _verify_nested(
    tape: QuantumTape,
    state: Any,
    op_checker_fn: Callable[[Operation, Any], Any],
) -> Any:
    """Traverse the nested quantum tape, carry a caller-defined state."""

    ctx = EvaluationContext.get_main_tracing_context()
    for op in tape.operations:
        state = op_checker_fn(op, state)
        if has_nested_tapes(op):
            for region in nested_quantum_regions(op):
                if region.trace is not None:
                    with EvaluationContext.frame_tracing_context(ctx, region.trace):
                        state = _verify_nested(region.quantum_tape, state, op_checker_fn)
                else:
                    state = _verify_nested(region.quantum_tape, state, op_checker_fn)
    return state


EMPTY_PROPERTIES = OperationProperties(False, False, False)


@transform
def verify_no_state_variance_returns(tape: QuantumTape) -> None:
    """Verify that no measuremnts contain state or variance."""

    if any(isinstance(m, (StateMP, VnEntropyMP, MutualInfoMP)) for m in tape.measurements):
        raise DifferentiableCompileError("State returns are forbidden in gradients")

    if any(isinstance(m, VarianceMP) for m in tape.measurements):
        raise DifferentiableCompileError("Variance returns are forbidden in gradients")

    return (tape,), lambda x: x[0]


@transform
def verify_operations(tape: QuantumTape, grad_method, qjit_device):
    """verify the quantum program against Catalyst requirements. This transform makes no
    transformations.

    Raises:
        DifferentiableCompileError: gradient-related error
        CompileError: compilation error
    """

    def _paramshift_op_checker(op):
        if not isinstance(op, HybridOp):
            if op.grad_method not in {"A", None}:
                raise DifferentiableCompileError(
                    f"{op.name} does not support analytic differentiation"
                )

    def _mcm_op_checker(op):
        if isinstance(op, MidCircuitMeasure):
            raise DifferentiableCompileError(f"{op.name} is not allowed in gradinets")

    def _adj_op_checker(op):
        if not qjit_device.qjit_capabilities.native_ops.get(
            op.name, EMPTY_PROPERTIES
        ).differentiable:
            raise DifferentiableCompileError(
                f"{op.name} is non-differentiable on '{qjit_device.original_device.name}' device"
            )

    def _ctrl_op_checker(op, in_control):
        # For PL controlled instances we don't recurse via nested tapes, so check the base op here.
        if type(op) in (Controlled, ControlledOp):
            if isinstance(op.base, HybridOp):
                raise CompileError(
                    f"Cannot compile PennyLane control of the hybrid op {type(op.base)}."
                )
            _ctrl_op_checker(op.base, True)
            return in_control
        # Early exit when not in inverse, only determine the control status for recursing later.
        elif not in_control:
            return isinstance(op, HybridCtrl)

        # For PL adjoint instances look at control support of the base gate, since Adjoint(Op)
        # is implemented as Op(..., inverse=True).
        if isinstance(op, Adjoint):
            op_name = op.base.name
        else:
            op_name = op.name

        if not qjit_device.qjit_capabilities.native_ops.get(op_name, EMPTY_PROPERTIES).controllable:
            raise CompileError(
                f"{op_name} is not controllable on '{qjit_device.original_device.name}' device"
            )

        return True

    def _inv_op_checker(op, in_inverse):
        # For PL adjoint instances we don't recurse via nested tapes, so check the base op here.
        if isinstance(op, Adjoint):
            if isinstance(op.base, HybridOp):
                raise CompileError(
                    f"Cannot compile PennyLane inverse of the hybrid op {type(op.base)}."
                )
            _inv_op_checker(op.base, in_inverse=True)
            return in_inverse
        # Early exit when not in inverse, only determine the inverse status for recursing later.
        elif not in_inverse:
            return isinstance(op, HybridAdjoint)

        # For PL controlled instances look at adjoint support of the base gate, since Controlled(Op)
        # is implemented as Op(..., control_wires=...).
        # TODO: remove ControlledQubitUnitary to treat it as independant gate everywhere
        if type(op) in (Controlled, ControlledOp, ControlledQubitUnitary):
            op_name = op.base.name
        else:
            op_name = op.name

        if not qjit_device.qjit_capabilities.native_ops.get(op_name, EMPTY_PROPERTIES).invertible:
            raise CompileError(
                f"{op_name} is not invertible on '{qjit_device.original_device.name}' device"
            )

        return True

    def _op_checker(op, state):

        # Don't check PennyLane Adjoint / Controlled instances directly since the compound name
        # (e.g. "Adjoint(Hadamard)") will not show up in the device capabilities. Instead the check
        # is handled in _inv_op_checker and _ctrl_op_checker.
        # Specialed control op classes (e.g. CRZ) should be checked directly though, which is why we
        # can't use isinstance(op, Controlled).
        if type(op) in (Controlled, ControlledOp) or isinstance(op, Adjoint):
            pass
        elif not qjit_device.qjit_capabilities.native_ops.get(op.name):
            raise CompileError(
                f"{op.name} is not supported on '{qjit_device.original_device.name}' device"
            )

        # check validity of ops nested inside control or adjoint
        in_inverse, in_control = state
        in_inverse = _inv_op_checker(op, in_inverse)
        in_control = _ctrl_op_checker(op, in_control)

        # check validity based on grad method if using
        if grad_method is not None:
            _mcm_op_checker(op)
            if grad_method == "adjoint":
                _adj_op_checker(op)
            elif grad_method == "parameter-shift":
                _paramshift_op_checker(op)

        return (in_inverse, in_control)

    _verify_nested(tape, (False, False), _op_checker)

    return (tape,), lambda x: x[0]


@transform
def validate_observables_parameter_shift(tape: QuantumTape):
    """Validate that the observables on the tape support parameter shift"""

    def _obs_checker(obs):
        if obs and obs.grad_method not in {"A", None}:
            raise DifferentiableCompileError(
                f"{obs.name} does not support analytic differentiation"
            )

    for m in tape.measurements:
        if m.obs:
            _obs_checker(m.obs)

    return (tape,), lambda x: x[0]


@transform
def validate_observables_adjoint_diff(tape: QuantumTape, qjit_device):
    """Validate that the observables on the tape support adjoint differentiation"""

    def _obs_checker(obs):
        if not qjit_device.qjit_capabilities.native_obs.get(
            obs.name, EMPTY_PROPERTIES
        ).differentiable:
            raise DifferentiableCompileError(
                f"{obs.name} is non-differentiable on "
                f"'{qjit_device.original_device.name}' device"
            )

    for m in tape.measurements:
        if m.obs:
            _obs_checker(m.obs)

    return (tape,), lambda x: x[0]


@transform
def validate_observables(
    tape: QuantumTape, qjit_capabilities: dict, name: str
) -> (Sequence[QuantumTape], Callable):
    """Validates the observables and measurements for a circuit against the capabilites
    from the TOML file.

    Args:
        tape (QuantumTape or QNode or Callable): a quantum circuit.
        qjit_capabilities (dict): specifies the capabilities of the qjitted device
        name (str): the name of the device to use in error messages.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumTape], function]:
        The unaltered input circuit.

    Raises:
        CompileError: if an observable is not supported by the device with Catalyst

    """

    def _observable_is_supported(obs) -> bool:
        """Specifies whether or not an observable is accepted by QJITDevice.

        If the observable is built on one or multiple other observables, check
        both that the overall observable is supported, and that its component
        parts are supported."""

        if isinstance(obs, Tensor):
            return all([_observable_is_supported(o) for o in obs.obs])

        if isinstance(obs, (Hamiltonian, CompositeOp, SymbolicOp)):

            if not qjit_capabilities.native_obs.get(obs.name):
                return False

            if hasattr(obs, "operands"):
                return all([_observable_is_supported(o) for o in obs.operands])
            elif hasattr(obs, "ops"):
                return all([_observable_is_supported(o) for o in obs.ops])
            elif hasattr(obs, "base"):
                return _observable_is_supported(obs.base)

        return qjit_capabilities.native_obs.get(obs.name)

    for m in tape.measurements:
        if m.obs and not _observable_is_supported(m.obs):
            raise CompileError(
                f"{m.obs} is not supported as an observable on the '{name}' device with Catalyst"
            )

    return (tape,), lambda x: x[0]
