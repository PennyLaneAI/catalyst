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

"""
This module contains the functions to verify quantum tapes are fully compatible
with the compiler and device.
"""

from typing import Any, Callable, List, Sequence, Union

from pennylane import transform
from pennylane.devices.capabilities import DeviceCapabilities, OperatorProperties
from pennylane.measurements import (
    ExpectationMP,
    MutualInfoMP,
    SampleMeasurement,
    StateMeasurement,
    StateMP,
    VarianceMP,
    VnEntropyMP,
)
from pennylane.measurements.shots import Shots
from pennylane.operation import Operation
from pennylane.ops import (
    Adjoint,
    BasisState,
    CompositeOp,
    Controlled,
    ControlledOp,
    StatePrep,
    SymbolicOp,
)
from pennylane.tape import QuantumTape

from catalyst.api_extensions import HybridAdjoint, HybridCtrl
from catalyst.device.op_support import (
    EMPTY_PROPERTIES,
    is_active,
    is_controllable,
    is_differentiable,
    is_invertible,
)
from catalyst.jax_tracer import HybridOp, has_nested_tapes, nested_quantum_regions
from catalyst.tracing.contexts import EvaluationContext
from catalyst.utils.exceptions import CompileError, DifferentiableCompileError


def _verify_nested(
    operations: List[Operation],
    state: Any,
    op_checker_fn: Callable[[Operation, Any], Any],
) -> Any:
    """Traverse the nested quantum tape, carry a caller-defined state."""

    for op in operations:
        inner_state = op_checker_fn(op, state)
        if has_nested_tapes(op):
            for region in nested_quantum_regions(op):
                if region.trace is not None:
                    with EvaluationContext.frame_tracing_context(region.trace):
                        inner_state = _verify_nested(
                            region.quantum_tape.operations, inner_state, op_checker_fn
                        )
                else:
                    inner_state = _verify_nested(
                        region.quantum_tape.operations, inner_state, op_checker_fn
                    )
    return state


def _verify_observable(obs: Operation, _obs_checker: Callable) -> bool:
    """Validates whether or not an observable is accepted by QJITDevice.

    Args:
        obs(Operator): the observable to be validated
        _obs_checker(Callable): a callable that takes an observable
            and raises an error if the observable is unsupported.

    Raises:
        DifferentiableCompileError: gradient-related error
        CompileError: compilation error

    If the observable is built on one or multiple other observables, check
    both that the overall observable is supported, and that its component
    parts are supported."""

    _obs_checker(obs)

    if isinstance(obs, CompositeOp):
        for o in obs.operands:
            _verify_observable(o, _obs_checker)

    elif isinstance(obs, SymbolicOp):
        _verify_observable(obs.base, _obs_checker)


EMPTY_PROPERTIES = OperatorProperties()


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

    supported_ops = qjit_device.capabilities.operations

    def _grad_method_op_checker(op, grad_method):
        """Check if an operation supports the specified gradient method."""
        if not is_differentiable(op, qjit_device.capabilities, grad_method):
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
        # If it's a PL Adjoint we also want to check its base to catch Adjoint(C(base)).
        # PL simplification should mean pure PL operators will not be more nested than this.
        if isinstance(op, Adjoint):
            _ctrl_op_checker(op.base, in_control)
            return in_control
        # Early exit when not in inverse, only determine the control status for recursing later.
        elif not in_control:
            return isinstance(op, HybridCtrl)

        if not is_controllable(op, qjit_device.capabilities):
            raise CompileError(
                f"{op.name} is not controllable on '{qjit_device.original_device.name}' device"
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
        # If its a PL Controlled we also want to check its base to catch C(Adjoint(base)).
        # PL simplification should mean pure PL operators will not be more nested than this.
        if type(op) in (Controlled, ControlledOp):
            _inv_op_checker(op.base, in_inverse)
            return in_inverse
        # Early exit when not in inverse, only determine the inverse status for recursing later.
        elif not in_inverse:
            return isinstance(op, HybridAdjoint)

        if not is_invertible(op, qjit_device.capabilities):
            raise CompileError(
                f"{op.name} is not invertible on '{qjit_device.original_device.name}' device"
            )

        return True

    def _op_checker(op, state):
        # Don't check PennyLane Adjoint / Controlled instances directly since the compound name
        # (e.g. "Adjoint(Hadamard)") will not show up in the device capabilities. Instead the check
        # is handled in _inv_op_checker and _ctrl_op_checker.
        # Specialized control op classes (e.g. CRZ) should be checked directly though, which is why
        # we can't use isinstance(op, Controlled).
        if type(op) in (Controlled, ControlledOp) or isinstance(op, (Adjoint)):
            pass
        elif not op.name in supported_ops:
            raise CompileError(
                f"{op.name} is not supported on '{qjit_device.original_device.name}' device"
            )

        # check validity of ops nested inside control or adjoint
        in_inverse, in_control = state
        in_inverse = _inv_op_checker(op, in_inverse)
        in_control = _ctrl_op_checker(op, in_control)

        # check validity based on grad method if using
        if grad_method is not None:
            _grad_method_op_checker(op, grad_method)

        return (in_inverse, in_control)

    ops_to_verify = tape.operations
    # state prep support only at the beginning of the program has a special flag
    if qjit_device.capabilities.initial_state_prep:
        # Catalyst only supports two types of state prep ops (see also comment in decomposition)
        if len(ops_to_verify) > 0 and type(ops_to_verify[0]) in (StatePrep, BasisState):
            # inactive state prep ops can also be allowed in differentiated programs
            if grad_method is None or not is_active(tape[0]):
                ops_to_verify = ops_to_verify[1:]

    _verify_nested(ops_to_verify, (False, False), _op_checker)

    return (tape,), lambda x: x[0]


@transform
def validate_observables_parameter_shift(tape: QuantumTape):
    """Validate that the observables on the tape support parameter shift"""

    # ToDo: add verification support for Composite and Symbolic op
    # observables with parameter shift

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
        if not qjit_device.capabilities.observables.get(obs.name, EMPTY_PROPERTIES).differentiable:
            raise DifferentiableCompileError(
                f"{obs.name} is non-differentiable on "
                f"'{qjit_device.original_device.name}' device"
            )

    for m in tape.measurements:
        if m.obs:
            _verify_observable(m.obs, _obs_checker)

    return (tape,), lambda x: x[0]


@transform
def validate_measurements(
    tape: QuantumTape, capabilities: DeviceCapabilities, name: str, shots: Union[int, Shots]
) -> (Sequence[QuantumTape], Callable):
    """Validates the observables and measurements for a circuit against the capabilites
    from the TOML file.

    Args:
        tape (QuantumTape or QNode or Callable): a quantum circuit.
        capabilities (DeviceCapabilities): specifies the capabilities of the qjitted device
        name: the name of the device to use in error messages
        shots: the shots on the device to use in error messages

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumTape], function]:
        The unaltered input circuit.

    Raises:
        CompileError: if a measurement is not supported by the given device with Catalyst

    """

    def _obs_checker(obs):
        if not obs.name in capabilities.observables:
            raise CompileError(
                f"{m.obs} is not supported as an observable on the '{name}' device with Catalyst"
            )

    for m in tape.measurements:
        # verify observable is supported
        if m.obs:
            if not isinstance(m, (ExpectationMP, VarianceMP)):
                raise CompileError(
                    "Only expectation value and variance measurements can "
                    "accept observables with Catalyst"
                )
            _verify_observable(m.obs, _obs_checker)
        # verify measurement process type is supported
        if shots and not isinstance(m, SampleMeasurement):
            raise CompileError(
                f"State-based measurements like {m} cannot work with finite shots. "
                "Please specify shots=None."
            )
        if not shots and not isinstance(m, StateMeasurement):
            raise CompileError(
                f"Sample-based measurements like {m} cannot work with shots=None. "
                "Please specify a finite number of shots."
            )
        mp_name = type(m).__name__
        if not mp_name in capabilities.measurement_processes:
            raise CompileError(
                f"{mp_name} is not a supported measurement process on '{name}' with Catalyst"
            )

    return (tape,), lambda x: x[0]
