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

from typing import Any, Callable, Sequence, Union

from pennylane import transform
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
from pennylane.operation import Operation, StatePrepBase, Tensor
from pennylane.ops import (
    Adjoint,
    CompositeOp,
    Controlled,
    ControlledOp,
    Hamiltonian,
    SymbolicOp,
)
from pennylane.tape import QuantumTape

from catalyst.api_extensions import HybridAdjoint, HybridCtrl, MidCircuitMeasure
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
        inner_state = op_checker_fn(op, state)
        if has_nested_tapes(op):
            for region in nested_quantum_regions(op):
                if region.trace is not None:
                    with EvaluationContext.frame_tracing_context(ctx, region.trace):
                        inner_state = _verify_nested(region.quantum_tape, inner_state, op_checker_fn)
                else:
                    inner_state = _verify_nested(region.quantum_tape, inner_state, op_checker_fn)
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

    # ToDo: currently we don't check that Tensor itself is supported, only its obs
    # The TOML files have followed the convention of dev.observables from PL and not
    # included Tensor, but this could be updated to validate
    if isinstance(obs, Tensor):
        for o in obs.obs:
            _verify_observable(o, _obs_checker)

    else:
        _obs_checker(obs)

        if isinstance(obs, CompositeOp):
            for o in obs.operands:
                _verify_observable(o, _obs_checker)
        elif isinstance(obs, Hamiltonian):
            for o in obs.ops:
                _verify_observable(o, _obs_checker)
        elif isinstance(obs, SymbolicOp):
            _verify_observable(obs.base, _obs_checker)


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

    def _adj_diff_op_checker(op):
        if type(op) in (Controlled, ControlledOp) or isinstance(op, Adjoint):
            op_name = op.base.name
        else:
            op_name = op.name
        if not qjit_device.qjit_capabilities.native_ops.get(
            op_name, EMPTY_PROPERTIES
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
        # If it's a PL Adjoint we also want to check its base to catch Adjoint(C(base)).
        # PL simplification should mean pure PL operators will not be more nested than this.
        if isinstance(op, Adjoint):
            _ctrl_op_checker(op.base, in_control)
            return in_control
        # Early exit when not in inverse, only determine the control status for recursing later.
        elif not in_control:
            return isinstance(op, HybridCtrl)

        if not qjit_device.qjit_capabilities.native_ops.get(op.name, EMPTY_PROPERTIES).controllable:
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

        if not qjit_device.qjit_capabilities.native_ops.get(op.name, EMPTY_PROPERTIES).invertible:
            raise CompileError(
                f"{op.name} is not invertible on '{qjit_device.original_device.name}' device"
            )

        return True

    def _op_checker(op, state):
        # Don't check PennyLane Adjoint / Controlled instances directly since the compound name
        # (e.g. "Adjoint(Hadamard)") will not show up in the device capabilities. Instead the check
        # is handled in _inv_op_checker and _ctrl_op_checker.
        # Specialed control op classes (e.g. CRZ) should be checked directly though, which is why we
        # can't use isinstance(op, Controlled).
        if type(op) in (Controlled, ControlledOp) or isinstance(op, (Adjoint)):
            pass
        # Don't check StatePrep since StatePrep is not in the list of device capabilities.
        # It is only valid when the TOML file has the initial_state_prep_flag.
        elif (
            isinstance(op, StatePrepBase) and qjit_device.qjit_capabilities.initial_state_prep_flag
        ):
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
                _adj_diff_op_checker(op)
            elif grad_method == "parameter-shift":
                _paramshift_op_checker(op)

        return (in_inverse, in_control)

    _verify_nested(tape, (False, False), _op_checker)

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
            if isinstance(m.obs, Tensor):
                _ = [_obs_checker(o) for o in m.obs.obs]
            else:
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
            _verify_observable(m.obs, _obs_checker)

    return (tape,), lambda x: x[0]


@transform
def validate_measurements(
    tape: QuantumTape, qjit_capabilities: dict, name: str, shots: Union[int, Shots]
) -> (Sequence[QuantumTape], Callable):
    """Validates the observables and measurements for a circuit against the capabilites
    from the TOML file.

    Args:
        tape (QuantumTape or QNode or Callable): a quantum circuit.
        qjit_capabilities (dict): specifies the capabilities of the qjitted device
        name: the name of the device to use in error messages
        shots: the shots on the device to use in error messages

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumTape], function]:
        The unaltered input circuit.

    Raises:
        CompileError: if a measurement is not supported by the given device with Catalyst

    """

    def _obs_checker(obs):
        if not qjit_capabilities.native_obs.get(obs.name):
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
        mp_name = m.return_type.value if m.return_type else type(m).__name__
        if not mp_name.title() in qjit_capabilities.measurement_processes:
            raise CompileError(
                f"{type(m)} is not a supported measurement process on '{name}' with Catalyst"
            )

    return (tape,), lambda x: x[0]
