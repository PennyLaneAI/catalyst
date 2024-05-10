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

from typing import Any, Callable

from pennylane.measurements import MeasurementProcess
from pennylane.operation import Observable, Operation
from pennylane.ops import Controlled
from pennylane.tape import QuantumTape

from catalyst.tracing.contexts import EvaluationContext
from catalyst.utils.exceptions import CompileError, DifferentiableCompileError
from catalyst.utils.toml import OperationProperties

# pylint: disable=import-outside-toplevel


def _verify_nested(
    tape: QuantumTape,
    state: Any,
    op_checker_fn: Callable[[Operation, Any], Any],
    obs_checker_fn: Callable[[Observable, Any], Any],
) -> Any:
    """Traverse the nested quantum tape, carry a caller-defined state."""

    # FIXME: How should we re-organize the code to avoid this kind of circular dependency.
    # Another candidate: `from catalyst.qjit_device import AnyQJITDevice`
    from catalyst.jax_tracer import has_nested_tapes, nested_quantum_regions

    ctx = EvaluationContext.get_main_tracing_context()
    for op in tape.operations:
        state = op_checker_fn(op, state)
        if has_nested_tapes(op):
            nested_state = state
            for region in nested_quantum_regions(op):
                if region.trace is not None:
                    with EvaluationContext.frame_tracing_context(ctx, region.trace):
                        nested_state = _verify_nested(
                            region.quantum_tape, nested_state, op_checker_fn, obs_checker_fn
                        )
                else:
                    nested_state = _verify_nested(
                        region.quantum_tape, nested_state, op_checker_fn, obs_checker_fn
                    )

    for obs in tape.observables:
        state = obs_checker_fn(obs, state)
    return state


EMPTY_PROPERTIES = OperationProperties(False, False, False)


def verify_inverses(device: "AnyQJITDevice", tape: QuantumTape) -> None:
    """Verify quantum program against the device capabilities.

    Raises: CompileError
    """

    # FIXME: How should we re-organize the code to avoid this kind of circular dependency?
    from catalyst.api_extensions.quantum_operators import Adjoint

    def _op_checker(op, in_inverse):
        if in_inverse > 0:
            op_name = op.base.name if isinstance(op, Controlled) else op.name
            if not device.qjit_capabilities.native_ops.get(op_name, EMPTY_PROPERTIES).invertible:
                raise CompileError(
                    f"{op_name} is not invertible on '{device.original_device.name}' device"
                )
        return (in_inverse + 1) if isinstance(op, Adjoint) else in_inverse

    def _obs_checker(_, state):
        return state

    _verify_nested(tape, 0, _op_checker, _obs_checker)


def verify_control(device: "AnyQJITDevice", tape: QuantumTape) -> None:
    """Verify quantum program against the device capabilities.

    Raises: CompileError
    """

    # FIXME: How should we re-organize the code to avoid this kind of circular dependency?
    from catalyst.api_extensions.quantum_operators import QCtrl

    def _op_checker(op, in_control):
        if in_control > 0:
            if not device.qjit_capabilities.native_ops.get(op.name, EMPTY_PROPERTIES).controllable:
                raise CompileError(
                    f"{op.name} is not controllable on '{device.original_device.name}' device"
                )
        return (in_control + 1) if isinstance(op, QCtrl) else in_control

    def _obs_checker(_, state):
        return state

    _verify_nested(tape, 0, _op_checker, _obs_checker)


def verify_adjoint_differentiability(device: "AnyQJITDevice", tape: QuantumTape) -> None:
    """Verify quantum program against the device capabilities.

    Raises: DifferentiableCompileError
    """

    def _op_checker(op, _):
        if not device.qjit_capabilities.native_ops.get(op.name, EMPTY_PROPERTIES).differentiable:
            raise DifferentiableCompileError(
                f"{op.name} is non-differentiable on '{device.original_device.name}' device"
            )

    def _obs_checker(obs, _):
        if isinstance(obs, MeasurementProcess):
            _obs_checker(obs.obs or [], _)
        elif isinstance(obs, list):
            for obs2 in obs:
                _obs_checker(obs2, _)
        else:
            if not device.qjit_capabilities.native_obs.get(
                obs.name, EMPTY_PROPERTIES
            ).differentiable:
                raise DifferentiableCompileError(
                    f"{obs.name} is non-differentiable on '{device.original_device.name}' device"
                )

    _verify_nested(tape, None, _op_checker, _obs_checker)


def verify_no_mid_circuit_measurement(_, tape: QuantumTape) -> None:
    """Verify quantum program against the device capabilities.

    Raises: DifferentiableCompileError
    """

    from catalyst.api_extensions import MidCircuitMeasure

    def _op_checker(op, _):
        if isinstance(op, MidCircuitMeasure):
            raise DifferentiableCompileError(f"{op.name} is not allowed in gradinets")

    def _obs_checker(_obs, st):
        return st

    _verify_nested(tape, None, _op_checker, _obs_checker)
