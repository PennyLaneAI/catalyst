# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for handling quantum operations."""

from typing import Union

import jax
import numpy as np
import pennylane as qml
from pennylane.devices.capabilities import DeviceCapabilities, OperatorProperties
from pennylane.operation import Operation, Operator

from catalyst.api_extensions import MidCircuitMeasure
from catalyst.jax_tracer import HybridOp
from catalyst.utils.exceptions import DifferentiableCompileError

EMPTY_PROPERTIES = OperatorProperties()


def get_base_operation_name(op: Operator) -> str:
    """Get the base operation name, handling controlled and adjoint operations."""
    if type(op) in (qml.ops.Controlled, qml.ops.ControlledOp) or isinstance(op, qml.ops.Adjoint):
        return op.base.name
    return op.name


def is_supported(op: Operator, capabilities: DeviceCapabilities) -> bool:
    """Check whether an operation is supported by the device."""
    return op.name in capabilities.operations


def _is_grad_recipe_same_as_catalyst(op):
    """Checks that the grad_recipe for the op matches the hard coded one in Catalyst."""

    def _is_active(maybe_tracer):
        return isinstance(maybe_tracer, jax.core.Tracer)

    def _is_grad_recipe_active(grad_recipe):
        active = False
        for recipe in grad_recipe:
            left, right = recipe
            active_left = any(map(_is_active, left))
            active_right = any(map(_is_active, right))
            active |= active_left or active_right
        return active

    if _is_grad_recipe_active(op.grad_recipe):
        # An active grad recipe is never the same as the one in catalyst
        return False

    if len(op.data) != len(op.grad_recipe):
        return False

    valid = True
    for grad_recipe in op.grad_recipe:
        left, right = grad_recipe
        # exp_param_shift_rule_{left,right} are constants in Catalyst
        # we must ensure that the rules seen in the op match Catalyst's implementation.
        exp_param_shift_rule_left = np.array([0.5, 1.0, np.pi / 2])
        exp_param_shift_rule_right = np.array([-0.5, 1.0, -np.pi / 2])
        obs_param_shift_rule_left = np.array(left)
        obs_param_shift_rule_right = np.array(right)
        is_left_valid = np.allclose(obs_param_shift_rule_left, exp_param_shift_rule_left)
        is_right_valid = np.allclose(obs_param_shift_rule_right, exp_param_shift_rule_right)
        valid &= is_left_valid and is_right_valid
    return valid


def _has_grad_recipe(op):
    """Checks whether grad_recipe is defined"""
    if not hasattr(op, "grad_recipe"):
        return False

    if not any(map(lambda x: x, op.grad_recipe)):
        return False

    return True


def _has_parameter_frequencies(op):
    try:
        if not hasattr(op, "parameter_frequencies"):
            return False
    except qml.operation.ParameterFrequenciesUndefinedError:
        return False
    return True


def _are_param_frequencies_same_as_catalyst(op):
    """Check if the parameter frequencies are all close to 1."""
    freqs = op.parameter_frequencies
    if len(freqs) != len(op.data):
        return False

    valid = True
    for freqs in op.parameter_frequencies:
        if len(freqs) != 1:
            return False
        valid &= np.allclose(freqs[0], 1.0)

    return valid


def _paramshift_op_checker(op):

    if isinstance(op, qml.QubitUnitary):
        # Cannot take param shift of qubit unitary.
        return False

    if type(op) in (qml.ops.Controlled, qml.ops.ControlledOp):
        # Cannot take param shift of controlled ops.
        # It will always be at least a four term shift rule.
        return False

    if _has_grad_recipe(op):
        return _is_grad_recipe_same_as_catalyst(op)

    if _has_parameter_frequencies(op):
        return _are_param_frequencies_same_as_catalyst(op)

    return isinstance(op, HybridOp)


def _adjoint_diff_op_checker(op, capabilities):
    op_name = get_base_operation_name(op)
    props = capabilities.operations.get(op_name, EMPTY_PROPERTIES)
    return props.differentiable


def is_differentiable(
    op: Operator, capabilities: DeviceCapabilities, grad_method: Union[str, None] = None
) -> bool:
    """Check whether an operation is differentiable on the given device.

    For controlled operations (e.g., CNOT) or adjoint operations (e.g., Adjoint(H)),
    this checks the differentiability of the base operation.
    """
    if grad_method is None:
        return True  # If no gradient method specified, operation is considered differentiable

    if isinstance(op, MidCircuitMeasure):
        raise DifferentiableCompileError(f"{op.name} is not allowed in gradients")

    # Note: Ops with constant parameters generally need not be differentiated. However, the Catalyst
    # compiler will presently consider all real parametrized gates (not including QubitUnitary and
    # StatePrep for example) as part of the hybrid autodiff boundary, and as such will schedule
    # them for differentiation.
    # For the parameter-shift rule, this means partial derivatives of unsupported ops may be wrong.
    # However, given that their parameters are constant, in the final computation of the hybrid
    # Jacobian these partial derivatives will be discarded, and so we can safely consider such
    # operations as inactive.
    # For the adjoint method, most operations present in the program, with the exception of state
    # preparation ops and unitaries, must be considered differentiable, even when acting on
    # constant or integer parameters. For this reason, we cannot skip the validation there.

    if grad_method == "adjoint":
        # lightning will accept constant unitaries
        if isinstance(op, qml.QubitUnitary) and not is_active(op):
            return True
        return _adjoint_diff_op_checker(op, capabilities)
    elif grad_method == "parameter-shift":
        if not is_active(op):
            return True
        return _paramshift_op_checker(op)
    elif grad_method == "fd":
        return True
    elif grad_method == "device":
        raise ValueError(
            "The device does not provide a catalyst compatible gradient method. "
            "Please specify either 'adjoint' or 'parameter-shift' in the QNode's diff_method."
        )
    elif grad_method == "finite-diff":
        raise ValueError(
            "Finite differences at the QNode level is not supported in Catalyst. "
            "Please specify 'fd' directly in the Catalyst gradient function, "
            "i.e. grad(f, method='fd')."
        )
    else:
        raise ValueError(f"Invalid gradient method: {grad_method}")


def is_controllable(op: Operator, capabilities: DeviceCapabilities) -> bool:
    """Check whether an operation is controllable."""
    return capabilities.operations.get(op.name, EMPTY_PROPERTIES).controllable


def is_invertible(op: Operator, capabilities: DeviceCapabilities) -> bool:
    """Check whether an operation is invertible."""
    return capabilities.operations.get(op.name, EMPTY_PROPERTIES).invertible


def is_active(op: Operation) -> bool:
    """Verify whether a gate is considered active for differentiation purposes.

    A gate is considered inactive if all (float) parameters are constant, that is none of them are
    JAX tracers.
    """

    for param in op.data:
        if isinstance(param, jax.core.Tracer):
            return True
        else:
            assert not isinstance(
                param, (list, tuple)
            ), "Operator converts any list/tuple parameters to NumPy arrays."

    return False
