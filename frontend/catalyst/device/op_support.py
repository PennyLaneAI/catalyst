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
import jax.numpy as jnp
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
    if not hasattr(op, "grad_recipe"):
        return True

    if not any(map(lambda x: x, op.grad_recipe)):
        return True

    for _, grad_recipe in zip(op.data, op.grad_recipe, strict=True):
        left, right = grad_recipe
        try:
            with jax.ensure_compile_time_eval():
                # exp_param_shift_rule_{left,right} are constants in Catalyst
                # we must ensure that the rules seen in the op match Catalyst's implementation.
                exp_param_shift_rule_left = jnp.array([0.5, 1.0, jnp.pi / 2])
                exp_param_shift_rule_right = jnp.array([-0.5, 1.0, -jnp.pi / 2])
                obs_param_shift_rule_left = jnp.array(left)
                obs_param_shift_rule_right = jnp.array(right)
                is_left_valid = jnp.allclose(obs_param_shift_rule_left, exp_param_shift_rule_left)
                is_right_valid = jnp.allclose(
                    obs_param_shift_rule_right, exp_param_shift_rule_right
                )
            return bool(is_left_valid and is_right_valid)
        except jax.errors.TracerBoolConversionError:
            return False


def _are_param_frequencies_same_as_catalyst(op):
    """Check if the parameter frequencies are all close to 1."""

    if not hasattr(op, "parameter_frequencies"):
        return True

    is_valid_len = len(op.data) == len(op.parameter_frequencies)
    if not is_valid_len:
        return False

    with jax.ensure_compile_time_eval():
        # We use jax.ensure_compile_time_eval
        # to evaluate op.parameter_frequencies (which we expect to always be known at compile time
        # and concrete) and compare with 1.0. Otherwise, jax may generate stablehlo
        # operations for the jnp.allclose
        # This is a purely stylistic choice and one may have also chosen to avoid jax and use
        # numpy instead.
        valid_frequencies = all(
            map(lambda x: jnp.allclose(jnp.array(x), 1.0), op.parameter_frequencies)
        )

    return valid_frequencies


def _paramshift_op_checker(op):

    if not _is_grad_recipe_same_as_catalyst(op):
        return False

    if not _are_param_frequencies_same_as_catalyst(op):
        return False

    if not isinstance(op, HybridOp):
        if op.grad_method not in {"A", None}:
            raise DifferentiableCompileError(f"{op.name} does not support analytic differentiation")
    return True


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
