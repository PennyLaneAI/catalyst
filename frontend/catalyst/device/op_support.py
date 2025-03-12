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


def _paramshift_op_checker(op):
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

    if not is_active(op):  # ops with constant parameters can be ignored
        return True

    if grad_method == "adjoint":
        return _adjoint_diff_op_checker(op, capabilities)
    elif grad_method == "parameter-shift":
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
    JAX tracers. The restriction to float values facilitates the process without knowing the
    semantics of each parameter, since generally (in Catalyst) only real numbers can be
    differentiated with respect to.
    """

    for param in op.data:
        if isinstance(param, jax.core.Tracer) and param.dtype.kind == "f":
            return True
        else:
            assert not isinstance(
                param, (list, tuple)
            ), "Operator converts any list/tuple parameters to NumPy arrays."

    return False
