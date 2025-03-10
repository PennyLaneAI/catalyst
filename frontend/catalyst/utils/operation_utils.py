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

import pennylane as qml
from pennylane.devices.capabilities import DeviceCapabilities, OperatorProperties
from pennylane.operation import Operator

EMPTY_PROPERTIES = OperatorProperties()


def get_operation_name(op: Operator) -> str:
    """Get the base operation name, handling controlled and adjoint operations."""
    if type(op) in (qml.ops.Controlled, qml.ops.ControlledOp) or isinstance(op, qml.ops.Adjoint):
        return op.base.name
    return op.name


def is_supported(op: Operator, capabilities: DeviceCapabilities) -> bool:
    """Check whether an operation is supported by the device."""
    op_name = get_operation_name(op)
    return op_name in capabilities.operations


def is_differentiable(op: Operator, capabilities: DeviceCapabilities) -> bool:
    """Check whether an operation is differentiable."""
    op_name = get_operation_name(op)
    return capabilities.operations.get(op_name, EMPTY_PROPERTIES).differentiable


def is_controllable(op: Operator, capabilities: DeviceCapabilities) -> bool:
    """Check whether an operation is controllable."""
    op_name = get_operation_name(op)
    return capabilities.operations.get(op_name, EMPTY_PROPERTIES).controllable


def is_invertible(op: Operator, capabilities: DeviceCapabilities) -> bool:
    """Check whether an operation is invertible."""
    op_name = get_operation_name(op)
    return capabilities.operations.get(op_name, EMPTY_PROPERTIES).invertible
