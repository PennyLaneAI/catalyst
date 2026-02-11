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
"""
Utilities for getting device capabilities in decomposition.
"""

import textwrap

import pennylane as qml
from pennylane.devices.capabilities import DeviceCapabilities

from catalyst.device.capabilities_utils import (
    filter_device_capabilities_with_shots as _filter_device_capabilities_with_shots,
)
from catalyst.device.capabilities_utils import load_device_capabilities as _load_device_capabilities
from catalyst.device.capabilities_utils import requires_shots as _requires_shots
from catalyst.device.decomposition import catalyst_acceptance as _catalyst_acceptance
from catalyst.jax_primitives_utils import _calculate_diff_method
from catalyst.utils.exceptions import CompileError


def catalyst_acceptance(op, capabilities, diff_method):
    """Check if an operation is supported by the device."""
    return _catalyst_acceptance(op, capabilities, diff_method)


def calculate_diff_method(device, closed_jaxpr):
    """Calculate the diff method for the device."""
    return _calculate_diff_method(device, closed_jaxpr)


def load_device_capabilities(device) -> DeviceCapabilities:
    """Get the contents of the device config toml file."""
    return _load_device_capabilities(device)


def requires_shots(capabilities):
    """Checks if a device capabilities requires shots."""
    return _requires_shots(capabilities)


def filter_device_capabilities_with_shots(
    capabilities, shots_present, unitary_support=None
) -> DeviceCapabilities:
    """Process device capabilities depending on shots and unitary support."""
    return _filter_device_capabilities_with_shots(capabilities, shots_present, unitary_support)


def get_device_capabilities(
    device,
    execution_config,
    shots_len=None,
) -> DeviceCapabilities:
    """Load and filter device capabilities for decomposition/compilation."""
    capabilities = load_device_capabilities(device)

    if execution_config is None:
        execution_config = qml.devices.ExecutionConfig()

    if shots_len == 0 and requires_shots(capabilities):
        raise CompileError(
            textwrap.dedent(
                f"""
                {device.name} does not support analytical simulation.
                Please supply the number of shots on the qnode.
                """
            )
        )
    capabilities = filter_device_capabilities_with_shots(
        capabilities=capabilities,
        shots_present=(shots_len > 0),
        unitary_support=getattr(device, "_to_matrix_ops", None),
    )

    return capabilities
