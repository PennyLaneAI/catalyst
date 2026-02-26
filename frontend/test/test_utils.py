# Copyright 2024-2025 Xanadu Quantum Technologies Inc.

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
Test utilities for device capabilities management.

This module provides helper functions for tests that need to construct custom device
capabilities. The function `get_device_capabilities` was moved here from the main codebase
as it is only used by tests.
"""

from pennylane.devices.capabilities import DeviceCapabilities

from catalyst.device.qjit_device import (
    QJITDevice,
    _load_device_capabilities,
    filter_device_capabilities_with_shots,
)


def get_device_capabilities(device, shots=False) -> DeviceCapabilities:
    """
    Get the capabilities from a device for testing purposes.

    This function is a test utility that allows tests to create custom device capabilities
    by loading capabilities from a base device (typically lightning.qubit) and then modifying
    them as needed for the specific test case.

    Args:
        device: A PennyLane device (should not be a QJITDevice instance)
        shots: Whether shots are present in the program (default: False)

    Returns:
        DeviceCapabilities: The device capabilities filtered based on whether shots are present

    Note:
        Tests typically use this function by calling it on a lightning device and then manually
        modifying the resulting capabilities (e.g., removing or adding operations) to create
        custom test scenarios. This approach piggy-backs off lightning's "full capabilities".
        In the future, tests should ideally construct their capabilities directly instead.
    """

    assert not isinstance(device, QJITDevice)

    return filter_device_capabilities_with_shots(
        _load_device_capabilities(device), bool(shots), getattr(device, "_to_matrix_ops", None)
    )
